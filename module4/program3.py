from pathlib import Path
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from rich import print as rich_print
from collections import Counter
import argparse
import json
import struct
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable
import math
import time
import re
import atexit

# lowercase + keep only a–z and digits (drops punctuation, quotes, hyphens)
TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9]+")  # faster than NLTK for large corpora
TAG_RE = re.compile(r"<[^>]+>")      # remove any residual tags inside P if present

def normalize(text: str) -> List[str]:
    # return TOKENIZER.tokenize(text.lower())  # original NLTK
    return TOKEN_RE.findall(text.lower())      # faster pure-regex

# ---- Logging wrapper: append every printed line to log.txt ----
_LOG_FH = None

def _get_log_fh():
    global _LOG_FH
    if _LOG_FH is None:
        _LOG_FH = open("log.txt", mode="a", encoding="utf-8", errors="ignore")
    return _LOG_FH

def _close_log_fh():
    global _LOG_FH
    try:
        if _LOG_FH is not None:
            _LOG_FH.flush()
            _LOG_FH.close()
    finally:
        _LOG_FH = None

atexit.register(_close_log_fh)

def print(*args, **kwargs):
    """Proxy print that writes to console (rich) and appends to log.txt.
    Respects sep/end if provided; ignores file/flush for the log side (we flush always).
    """
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    # Build plain text for log
    text = sep.join(str(a) for a in args) + end
    fh = _get_log_fh()
    fh.write(text)
    fh.flush()
    # Console output with rich formatting
    rich_print(*args, **kwargs)

@dataclass
class Posting:
    doc_id: int
    tf: int

@dataclass
class DictionaryEntry:
    term: str
    df: int
    cf: int
    offset: int  # byte offset in postings (binary file)
    # postings length in bytes is df * (2 * 4) (two ints) but we recompute if needed

class InMemoryIndex:
    """Memory-based inversion structure (non-positional)."""
    def __init__(self):
        # term -> inner mapping: doc_id -> term frequency in that doc
        self._postings: Dict[str, Dict[int, int]] = {}
        self.collection_size: int = 0  # total tokens (with repetition)
        self.vocabulary_size: int = 0
        self.number_of_documents: int = 0

    def add_document(self, doc_id: int, tokens: Iterable[str]):
        self.number_of_documents += 1
        counts = Counter(tokens)  # term frequencies in this doc
        doc_token_total = sum(counts.values())
        self.collection_size += doc_token_total
        for term, tf in counts.items():
            bucket = self._postings.get(term)
            if bucket is None:
                self._postings[term] = {doc_id: tf}
            else:
                # Should not normally collide (doc ids unique) but handle anyway.
                bucket[doc_id] = bucket.get(doc_id, 0) + tf
        self.vocabulary_size = len(self._postings)

    # --- Accessors ---
    def terms(self) -> Iterable[str]:
        return self._postings.keys()

    def postings(self, term: str) -> List[Posting]:
        m = self._postings.get(term, {})
        return [Posting(d, tf) for d, tf in sorted(m.items())]

    def dictionary_stats(self) -> List[Tuple[str, int, int]]:
        out = []
        for term, mapping in self._postings.items():
            df = len(mapping)
            cf = sum(mapping.values())
            out.append((term, df, cf))
        return out

# ---- Parsing ----

def parse_p_docs_tag_soup(path: str) -> InMemoryIndex:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup("<root>" + raw + "</root>", "html.parser")
    index = InMemoryIndex()
    docs_added = 0
    for p in soup.find_all("p"):
        doc_id_val = p.get("id")
        if doc_id_val is None:
            continue
        try:
            doc_id = int(doc_id_val) if str(doc_id_val).isdigit() else int(doc_id_val)
        except ValueError:
            # skip non-integer ids for simplicity (require numeric doc ids for binary format)
            continue
        tokens_list = normalize(p.get_text())
        index.add_document(doc_id, tokens_list)
        docs_added += 1
        if docs_added % 1000 == 0:
            print(f"[dim]Indexed {docs_added} documents...")
    if docs_added and docs_added % 1000 != 0:
        print(f"[dim]Indexed {docs_added} documents (final)...")
    return index

# Fast streaming parser that avoids loading the entire file or using BeautifulSoup
START_P_RE = re.compile(r"<\s*P\s+ID\s*=\s*\"?(\d+)\"?\s*>", re.IGNORECASE)
END_P_RE = re.compile(r"</\s*P\s*>", re.IGNORECASE)

def parse_p_docs_stream(path: str) -> InMemoryIndex:
    index = InMemoryIndex()
    docs_added = 0
    in_doc = False
    doc_id = None
    buf: List[str] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not in_doc:
                m = START_P_RE.search(line)
                if m:
                    doc_id = int(m.group(1))
                    in_doc = True
                    # collect any text on the same line after the start tag
                    after = line[m.end():]
                    buf.append(after)
                continue
            # in_doc
            if END_P_RE.search(line):
                # up to just before the end tag
                end_pos = END_P_RE.search(line).start()
                buf.append(line[:end_pos])
                text = ''.join(buf)
                # strip any inner tags and tokenize
                text = TAG_RE.sub(' ', text)
                tokens_list = normalize(text)
                if doc_id is not None:
                    index.add_document(doc_id, tokens_list)
                    docs_added += 1
                    if docs_added % 1000 == 0:
                        print(f"[dim]Indexed {docs_added} documents...")
                # reset state
                in_doc = False
                doc_id = None
                buf.clear()
            else:
                buf.append(line)
    if docs_added and docs_added % 1000 != 0:
        print(f"[dim]Indexed {docs_added} documents (final)...")
    return index

# ---- Index Writing (Dictionary JSON + Binary Postings) ----

INT_STRUCT = struct.Struct('<i')  # little-endian 4-byte signed int
PAIR_STRUCT = struct.Struct('<ii')  # doc_id, tf
DOC_LEN_STRUCT = struct.Struct('<if')  # doc_id:int32, length:float32

@dataclass
class WrittenIndexMetadata:
    dictionary_path: str
    postings_path: str
    postings_count_bytes: int


def write_index(index: InMemoryIndex, out_dir: str, dictionary_filename: str = 'dictionary.json', postings_filename: str = 'inverted.bin') -> WrittenIndexMetadata:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dict_path = Path(out_dir) / dictionary_filename
    post_path = Path(out_dir) / postings_filename

    # Sort terms lexicographically
    sorted_terms = sorted(index.terms())

    dictionary: Dict[str, Dict[str, int]] = {}
    offset = 0
    with open(post_path, 'wb') as pb:
        for term in sorted_terms:
            posts = index.postings(term)  # sorted by doc id
            df = len(posts)
            cf = sum(p.tf for p in posts)
            # Write postings sequentially: doc_id(int32), tf(int32) * df
            for p in posts:
                pb.write(PAIR_STRUCT.pack(p.doc_id, p.tf))
            entry = {
                'df': df,
                'cf': cf,
                'offset': offset  # byte offset where this term's postings start
            }
            dictionary[term] = entry
            offset += df * PAIR_STRUCT.size

    with open(dict_path, 'w', encoding='utf-8') as dj:
        json.dump({
            'stats': {
                'documents': index.number_of_documents,
                'vocab_size': index.vocabulary_size,
                'collection_size': index.collection_size
            },
            'terms': dictionary
        }, dj, ensure_ascii=False, indent=2)

    postings_size = post_path.stat().st_size
    return WrittenIndexMetadata(str(dict_path), str(post_path), postings_size)

# ---- Reporting ----

def report_stats(index: InMemoryIndex):
    print(f"{'Number of paragraphs processed:':<40} {index.number_of_documents:>10}")
    print(f"{'Vocab size (number of unique words):':<40} {index.vocabulary_size:>10}")
    print(f"{'Collection size (total number of words):':<40} {index.collection_size:>10}")

    # produce top 100 tokens by collection frequency
    entries = []
    for term in index.terms():
        posts = index.postings(term)
        cf = sum(p.tf for p in posts)
        df = len(posts)
        entries.append((term, cf, df))
    entries.sort(key=lambda x: (-x[1], x[0]))

    top_k = min(100, len(entries))
    print("\n[bold yellow]Top 100 tokens")
    print(f"{'Rank':<5}  {'Token':<15}  {'Collection Frequency':<22}  {'Document Frequency':>18}")
    for rank, (term, cf, df) in enumerate(entries[:top_k], start=1):
        print(f"{rank:<5}  {term:<15}  {cf:<22}  {df:>18}")

    def print_rank(r: int):
        if r <= len(entries):
            term, cf, df = entries[r - 1]
            print(f"~{r}\t{term}\tCF={cf}\tDF={df}")
        else:
            print(f"~{r}\tN/A (vocab size = {len(entries)})")

    print_rank(500)
    print_rank(1000)
    print_rank(5000)

    df_eq_1 = sum(1 for _, _, df in entries if df == 1)
    pct = (df_eq_1 / len(entries) * 100.0) if entries else 0.0
    print(f"Terms with DocumentFrequency=1 (occur in only one document): {df_eq_1}")
    print(f"Percent of terms with only one doc: {df_eq_1} / {len(entries)} ({pct:.2f}%)")

# ---- Loading and Access Helpers ----

def load_dictionary(dict_path: str) -> dict:
    return json.loads(Path(dict_path).read_text(encoding='utf-8'))

def read_postings(postings_path: str, offset: int, df: int) -> List[Posting]:
    posts: List[Posting] = []
    with open(postings_path, 'rb') as f:
        f.seek(offset)
        for _ in range(df):
            raw = f.read(PAIR_STRUCT.size)
            if len(raw) < PAIR_STRUCT.size:
                break
            doc_id, tf = PAIR_STRUCT.unpack(raw)
            posts.append(Posting(doc_id, tf))
    return posts

# ---- Document Lengths (for cosine) ----

def compute_and_write_doc_lengths(dictionary_path: str, postings_path: str, out_path: str) -> int:
    """Compute L2 length of each document using w_td = (1 + log(tf)) * idf, idf = log(N/df). Write binary pairs <doc_id:int32, length:float32>.
    Returns number of doc entries written.
    """
    d = load_dictionary(dictionary_path)
    N = int(d['stats']['documents'])
    terms = d['terms']

    sums: Dict[int, float] = {}
    with open(postings_path, 'rb') as f:
        for term, meta in terms.items():
            df = int(meta['df'])
            if df == 0:
                continue
            idf = math.log(N / df)
            offset = int(meta['offset'])
            # read df postings starting at offset
            f.seek(offset)
            for _ in range(df):
                raw = f.read(PAIR_STRUCT.size)
                if len(raw) < PAIR_STRUCT.size:
                    break
                doc_id, tf = PAIR_STRUCT.unpack(raw)
                # weight
                w_td = (1.0 + math.log(tf)) * idf if tf > 0 else 0.0
                prev = sums.get(doc_id, 0.0)
                sums[doc_id] = prev + (w_td * w_td)

    # finalize and write binary
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as outb:
        for doc_id in sorted(sums.keys()):
            length = math.sqrt(sums[doc_id])
            outb.write(DOC_LEN_STRUCT.pack(doc_id, float(length)))
    return len(sums)

def load_doc_lengths(path: str) -> Dict[int, float]:
    m: Dict[int, float] = {}
    if not Path(path).exists():
        return m
    data = Path(path).read_bytes()
    size = DOC_LEN_STRUCT.size
    for i in range(0, len(data), size):
        doc_id, length = DOC_LEN_STRUCT.unpack(data[i:i+size])
        m[doc_id] = float(length)
    return m

# ---- Query Parsing (topics) ----

def parse_topics_file(path: str) -> List[Tuple[int, Counter]]:
    """Parse a topics file with blocks like <Q ID=123> ... </Q>, return list of (qid, Counter(tokens)). Case-insensitive tags.
    """
    raw = Path(path).read_text(encoding='utf-8', errors='ignore')
    pattern = re.compile(r"<\s*Q\s+ID\s*=\s*([0-9]+)\s*>\s*(.*?)\s*<\s*/\s*Q\s*>", re.IGNORECASE | re.DOTALL)
    out: List[Tuple[int, Counter]] = []
    for m in pattern.finditer(raw):
        qid = int(m.group(1))
        content = m.group(2)
        toks = normalize(content)
        out.append((qid, Counter(toks)))
    return out

# ---- Cosine Retrieval ----

def cosine_search(dictionary_path: str, postings_path: str, doclengths_path: str, topics_path: str, output_path: str, jhed: str = 'JHED', k: int = 1000):
    d = load_dictionary(dictionary_path)
    stats = d['stats']
    N = int(stats['documents'])
    term_meta: Dict[str, Dict[str, int]] = d['terms']
    doc_lengths = load_doc_lengths(doclengths_path)

    # parse queries
    queries = parse_topics_file(topics_path)
    if not queries:
        print(f"[bold red]No queries parsed from {topics_path}")
        return

    # For printing weighted terms of only the first query
    def compute_query_weights(qtf: Counter) -> Dict[str, float]:
        wq: Dict[str, float] = {}
        for term, tf in qtf.items():
            meta = term_meta.get(term)
            if not meta:
                continue
            df = int(meta['df'])
            if df == 0:
                continue
            idf = math.log(N / df)
            wq[term] = (1.0 + math.log(tf)) * idf if tf > 0 else 0.0
        return wq

    # open postings only once for repeated seeks
    with open(postings_path, 'rb') as pf, open(output_path, 'w', encoding='utf-8') as outf:
        # Header for ranking output
        header = "qid Q0 docid rank score JHED"
        print("[bold cyan]Ranking output (six columns):")
        print(header)
        outf.write(header + "\n")

        # Weighted terms for the first query
        first_qid, first_qtf = queries[0]
        first_wq = compute_query_weights(first_qtf)
        if first_wq:
            pretty = sorted(((t, first_wq[t]) for t in first_wq), key=lambda x: (-x[1], x[0]))
            show = ", ".join(f"({t}, {w:.4f})" for t, w in pretty)
            print(f"[bold cyan]First query weighted terms: {show}")
        else:
            print("[bold cyan]First query weighted terms: (none of the terms appear in corpus)")

        for qid, qtf in queries:
            # compute query weights and query length
            wq = compute_query_weights(qtf)
            if not wq:
                continue
            query_len = math.sqrt(sum(w * w for w in wq.values()))
            scores: Dict[int, float] = {}

            # loop query terms; accumulate dot products
            for term, wqt in wq.items():
                meta = term_meta.get(term)
                if not meta:
                    continue
                df = int(meta['df'])
                if df == 0:
                    continue
                idf = math.log(N / df)
                offset = int(meta['offset'])
                # scan postings for this term
                pf.seek(offset)
                for _ in range(df):
                    raw = pf.read(PAIR_STRUCT.size)
                    if len(raw) < PAIR_STRUCT.size:
                        break
                    doc_id, tf = PAIR_STRUCT.unpack(raw)
                    w_td = (1.0 + math.log(tf)) * idf if tf > 0 else 0.0
                    scores[doc_id] = scores.get(doc_id, 0.0) + (w_td * wqt)

            # normalize by |d| and |q|, with progress every 1000 documents
            ranked: List[Tuple[int, float]] = []
            processed = 0
            for doc_id, dot in scores.items():
                dl = doc_lengths.get(doc_id, 0.0)
                if dl == 0.0 or query_len == 0.0:
                    continue
                ranked.append((doc_id, dot / (dl * query_len)))
                processed += 1
                if processed % 1000 == 0:
                    print(f"[dim]Query {qid}: Scored {processed} documents...")
            if processed and processed % 1000 != 0:
                print(f"[dim]Query {qid}: Scored {processed} documents (final)...")

            # sort and write top-k
            ranked.sort(key=lambda x: (-x[1], x[0]))
            for rank, (doc_id, score) in enumerate(ranked[:k], start=1):
                line = f"{qid} Q0 {doc_id} {rank} {score:.6f} {jhed}"
                outf.write(line + "\n")
                print(line)

    print(f"[bold green]Wrote rankings to {output_path}")

# ---- Main CLI ----

def main():
    parser = argparse.ArgumentParser(description='Build a non-positional inverted index and/or run cosine retrieval.')
    parser.add_argument('--input', default='rfa.txt', help='Input corpus file containing <p id=...> elements.')
    parser.add_argument('--out-dir', default='index', help='Directory to place dictionary and postings files.')
    parser.add_argument('--dict-name', default='dictionary.json', help='Dictionary JSON filename.')
    parser.add_argument('--postings-name', default='inverted.bin', help='Binary postings filename.')
    parser.add_argument('--doclengths-name', default='doclengths.bin', help='Binary doc lengths filename.')
    parser.add_argument('--fast', action='store_true', help='Use fast streaming parser and regex tokenizer (recommended for large files).')
    # Retrieval args
    parser.add_argument('--queries', help='Topics file (e.g., animal.topics.txt or cord19.topics.keyword.txt). If provided, runs retrieval.')
    parser.add_argument('--output', help='Output rankings file (six-field TREC format). Required when --queries is set.')
    parser.add_argument('--jhed', default='mxie17', help='JHED id to place in rankings output (last column).')
    parser.add_argument('--k', type=int, default=1000, help='Top-k documents per query to output.')

    args = parser.parse_args()

    start = time.perf_counter()
    print(f"[bold cyan]Parsing and indexing {args.input} ...")
    if args.fast:
        index = parse_p_docs_stream(args.input)
    else:
        index = parse_p_docs_tag_soup(args.input)

    print("[bold cyan]Writing index files ...")
    meta = write_index(index, args.out_dir, args.dict_name, args.postings_name)

    # File size comparisons
    dict_size = Path(meta.dictionary_path).stat().st_size
    postings_size = Path(meta.postings_path).stat().st_size
    original_size = Path(args.input).stat().st_size if Path(args.input).exists() else 0

    report_stats(index)

    print("\n[bold magenta]File Size Analysis")
    print(f"Dictionary JSON size (bytes): {dict_size}")
    print(f"Inverted file size (bytes): {postings_size}")
    print(f"Original text size (bytes): {original_size}")
    total_index = dict_size + postings_size
    print(f"Total index size (dict + postings): {total_index}")
    if original_size:
        ratio = total_index / original_size * 100
        print(f"Index is {ratio:.2f}% of original text size.")
        print("Index size + dictionary size smaller than original text?", 'YES' if total_index < original_size else 'NO')
    print("Dictionary vs Postings: larger component is:", 'dictionary' if dict_size > postings_size else 'postings' if postings_size > dict_size else 'equal')

    # Compute and write doc lengths for cosine
    doclen_path = str(Path(args.out_dir) / args.doclengths_name)
    print("[bold cyan]Computing document vector lengths (cosine)...")
    count_docs = compute_and_write_doc_lengths(str(Path(args.out_dir) / args.dict_name), str(Path(args.out_dir) / args.postings_name), doclen_path)
    print(f"Doc lengths written for {count_docs} documents -> {doclen_path} (bytes: {Path(doclen_path).stat().st_size})")

    end = time.perf_counter()
    print(f"[bold green]Index build + doclengths time: {(end - start):.2f} seconds")

    # Retrieval stage if queries provided
    if args.queries:
        if not args.output:
            raise SystemExit("--output is required when --queries is provided")
        print(f"[bold cyan]Running cosine retrieval for topics in {args.queries} ...")
        cosine_search(
            dictionary_path=str(Path(args.out_dir) / args.dict_name),
            postings_path=str(Path(args.out_dir) / args.postings_name),
            doclengths_path=doclen_path,
            topics_path=args.queries,
            output_path=args.output,
            jhed=args.jhed,
            k=args.k
        )

if __name__ == '__main__':
    main()