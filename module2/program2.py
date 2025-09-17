from pathlib import Path
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from rich import print
from collections import Counter
import argparse
import json
import struct
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable

# lowercase + keep only a–z and digits (drops punctuation, quotes, hyphens)
TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")

def normalize(text: str) -> List[str]:
    return TOKENIZER.tokenize(text.lower())  # returns a LIST

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
    return index

# ---- Index Writing (Dictionary JSON + Binary Postings) ----

INT_STRUCT = struct.Struct('<i')  # little-endian 4-byte signed int
PAIR_STRUCT = struct.Struct('<ii')  # doc_id, tf

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

# ---- Loading and Access Helpers (used by test2.py) ----

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

# ---- Main CLI ----

def main():
    parser = argparse.ArgumentParser(description='Build a non-positional inverted index (memory-based).')
    parser.add_argument('--input', default='rfa.txt', help='Input corpus file containing <p id=...> elements.')
    parser.add_argument('--out-dir', default='index', help='Directory to place dictionary and postings files.')
    parser.add_argument('--dict-name', default='dictionary.json', help='Dictionary JSON filename.')
    parser.add_argument('--postings-name', default='inverted.bin', help='Binary postings filename.')
    args = parser.parse_args()

    print(f"[bold cyan]Parsing and indexing {args.input} ...")
    index = parse_p_docs_tag_soup(args.input)

    print("[bold cyan]Writing index files ...")
    meta = write_index(index, args.out_dir, args.dict_name, args.postings_name)

    # File size comparisons
    dict_size = Path(meta.dictionary_path).stat().st_size
    postings_size = Path(meta.postings_path).stat().st_size
    original_size = Path(args.input).stat().st_size if Path(args.input).exists() else 0

    report_stats(index)

    print("\n[bold magenta]Index File Sizes")
    print(f"Dictionary JSON size (bytes): {dict_size}")
    print(f"Inverted file size (bytes): {postings_size}")
    print(f"Original text size (bytes): {original_size}")
    total_index = dict_size + postings_size
    print(f"Total index size (dict + postings): {total_index}")
    if original_size:
        ratio = total_index / original_size * 100
        print(f"Index is {ratio:.2f}% of original text size.")
        print("Index smaller than original text?", 'YES' if total_index < original_size else 'NO')
    print("Dictionary vs Postings: larger component is:", 'dictionary' if dict_size > postings_size else 'postings' if postings_size > dict_size else 'equal')

if __name__ == '__main__':
    main()