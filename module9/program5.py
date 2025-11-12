from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


DocId = str


def read_tsv(path: Path) -> Tuple[List[DocId], List[str]]:
    """Read TSV file with format: docid<TAB>text."""
    ids: List[DocId] = []
    texts: List[str] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                # Skip malformed lines silently
                continue
            ids.append(parts[0])
            texts.append(parts[1])
    return ids, texts


def normalize_text(text: str) -> List[str]:
    # Uppercase and keep alphanumerics as tokens; collapse spaces
    text = text.upper()
    # Replace non-alphanum with space
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    tokens = text.split()
    return tokens


def word_shingles(tokens: Sequence[str], k: int) -> Set[str]:
    if k <= 0:
        return set()
    return {" ".join(tokens[i : i + k]) for i in range(0, max(0, len(tokens) - k + 1))}


def crc32_int(s: str) -> int:
    import binascii

    return binascii.crc32(s.encode("utf-8")) & 0xFFFFFFFF


class MinHasher:
    PRIME = 4294967311  # > 2^32 large prime

    def __init__(self, num_perm: int, seed: int = 42):
        self.num_perm = num_perm
        rng = random.Random(seed)
        # a in [1, PRIME-1], b in [0, PRIME-1]
        self.coeffs = [(rng.randrange(1, self.PRIME - 1), rng.randrange(0, self.PRIME - 1)) for _ in range(num_perm)]

    def signature(self, hashed_shingles: Iterable[int]) -> np.ndarray:
        sig = np.full(self.num_perm, fill_value=self.PRIME, dtype=np.uint64)
        for x in hashed_shingles:
            xv = int(x)
            for i, (a, b) in enumerate(self.coeffs):
                val = (a * xv + b) % self.PRIME
                if val < sig[i]:
                    sig[i] = val
        return sig


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass
class MinHashParams:
    k: int = 5  # word shingle size
    num_perm: int = 100  # number of hash functions
    bands: int = 20  # number of bands
    rows_per_band: int = 5  # rows per band (bands * rows_per_band must equal num_perm)
    jaccard_threshold: float = 0.8  # edge threshold for single-link


def lsh_candidate_pairs(signatures: np.ndarray, bands: int, rows: int) -> Set[Tuple[int, int]]:
    n_docs, sig_len = signatures.shape
    assert sig_len == bands * rows, "num_perm must equal bands * rows_per_band"
    candidates: Set[Tuple[int, int]] = set()
    for b in range(bands):
        start = b * rows
        end = start + rows
        buckets: Dict[Tuple[int, ...], List[int]] = collections.defaultdict(list)
        for i in range(n_docs):
            key = tuple(int(x) for x in signatures[i, start:end])
            buckets[key].append(i)
        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                # all pairs within the bucket
                bd = bucket_docs
                for i in range(len(bd)):
                    for j in range(i + 1, len(bd)):
                        candidates.add((bd[i], bd[j]))
    return candidates


def compute_clusters_minhash(ids: List[DocId], texts: List[str], params: MinHashParams) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
    # Step 1: shingles
    shingle_sets: List[Set[str]] = []
    hashed_sets: List[Set[int]] = []
    for t in texts:
        tokens = normalize_text(t)
        sset = word_shingles(tokens, params.k)
        shingle_sets.append(sset)
        hashed_sets.append({crc32_int(s) for s in sset})

    # Step 2: MinHash signatures
    assert params.num_perm == params.bands * params.rows_per_band, "num_perm must equal bands * rows_per_band"
    m = MinHasher(num_perm=params.num_perm)
    sigs = np.vstack([m.signature(hs) for hs in hashed_sets])  # shape: (n_docs, num_perm)

    # Step 3: LSH to get candidate pairs
    cand_pairs = lsh_candidate_pairs(sigs, params.bands, params.rows_per_band)

    # Step 4: Filter candidates by (approx or exact) Jaccard and union-find
    uf = UnionFind(len(ids))
    accepted_edges = 0
    for i, j in cand_pairs:
        a, b = shingle_sets[i], shingle_sets[j]
        if not a or not b:
            continue
        inter = len(a & b)
        if inter == 0:
            continue
        jac = inter / float(len(a | b))
        if jac >= params.jaccard_threshold:
            uf.union(i, j)
            accepted_edges += 1

    # Step 5: Build clusters (singletons included)
    clusters_dict: Dict[int, List[int]] = collections.defaultdict(list)
    for idx in range(len(ids)):
        clusters_dict[uf.find(idx)].append(idx)
    clusters = list(clusters_dict.values())
    # Sort indices within clusters for stable output
    for c in clusters:
        c.sort(key=lambda i: (int(ids[i]) if ids[i].isdigit() else ids[i]))

    stats = {"candidate_pairs": len(cand_pairs), "accepted_edges": accepted_edges}
    return clusters, stats, {"num_shingles_nonempty": sum(1 for s in shingle_sets if s), "avg_shingle_set_size": int(sum(len(s) for s in shingle_sets) / max(1, len(shingle_sets)))}


@dataclass
class TFIDFParams:
    analyzer: str = "char"
    ngram_min: int = 3
    ngram_max: int = 5
    cosine_threshold: float = 0.9

def _doc_term_counts(text: str, analyzer: str, nmin: int, nmax: int) -> Dict[str, int]:
    cnt: Dict[str, int] = collections.Counter()
    if analyzer == "word":
        toks = normalize_text(text)
        for n in range(nmin, nmax + 1):
            if n <= 0:
                continue
            for i in range(0, max(0, len(toks) - n + 1)):
                gram = " ".join(toks[i : i + n])
                cnt[gram] += 1
    else:  # char ngrams over normalized string with spaces
        toks = normalize_text(text)
        norm = " ".join(toks)
        for n in range(nmin, nmax + 1):
            if n <= 0:
                continue
            for i in range(0, max(0, len(norm) - n + 1)):
                gram = norm[i : i + n]
                cnt[gram] += 1
    return cnt


def _build_tfidf(texts: List[str], analyzer: str, nmin: int, nmax: int) -> Tuple[List[Dict[int, float]], Dict[str, int]]:
    term_counts: List[Dict[str, int]] = []
    df: Dict[str, int] = collections.Counter()
    for t in texts:
        counts = _doc_term_counts(t, analyzer, nmin, nmax)
        term_counts.append(counts)
        for term in counts.keys():
            df[term] += 1

    vocab: Dict[str, int] = {}
    for term in df.keys():
        vocab[term] = len(vocab)

    N = len(texts)
    idf = np.zeros(len(vocab), dtype=np.float64)
    for term, idx in vocab.items():
        idf[idx] = math.log((N + 1) / (df[term] + 1)) + 1.0

    docs_vec: List[Dict[int, float]] = []
    for counts in term_counts:
        vec: Dict[int, float] = {}
        for term, tf in counts.items():
            idx = vocab[term]
            vec[idx] = float(tf) * idf[idx]
        norm = math.sqrt(sum(w * w for w in vec.values()))
        if norm > 0:
            for k in list(vec.keys()):
                vec[k] = vec[k] / norm
        docs_vec.append(vec)
    return docs_vec, vocab


def compute_clusters_tfidf(ids: List[DocId], texts: List[str], params: TFIDFParams) -> Tuple[List[List[int]], Dict[str, int]]:
    docs_vec, vocab = _build_tfidf(texts, params.analyzer, params.ngram_min, params.ngram_max)
    postings: Dict[int, List[Tuple[int, float]]] = collections.defaultdict(list)
    for di, vec in enumerate(docs_vec):
        for term_idx, w in vec.items():
            postings[term_idx].append((di, w))

    dot: Dict[Tuple[int, int], float] = {}
    for plist in postings.values():
        if len(plist) < 2:
            continue
        for i in range(len(plist)):
            di, wi = plist[i]
            for j in range(i + 1, len(plist)):
                dj, wj = plist[j]
                key = (di, dj) if di < dj else (dj, di)
                dot[key] = dot.get(key, 0.0) + wi * wj

    uf = UnionFind(len(ids))
    accepted = 0
    thr = float(params.cosine_threshold)
    for (i, j), val in dot.items():
        if val >= thr:
            uf.union(i, j)
            accepted += 1

    clusters_dict: Dict[int, List[int]] = collections.defaultdict(list)
    for idx in range(len(ids)):
        clusters_dict[uf.find(idx)].append(idx)
    clusters = list(clusters_dict.values())
    for c in clusters:
        c.sort(key=lambda i: (int(ids[i]) if ids[i].isdigit() else ids[i]))
    stats = {"accepted_edges": accepted, "vocab_size": len(vocab), "pair_candidates": len(dot)}
    return clusters, stats


def clusters_to_mapping(ids: List[DocId], clusters: List[List[int]]) -> Dict[DocId, int]:
    mapping: Dict[DocId, int] = {}
    for cid, idxs in enumerate(clusters):
        for i in idxs:
            mapping[ids[i]] = cid
    return mapping


def load_gold_clusters(path: Path) -> Dict[DocId, int]:
    mapping: Dict[DocId, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docids = line.split()
            cid = len(mapping) + 1  # temporary; will be overwritten for each doc
            for d in docids:
                mapping[d] = cid
    # Reassign to contiguous cluster ids
    remap: Dict[int, int] = {}
    next_id = 0
    out: Dict[DocId, int] = {}
    for d, c in mapping.items():
        if c not in remap:
            remap[c] = next_id
            next_id += 1
        out[d] = remap[c]
    return out


def bcubed(pred: Dict[DocId, int], gold: Dict[DocId, int]) -> Tuple[float, float, float]:
    # Only consider docs present in gold
    docs = [d for d in pred.keys() if d in gold]
    if not docs:
        return 0.0, 0.0, 0.0
    # Build reverse cluster mappings
    pred_rev: Dict[int, Set[DocId]] = collections.defaultdict(set)
    gold_rev: Dict[int, Set[DocId]] = collections.defaultdict(set)
    for d in docs:
        pred_rev[pred[d]].add(d)
        gold_rev[gold[d]].add(d)
    # Per-item precision/recall
    p_sum = 0.0
    r_sum = 0.0
    for d in docs:
        Pc = pred_rev[pred[d]]
        Gc = gold_rev[gold[d]]
        inter = len(Pc & Gc)
        if len(Pc) == 0 or len(Gc) == 0:
            continue
        p_sum += inter / len(Pc)
        r_sum += inter / len(Gc)
    P = p_sum / len(docs)
    R = r_sum / len(docs)
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1


def write_clusters_file(out_path: Path, ids: List[DocId], clusters: List[List[int]]):
    with out_path.open("w", encoding="utf-8") as f:
        for idxs in clusters:
            line = " ".join(ids[i] for i in idxs)
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Program #5: Near Duplicate Detection (clustering)")
    parser.add_argument("--input", required=True, help="Path to TSV file: docid<TAB>text")
    parser.add_argument("--algo", choices=["minhash", "tfidf", "all"], default="all")
    parser.add_argument("--outdir", default="outputs", help="Directory to write outputs into (algo subfolder will be created)")
    parser.add_argument("--size", default="twok", help="Dataset size tag for output filename (e.g., 'twok')")
    parser.add_argument("--jhed", default="mxie17", help="Your JHED to name the output file")
    parser.add_argument("--gold", default=None, help="Optional path to gold clusters file for evaluation (one cluster per line)")
    # MinHash params
    parser.add_argument("--k", type=int, default=5, help="Word shingle size (k)")
    parser.add_argument("--num-perm", type=int, default=100, help="Number of MinHash permutations")
    parser.add_argument("--bands", type=int, default=20, help="Number of LSH bands")
    parser.add_argument("--rows-per-band", type=int, default=5, help="Rows per LSH band")
    parser.add_argument("--jaccard-thr", type=float, default=0.8, help="Jaccard threshold for linking (minhash algo)")
    # TFIDF params
    parser.add_argument("--tf-analyzer", choices=["word", "char"], default="char")
    parser.add_argument("--tf-min", type=int, default=3)
    parser.add_argument("--tf-max", type=int, default=5)
    parser.add_argument("--cosine-thr", type=float, default=0.9, help="Cosine similarity threshold for linking (tfidf algo)")

    args = parser.parse_args()

    input_path = Path(args.input)
    ids, texts = read_tsv(input_path)
    if not ids:
        raise SystemExit("No documents loaded from input TSV.")

    # Resolve JHED (default 'mxie17' or fallback to file if argument empty)
    jhed = args.jhed or (Path(__file__).with_name("JHED.txt").read_text(encoding="utf-8").strip() if Path(__file__).with_name("JHED.txt").exists() else "output")

    def run_one(algo: str):
        tracemalloc.start()
        t0 = time.perf_counter()

        algo_stats: Dict[str, int] = {}
        extra_stats: Dict[str, int] = {}
        if algo == "minhash":
            mh_params = MinHashParams(
                k=args.k,
                num_perm=args.num_perm,
                bands=args.bands,
                rows_per_band=args.rows_per_band,
                jaccard_threshold=args.jaccard_thr,
            )
            clusters, algo_stats, extra_stats = compute_clusters_minhash(ids, texts, mh_params)
        elif algo == "tfidf":
            tf_params = TFIDFParams(
                analyzer=args.tf_analyzer,
                ngram_min=args.tf_min,
                ngram_max=args.tf_max,
                cosine_threshold=args.cosine_thr,
            )
            clusters, algo_stats = compute_clusters_tfidf(ids, texts, tf_params)
        else:
            raise ValueError(f"Unknown algo: {algo}")

        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        pred_map = clusters_to_mapping(ids, clusters)

        out_root = Path(args.outdir) / algo
        out_root.mkdir(parents=True, exist_ok=True)
        out_file = out_root / f"{jhed}-{args.size}.txt"
        write_clusters_file(out_file, ids, clusters)

        eval_metrics = None
        if args.gold:
            gold_map = load_gold_clusters(Path(args.gold))
            P, R, F1 = bcubed(pred_map, gold_map)
            eval_metrics = {"bcubed_precision": P, "bcubed_recall": R, "bcubed_f1": F1}

        log = {
            "algo": algo,
            "input": str(input_path),
            "n_docs": len(ids),
            "runtime_seconds": round(elapsed, 4),
            "peak_memory_mb": round(peak / (1024 * 1024), 2),
            "n_clusters": len(clusters),
            "avg_cluster_size": round(sum(len(c) for c in clusters) / max(1, len(clusters)), 3),
            "algo_stats": algo_stats,
        }
        if extra_stats:
            log.update(extra_stats)
        if algo == "minhash":
            log.update(
                {
                    "k": args.k,
                    "num_perm": args.num_perm,
                    "bands": args.bands,
                    "rows_per_band": args.rows_per_band,
                    "jaccard_threshold": args.jaccard_thr,
                }
            )
        else:
            log.update(
                {
                    "tf_analyzer": args.tf_analyzer,
                    "tf_min": args.tf_min,
                    "tf_max": args.tf_max,
                    "cosine_threshold": args.cosine_thr,
                }
            )
        if eval_metrics:
            log.update(eval_metrics)

        log_path = out_root / f"{jhed}-{args.size}.log.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        print(
            json.dumps(
                {
                    "algo": algo,
                    "wrote": str(out_file),
                    "log": str(log_path),
                    **({} if not eval_metrics else eval_metrics),
                },
                indent=2,
            )
        )

    if args.algo == "all":
        for algo in ("minhash", "tfidf"):
            run_one(algo)
    else:
        run_one(args.algo)


if __name__ == "__main__":
    main()