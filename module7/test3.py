from pathlib import Path
from rich import print
from module4.program3 import load_dictionary, read_postings, PAIR_STRUCT, Posting

TERMS_WITH_POSTINGS = ["panda", "python", "Egyptian", "Wyoming"]
TERMS_DF_ONLY = ["Hopkins", "Stanford", "Brown", "college"]
INTERSECT_TERMS = ["Tony", "Blair"]

def lower(t: str) -> str:
    return t.lower()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test loading the built inverted index.")
    parser.add_argument('--dict', default='index/dictionary.json')
    parser.add_argument('--postings', default='index/inverted.bin')
    args = parser.parse_args()

    if not Path(args.dict).exists() or not Path(args.postings).exists():
        print("[bold red]Dictionary or postings file missing. Run program2.py first.")
        return

    d = load_dictionary(args.dict)
    term_meta = d['terms']

    def get_postings(term: str):
        md = term_meta.get(term)
        if not md:
            return []
        return read_postings(args.postings, md['offset'], md['df'])

    print("[bold cyan]=== Terms with postings (DF + postings list) ===")
    for t in TERMS_WITH_POSTINGS:
        key = lower(t)
        md = term_meta.get(key)
        if not md:
            print(f"{t}: NOT FOUND")
            continue
        posts = get_postings(key)
        df = md['df']
        print(f"{t}: DF={df} [", end="")
        inside = ", ".join(f"({p.doc_id}, {p.tf})" for p in posts)
        print(inside + "]")

    print("\n[bold cyan]=== Terms DF only ===")
    for t in TERMS_DF_ONLY:
        key = lower(t)
        md = term_meta.get(key)
        if not md:
            print(f"{t}: NOT FOUND")
            continue
        print(f"{t}: DF={md['df']}")

    print("\n[bold cyan]=== Intersection: documents containing both 'Tony' and 'Blair' in sorted order ===")
    postings_sets = None
    for t in INTERSECT_TERMS:
        md = term_meta.get(lower(t))
        if not md:
            postings_sets = set()  # empty result
            break
        docs = {p.doc_id for p in read_postings(args.postings, md['offset'], md['df'])}
        postings_sets = docs if postings_sets is None else (postings_sets & docs)

    result = sorted(postings_sets) if postings_sets else []
    print("DocIDs:", result if result else "(none)")

if __name__ == '__main__':
    main()
