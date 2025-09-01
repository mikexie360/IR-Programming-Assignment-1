from pathlib import Path
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from rich import print

# lowercase + keep only a–z and digits (drops punctuation, quotes, hyphens)
TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")

def normalize(text: str):
    return TOKENIZER.tokenize(text.lower())  # returns a LIST

class Token:
    def __init__(self, token: str = "", collection_frequency: int = 0,
                 document_frequency: int = 0, document_listing=None):
        self.token = token
        self.collection_frequency = collection_frequency
        self.document_frequency = document_frequency
        self.document_listing = list(document_listing) if document_listing else []

    def incrementCollectionFrequency(self):
        self.collection_frequency += 1

    def incrementDocumentFrequency(self):
        self.document_frequency += 1

    def addDocumentListing(self, newDoc: int):
        self.document_listing.append(newDoc)

class DataStructure:
    def __init__(self):
        self.tokens: dict[str, Token] = {}
        self.vocabulary_size = 0
        self.collection_size = 0
        self.number_of_documents = 0

    def addDocument(self, doc_id: int, tokens: list[str]):
        self.number_of_documents +=1
        seen_in_doc = set()  # avoid bumping DF multiple times per doc
        for t in tokens:
            self.collection_size += 1
            tok = self.tokens.get(t)
            if tok is None:
                self.tokens[t] = Token(t, 1, 1, [doc_id])
                self.vocabulary_size += 1
                seen_in_doc.add(t)
            else:
                tok.incrementCollectionFrequency()
                if t not in seen_in_doc and doc_id not in tok.document_listing:
                    tok.addDocumentListing(doc_id)
                    tok.incrementDocumentFrequency()
                    seen_in_doc.add(t)

def parse_p_docs_tag_soup(path: str):
    # Tolerant parse: treat your fragments as HTML
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup("<root>" + raw + "</root>", "html.parser")  # use "html5lib" if installed

    ds = DataStructure()
    for p in soup.find_all("p"):
        doc_id = p.get("id")
        print(f"parsing doc id: [bold yellow]{doc_id}")
        if doc_id is None:
            continue
        doc_id = int(doc_id) if str(doc_id).isdigit() else doc_id
        text = p.get_text()
        tokens_list = normalize(text)        # <-- already a list; DO NOT .split()
        ds.addDocument(doc_id, tokens_list)
    return ds

def report_stats(ds: DataStructure):
    print(f"{'Number of paragraphs processed:':<20} {ds.number_of_documents:>10}" )
    print(f"{'Vocab size (number of unique words):':<20} {ds.vocabulary_size:>10}")
    print(f"{'Collection size (total number of words):':<20} {ds.collection_size:>10}")
    """Print top-100 by collection frequency, then ranks 500/1000/5000, and DF==1 stats."""
    # Sort terms: highest CF first, tie-break by term for deterministic output
    items = list(ds.tokens.items())  # [(term, Token)]
    sorted_items = sorted(items, key=lambda kv: (-kv[1].collection_frequency, kv[0]))
    vocab_size = len(sorted_items)

    # Top 100 lines: rank, term, CF, DF
    top_k = min(100, vocab_size)
    print()
    print(f"[bold yellow]Top 100 tokens")
    print(f"{'Rank':<5}  {'Token':<10}  {'Collection Frequency':<25}  {'Document Frequency':>25}")
    for rank, (term, tok) in enumerate(sorted_items[:top_k], start=1):
        print(f"{rank:<5}  {term:<10}  {tok.collection_frequency:<25}  {tok.document_frequency:>25}")
    
    # Also print the 500th, 1000th, 5000th ranked terms (if they exist)
    def print_rank(r):
        if r <= vocab_size:
            term, tok = sorted_items[r - 1]   # 1-based rank
            print(f"~{r}\t{term}\tCF={tok.collection_frequency}\tDF={tok.document_frequency}")
        else:
            print(f"~{r}\tN/A (vocab size = {vocab_size})")

    print_rank(500)
    print_rank(1000)
    print_rank(5000)

    # Words that occur in only one document (DF == 1)
    df_eq_1 = sum(1 for _, tok in items if tok.document_frequency == 1)
    pct = (df_eq_1 / vocab_size * 100.0) if vocab_size else 0.0
    print(f"Terms with DocumentFrequency=1 (occur in only one document): {df_eq_1}")
    print(f"Percent of terms with only one doc: {df_eq_1} / {vocab_size} ({pct:.2f}%)")

if __name__ == "__main__":
    data1 = parse_p_docs_tag_soup("rfa.txt")
    data2 = parse_p_docs_tag_soup("sense.txt")

    print("------------------------------")
    print("[bold red]rfa.txt data")
    report_stats(data1)
    print("------------------------------")
    print("[bold red]sense.txt data")
    report_stats(data2)