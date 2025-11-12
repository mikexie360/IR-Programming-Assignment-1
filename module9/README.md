# IR-Programming-Assignment-1

Create a python virtual environment
```
python -m venv env
```

Activate the python virtual environment
```
env\Scripts\activate
```

deactivate the python virtual environment
```
deactivate
```

freeze python requirements
```
pip freeze > requirements.txt
```

install python requirements
```
pip install -r requirements.txt
python -m pip install -r .\requirements.txt

```

Program 5 – Near-duplicate clustering (MinHash/TF‑IDF)

Quick start (Windows PowerShell):

1) Activate your venv (from the module9 folder):
```
env\Scripts\Activate.ps1
```

2) Run both algorithms (default) on the 2k test set and evaluate:
```
python program5.py --input .\twoktests\twok.tsv --outdir .\outputs --size twok --gold .\twoktests\gold-twok.txt
```

Run a single algorithm explicitly:

MinHash+LSH:
```
python program5.py --input .\twoktests\twok.tsv --algo minhash --outdir .\outputs --size twok --gold .\twoktests\gold-twok.txt
```

Useful MinHash flags (optional):
- `--k` word shingle size (default 5)
- `--num-perm` number of MinHash permutations (default 100)
- `--bands` LSH bands and `--rows-per-band` rows per band (must multiply to num-perm)
- `--jaccard-thr` Jaccard threshold to link docs in single-link clustering (default 0.8)

3) Run TF‑IDF cosine clustering (no sklearn required):
```
python program5.py --input .\twoktests\twok.tsv --algo tfidf --tf-analyzer char --tf-min 3 --tf-max 5 --cosine-thr 0.90 --outdir .\outputs --size twok --gold .\twoktests\gold-twok.txt
```

TF‑IDF flags:
- `--tf-analyzer` char|word (default char)
- `--tf-min/--tf-max` n‑gram sizes (default 3..5)
- `--cosine-thr` cosine similarity threshold to link docs (default 0.90)

Output & logs:
- Clusters file: `outputs/<algo>/<JHED>-<size>.txt` (one line per cluster, space-separated docids; each doc appears once; no blank lines)
- Log JSON: `outputs/<algo>/<JHED>-<size>.log.json` with runtime, peak memory, cluster stats, and optional B‑Cubed metrics when `--gold` is provided.

Notes:
- By default, both algorithms run (`--algo all`). Use `--algo minhash` or `--algo tfidf` to run just one.
- The TF‑IDF implementation is self-contained (pure Python/NumPy). No scikit‑learn is used.


```
python program5.py --input .\duplicatetests\hundred.tsv --outdir .\outputs --size hundred
python program5.py --input .\duplicatetests\hundredk.tsv --algo minhash --outdir .\outputs --size hundredk
python program5.py --input .\duplicatetests\onek.tsv --outdir .\outputs --size onek
python program5.py --input .\duplicatetests\tenk.tsv --algo minhash --outdir .\outputs --size tenk
python program5.py --input .\duplicatetests\thirty.tsv --outdir .\outputs --size thirty
python program5.py --input .\duplicatetests\thirtyk.tsv --algo minhash --outdir .\outputs --size thirtyk
python program5.py --input .\duplicatetests\threehundred.tsv --outdir .\outputs --size threehundred
python program5.py --input .\duplicatetests\threek.tsv --algo minhash --outdir .\outputs --size threek
```