from __future__ import annotations
from pathlib import Path
import json
import argparse
import textwrap
import subprocess
import tempfile
import sys
import asyncio
import inspect
import time
from typing import List, Tuple, Dict, Any

# ML stack
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.feature_selection import chi2
    import numpy as np
    from googletrans import Translator

except ImportError:
    raise SystemExit("This script requires scikit-learn and numpy. Install them in your env: pip install scikit-learn numpy")

# IO helpers

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

# Training / Evaluation utilities

def analyze_english_features(train_en: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[str, float, float]]:
    """Return 2*top_k features: positive-leaning and negative-leaning by chi^2,
    along with their document-frequency percentages in pos and neg classes.
    Each tuple is (term, pct_pos, pct_neg).
    """
    texts = [ex['text'] for ex in train_en]
    y = np.array([int(ex['label']) for ex in train_en])

    # Binary presence vectorizer for document frequency measurement
    vbin = TfidfVectorizer(lowercase=True, analyzer='word', ngram_range=(1,3), use_idf=False, norm=None)
    X = vbin.fit_transform(texts)
    X_bin = (X > 0).astype(int)

    # chi2 to rank terms for class 1
    chi2_vals, _ = chi2(X_bin, y)
    order = np.argsort(chi2_vals)[::-1]

    vocab = np.array(vbin.get_feature_names_out())
    pos_terms: List[Tuple[str, float, float]] = []
    neg_terms: List[Tuple[str, float, float]] = []

    pos_mask = (y == 1)
    neg_mask = (y == 0)
    X_pos = X_bin[pos_mask]
    X_neg = X_bin[neg_mask]
    n_pos_docs = X_pos.shape[0]
    n_neg_docs = X_neg.shape[0]

    def df_pct(col: int) -> Tuple[float, float]:
        pos_df = int(X_pos[:, col].sum())
        neg_df = int(X_neg[:, col].sum())
        pos_pct = 100.0 * pos_df / n_pos_docs if n_pos_docs else 0.0
        neg_pct = 100.0 * neg_df / n_neg_docs if n_neg_docs else 0.0
        return pos_pct, neg_pct

    for idx in order:
        term = vocab[idx]
        pos_pct, neg_pct = df_pct(idx)
        if pos_pct < 0.2 and neg_pct < 0.2:
            continue
        if pos_pct > neg_pct and len(pos_terms) < top_k:
            pos_terms.append((term, pos_pct, neg_pct))
        elif neg_pct > pos_pct and len(neg_terms) < top_k:
            neg_terms.append((term, pos_pct, neg_pct))
        if len(pos_terms) >= top_k and len(neg_terms) >= top_k:
            break

    return pos_terms + neg_terms

def train_classifier(train_items: List[Dict[str, Any]], model: str = 'linsvc', analyzer: str = 'word', ngram_range=(1,3)):
    texts = [ex['text'] for ex in train_items]
    y = np.array([int(ex['label']) for ex in train_items])
    vec = TfidfVectorizer(lowercase=True, analyzer=analyzer, ngram_range=ngram_range)
    X = vec.fit_transform(texts)
    if model == 'logreg':
        clf = LogisticRegression(max_iter=20000, solver='liblinear')
    else:
        clf = LinearSVC()
    clf.fit(X, y)
    return vec, clf, X.shape[1]


def predict(items: List[Dict[str, Any]], vec: TfidfVectorizer, clf) -> np.ndarray:
    texts = [ex['text'] for ex in items]
    X = vec.transform(texts)
    yhat = clf.predict(X)
    return yhat


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]

    def prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    p1, r1, f1_1 = prf(tp, fp, fn)
    p0, r0, f1_0 = prf(tn, fn, fp)

    return {
        'confusion_matrix': cm,
        'class_0': {'precision': p0, 'recall': r0, 'f1': f1_0, 'tp': int(tn), 'fp': int(fn), 'fn': int(fp)},
        'class_1': {'precision': p1, 'recall': r1, 'f1': f1_1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)},
    }


def print_prf_details(title: str, metrics: Dict[str, Any]):
    print(f"\n=== {title} ===")
    cm = metrics['confusion_matrix']
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    for cls in ['class_0', 'class_1']:
        m = metrics[cls]
        if cls == 'class_1':
            print(f"{cls}: Precision = {m['tp']} / ({m['tp']} + {m['fp']}) = {m['precision']:.4f}; "
                  f"Recall = {m['tp']} / ({m['tp']} + {m['fn']}) = {m['recall']:.4f}; F1 = {m['f1']:.4f}")
        else:
            print(f"{cls}: Precision = {m['tp']} / ({m['tp']} + {m['fp']}) = {m['precision']:.4f}; "
                  f"Recall = {m['tp']} / ({m['tp']} + {m['fn']}) = {m['recall']:.4f}; F1 = {m['f1']:.4f}")


def print_first_10_predictions(title: str, test_items: List[Dict[str, Any]], y_pred: np.ndarray):
    print(f"\n=== First 10 predictions: {title} ===")
    for i, ex in enumerate(test_items[:10]):
        print(f"{ex['id']}\t{int(y_pred[i])}")


def print_sample_errors(title: str, test_items: List[Dict[str, Any]], y_pred: np.ndarray, max_samples: int = 10):
    print(f"\n=== Sample misclassifications: {title} ===")
    errors = []
    for ex, yp in zip(test_items, y_pred):
        y = int(ex.get('label', -1))
        if y != -1 and y != int(yp):
            errors.append((ex['id'], y, int(yp), ex['text']))
        if len(errors) >= max_samples:
            break
    if not errors:
        print("(no misclassifications in first scan)")
        return
    for i, (docid, y, yp, text) in enumerate(errors, start=1):
        snippet = textwrap.shorten(text.replace('\n', ' '), width=300, placeholder='…')
        print(f"{i}. id={docid} true={y} pred={yp} | {snippet}")

    

# Main pipeline per tasks (a)-(e)

def main():
    parser = argparse.ArgumentParser(description='Program #4: Binary Classification (English/Spanish + Cross-language).')
    parser.add_argument('--data-dir', default='prog4-movies', help='Directory containing the JSONL datasets.')
    parser.add_argument('--model', default='linsvc', choices=['linsvc', 'logreg'], help='Classifier type.')
    # Word n-gram settings for monolingual EN/ES models
    parser.add_argument('--word-ngram-min', type=int, default=1, help='Minimum n for word n-grams (default: 1).')
    parser.add_argument('--word-ngram-max', type=int, default=1, help='Maximum n for word n-grams (default: 1).')
    # Character n-gram size for cross-language model
    parser.add_argument('--char-ngram', type=int, default=4, help='Character n-gram size for cross-language model (e.g., 4).')
    # Cross-language mode: char n-grams (default) or MT ES->EN + English model
    parser.add_argument('--xlang-mode', choices=['char', 'mt'], default='char', help='Cross-language approach: char = character n-grams; mt = translate ES→EN (googletrans) and use English model.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    eng_train = read_jsonl(data_dir / 'eng.imdb.train.jsonl')
    eng_test = read_jsonl(data_dir / 'eng.imdb.test.jsonl')
    spa_train = read_jsonl(data_dir / 'spa.muchocine.train.jsonl')
    spa_test = read_jsonl(data_dir / 'spa.muchocine.test.jsonl')

    # (a) Analyze English training data for indicative features
    print("[TASK (a)] Studying English training data: indicative features (10)")
    feats = analyze_english_features(eng_train, top_k=10)
    for term, pos_pct, neg_pct in feats:
        print(f"{term}\t{pos_pct:.1f}% (pos)\t{neg_pct:.1f}% (neg)")

    # (b) Train English classifier and predict
    print("\n[TASK (b)] Train English classifier and make predictions")
    # Resolve word n-gram range and ensure it is valid
    word_range = (1,3)

    vec_en, clf_en, nfeat_en = train_classifier(eng_train, model=args.model, analyzer='word', ngram_range=word_range)
    y_en_true = np.array([int(x['label']) for x in eng_test])
    y_en_pred = predict(eng_test, vec_en, clf_en)

    print("Approach: TF-IDF word n-grams; analyzer=word; ngram_range=", word_range, "; model=", args.model)
    print("Total features:", nfeat_en)
    print_first_10_predictions("English test", eng_test, y_en_pred)

    # (c) Evaluate English predictions
    print("\n[TASK (c)] Evaluation on English test set")
    en_metrics = evaluate(y_en_true, y_en_pred)
    print_prf_details("English (classes 0 and 1)", en_metrics)
    print_sample_errors("English", eng_test, y_en_pred, max_samples=10)

    # (d) Train Spanish classifier and evaluate
    print("\n[TASK (d)] Train and evaluate Spanish classifier")
    vec_es, clf_es, nfeat_es = train_classifier(spa_train, model=args.model, analyzer='word', ngram_range=word_range)
    y_es_true = np.array([int(x['label']) for x in spa_test])
    y_es_pred = predict(spa_test, vec_es, clf_es)

    print("Approach: TF-IDF word n-grams; analyzer=word; ngram_range=", word_range, "; model=", args.model)
    print("Total features:", nfeat_es)
    print_first_10_predictions("Spanish test", spa_test, y_es_pred)

    es_metrics = evaluate(y_es_true, y_es_pred)
    print_prf_details("Spanish (classes 0 and 1)", es_metrics)
    print_sample_errors("Spanish", spa_test, y_es_pred, max_samples=10)

    # (e) Cross-language classification (English -> Spanish) via googletrans
    print("\n[TASK (e)] Cross-language classification via MT (ES→EN translate with googletrans, then English model)")
    translator = Translator()
    texts_es = [ex['text'] for ex in spa_test]

    def translate_in_batches(texts: List[str], batch_size: int = 50) -> List[str]:
        outputs: List[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            try:
                result = translator.translate(chunk, src='es', dest='en')
                # Handle async googletrans variants that return a coroutine
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_event_loop()
                        result = loop.run_until_complete(result)
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(result)
                # Normalize to list
                if not isinstance(result, list):
                    result = [result]
                outputs.extend([getattr(r, 'text', str(r)) for r in result])
            except Exception as e:
                # Fallback: translate item by item with small delay
                for t in chunk:
                    try:
                        r = translator.translate(t, src='es', dest='en')
                        if inspect.isawaitable(r):
                            try:
                                loop = asyncio.get_event_loop()
                                r = loop.run_until_complete(r)
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                r = loop.run_until_complete(r)
                        outputs.append(getattr(r, 'text', str(r)))
                        time.sleep(0.05)
                    except Exception:
                        outputs.append("")
                        time.sleep(0.1)
        return outputs

    texts_en = translate_in_batches(texts_es, batch_size=50)

    # Classify translated English texts with the previously trained English vectorizer/model
    X_en = vec_en.transform(texts_en)
    y_x_pred = clf_en.predict(X_en)
    y_x_true = y_es_true

    print("Approach: MT googletrans ES→EN + English TF-IDF word n-grams; analyzer=word; ngram_range=", word_range, "; model=", args.model)
    print_first_10_predictions("Cross-lang MT (Spanish→English test)", spa_test, y_x_pred)

    x_metrics = evaluate(y_x_true, y_x_pred)
    print_prf_details("Cross-language MT on Spanish test (classes 0 and 1)", x_metrics)
    print_sample_errors("Cross-language MT", spa_test, y_x_pred, max_samples=10)


if __name__ == '__main__':
    main()