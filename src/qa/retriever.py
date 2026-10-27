import argparse
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.qa.data import build_corpus, load_squad
from src.qa.evaluate import save_json
from src.qa.preprocess import tokenize_for_tfidf


class Retriever:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs

    def score(self, questions):
        raise NotImplementedError

    def retrieve(self, question, k=5):
        scores = self.score([question])[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [{**self.paragraphs[i], "score": float(scores[i])} for i in top]


class TfidfRetriever(Retriever):
    def __init__(self, paragraphs, ngram_max=2):
        super().__init__(paragraphs)
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_for_tfidf, lowercase=False, token_pattern=None,
            ngram_range=(1, ngram_max), sublinear_tf=True,
        )
        self.doc_matrix = self.vectorizer.fit_transform(p["text"] for p in paragraphs)

    def score(self, questions):
        return (self.vectorizer.transform(questions) @ self.doc_matrix.T).toarray()


class BM25Retriever(Retriever):
    def __init__(self, paragraphs, k1=1.5, b=0.75):
        super().__init__(paragraphs)
        self.bm25 = BM25Okapi([tokenize_for_tfidf(p["text"]) for p in paragraphs], k1=k1, b=b)

    def score(self, questions):
        return np.vstack([self.bm25.get_scores(tokenize_for_tfidf(q)) for q in questions])


def build_retriever(method, paragraphs, args):
    if method == "tfidf":
        return TfidfRetriever(paragraphs, args.ngram_max)
    if method == "bm25":
        return BM25Retriever(paragraphs, args.k1, args.b)
    raise ValueError(method)


def evaluate_retriever(retriever, questions, gold_ids, ks, batch_size=500):
    titles = np.array([p["title"] for p in retriever.paragraphs])
    ranks, article_hits = [], {k: 0 for k in ks}
    for start in range(0, len(questions), batch_size):
        scores = retriever.score(questions[start : start + batch_size])
        order = np.argsort(-scores, axis=1, kind="stable")
        for row, gold in zip(order, gold_ids[start : start + batch_size]):
            ranks.append(int(np.where(row == gold)[0][0]) + 1)
            for k in ks:
                article_hits[k] += titles[gold] in titles[row[:k]]
    ranks = np.array(ranks)
    return {
        "recall": {f"@{k}": round(100 * float((ranks <= k).mean()), 2) for k in ks},
        "article_recall": {f"@{k}": round(100 * article_hits[k] / len(ranks), 2) for k in ks},
        "mrr": round(float((1 / ranks).mean()), 4),
        "median_rank": float(np.median(ranks)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF and BM25 paragraph retrieval on SQuAD (Recall@k)")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--methods", nargs="+", default=["tfidf", "bm25"], choices=["tfidf", "bm25"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--n-samples", type=int, default=None, help="questions to evaluate; the corpus is always the full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--output", default="results/retriever_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    corpus_dataset = load_squad(args.split, cache_dir=args.cache_dir)
    paragraphs, question_to_paragraph = build_corpus(corpus_dataset)
    eval_dataset = load_squad(args.split, args.n_samples, args.seed, args.cache_dir)
    questions = list(eval_dataset["question"])
    gold_ids = [question_to_paragraph[qid] for qid in eval_dataset["id"]]
    print(f"corpus: {len(paragraphs)} paragraphs, questions: {len(questions)}")

    results = {"corpus_paragraphs": len(paragraphs), "questions": len(questions), "config": vars(args)}
    for method in args.methods:
        t0 = time.time()
        retriever = build_retriever(method, paragraphs, args)
        index_seconds = time.time() - t0
        metrics = evaluate_retriever(retriever, questions, gold_ids, args.ks)
        metrics["index_seconds"] = round(index_seconds, 1)
        metrics["eval_seconds"] = round(time.time() - t0 - index_seconds, 1)
        results[method] = metrics
        recall = "  ".join(f"R{k}={v:.1f}" for k, v in metrics["recall"].items())
        print(f"{method:<6} {recall}  MRR={metrics['mrr']:.3f}  "
              f"(index {metrics['index_seconds']:.0f}s, eval {metrics['eval_seconds']:.0f}s)")

    save_json(results, args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
