import argparse
import time

from src.qa.data import build_corpus, load_squad
from src.qa.evaluate import evaluate_predictions, save_json
from src.qa.reader import QAReader
from src.qa.retriever import BM25Retriever, TfidfRetriever


class OpenDomainQA:
    def __init__(self, retriever, reader, top_k=5):
        self.retriever = retriever
        self.reader = reader
        self.top_k = top_k

    def candidates(self, question, top_k=None):
        hits = self.retriever.retrieve(question, top_k or self.top_k)
        answers = self.reader.answer_many(question, [hit["text"] for hit in hits])
        return [
            {
                **answer,
                "rank": rank + 1,
                "paragraph_id": hit["paragraph_id"],
                "title": hit["title"],
                "context": hit["text"],
                "retriever_score": hit["score"],
            }
            for rank, (hit, answer) in enumerate(zip(hits, answers))
        ]

    def answer(self, question, top_k=None):
        candidates = self.candidates(question, top_k)
        best = max(candidates, key=lambda c: c["score"])
        return {**best, "candidates": candidates}


def build_retriever(name, paragraphs):
    return BM25Retriever(paragraphs) if name == "bm25" else TfidfRetriever(paragraphs)


def evaluate_pipeline(pipeline, dataset, question_to_paragraph, ks):
    max_k = max(ks)
    predictions = {k: {} for k in ks}
    gold_predictions = {}
    retrieval_hits = {k: 0 for k in ks}
    picked_gold = {k: 0 for k in ks}
    for i, example in enumerate(dataset):
        gold_id = question_to_paragraph[example["id"]]
        candidates = pipeline.candidates(example["question"], max_k)
        for k in ks:
            best = max(candidates[:k], key=lambda c: c["score"])
            predictions[k][example["id"]] = best["answer"]
            retrieval_hits[k] += any(c["paragraph_id"] == gold_id for c in candidates[:k])
            picked_gold[k] += best["paragraph_id"] == gold_id
        gold_predictions[example["id"]] = pipeline.reader.answer(example["question"], example["context"])["answer"]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dataset)}")

    n = len(dataset)
    results = {}
    for k in ks:
        metrics = evaluate_predictions(predictions[k], dataset)
        results[f"top{k}"] = {
            "exact_match": metrics["exact_match"],
            "f1": metrics["f1"],
            "retrieval_recall": round(100 * retrieval_hits[k] / n, 2),
            "answer_from_gold_paragraph": round(100 * picked_gold[k] / n, 2),
            "by_question_type": metrics["by_question_type"],
        }
    gold = evaluate_predictions(gold_predictions, dataset)
    results["gold_paragraph"] = {"exact_match": gold["exact_match"], "f1": gold["f1"]}
    return results, predictions[max_k]


def parse_args():
    parser = argparse.ArgumentParser(description="Open-domain QA: retriever (top-k paragraphs) + extractive reader")
    parser.add_argument("--reader", default="dariurs/distilbert-base-uncased-squad")
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "tfidf"])
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--question", default=None, help="answer a single question and exit")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    paragraphs, question_to_paragraph = build_corpus(load_squad(args.split, cache_dir=args.cache_dir))
    pipeline = OpenDomainQA(build_retriever(args.retriever, paragraphs), QAReader(args.reader), args.top_k)
    print(f"{args.retriever} over {len(paragraphs)} paragraphs + {args.reader} ready ({time.time() - t0:.0f}s)")

    if args.question:
        result = pipeline.answer(args.question)
        print(f"\nanswer: {result['answer']!r}  (score {result['score']:.2f}, paragraph #{result['rank']} from {result['title']})")
        for c in result["candidates"]:
            print(f"  #{c['rank']} {c['title']:<30} retriever {c['retriever_score']:6.2f}  reader {c['score']:6.2f}  {c['answer']!r}")
        return

    dataset = load_squad(args.split, args.n_samples, args.seed, args.cache_dir)
    results, predictions = evaluate_pipeline(pipeline, dataset, question_to_paragraph, args.ks)
    results["questions"] = len(dataset)
    results["seconds"] = round(time.time() - t0)
    results["config"] = vars(args)

    print(f"questions: {len(dataset)}  ({results['seconds']}s)")
    gold = results["gold_paragraph"]
    print(f"  gold paragraph  EM={gold['exact_match']:6.2f}  F1={gold['f1']:6.2f}")
    for k in args.ks:
        r = results[f"top{k}"]
        print(f"  top-{k:<3} EM={r['exact_match']:6.2f}  F1={r['f1']:6.2f}  "
              f"recall={r['retrieval_recall']:6.2f}  answer from gold paragraph={r['answer_from_gold_paragraph']:6.2f}")

    prefix = args.output_prefix or f"results/pipeline_{args.retriever}_{args.reader.rstrip('/').split('/')[-1]}"
    save_json(predictions, f"{prefix}_top{max(args.ks)}_predictions.json")
    save_json(results, f"{prefix}_metrics.json")
    print(f"saved: {prefix}_metrics.json")


if __name__ == "__main__":
    main()
