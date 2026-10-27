import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.qa.data import load_squad
from src.qa.preprocess import normalize_answer, question_type

CONTEXT_LENGTH_BINS = [(0, 100, "<100"), (100, 150, "100-149"), (150, 200, "150-199"), (200, float("inf"), ">=200")]


def exact_match(prediction, truth):
    return float(normalize_answer(prediction) == normalize_answer(truth))


def f1_score(prediction, truth):
    prediction_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    num_same = sum((Counter(prediction_tokens) & Counter(truth_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_truths(metric, prediction, truths):
    return max(metric(prediction, truth) for truth in truths)


def context_length_bin(context):
    n_words = len(context.split())
    return next(label for low, high, label in CONTEXT_LENGTH_BINS if low <= n_words < high)


def score_examples(predictions, dataset):
    rows = []
    for example in dataset:
        if example["id"] not in predictions:
            continue
        prediction = predictions[example["id"]]
        truths = example["answers"]["text"]
        rows.append({
            "id": example["id"],
            "question_type": question_type(example["question"]),
            "context_length": context_length_bin(example["context"]),
            "exact_match": max_over_truths(exact_match, prediction, truths),
            "f1": max_over_truths(f1_score, prediction, truths),
        })
    return rows


def aggregate(rows):
    return {
        "count": len(rows),
        "exact_match": round(100 * sum(r["exact_match"] for r in rows) / len(rows), 2),
        "f1": round(100 * sum(r["f1"] for r in rows) / len(rows), 2),
    }


def aggregate_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: aggregate(group) for name, group in sorted(groups.items(), key=lambda g: -len(g[1]))}


def evaluate_predictions(predictions, dataset):
    rows = score_examples(predictions, dataset)
    metrics = aggregate(rows)
    metrics["missing_predictions"] = len(predictions) - len(rows)
    metrics["by_question_type"] = aggregate_by(rows, "question_type")
    metrics["by_context_length"] = aggregate_by(rows, "context_length")
    return metrics


def check_against_hf(predictions, dataset):
    import evaluate

    squad_metric = evaluate.load("squad")
    examples = [e for e in dataset if e["id"] in predictions]
    return squad_metric.compute(
        predictions=[{"id": e["id"], "prediction_text": predictions[e["id"]]} for e in examples],
        references=[{"id": e["id"], "answers": e["answers"]} for e in examples],
    )


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute EM and F1 for a predictions file {id: answer}")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--output", default=None)
    parser.add_argument("--check-hf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)
    dataset = load_squad(args.split, cache_dir=args.cache_dir)
    metrics = evaluate_predictions(predictions, dataset)

    print(f"examples: {metrics['count']}  EM: {metrics['exact_match']:.2f}  F1: {metrics['f1']:.2f}")
    for name, group in metrics["by_question_type"].items():
        print(f"  {name:<10} n={group['count']:<5} EM={group['exact_match']:6.2f}  F1={group['f1']:6.2f}")

    if args.check_hf:
        hf = check_against_hf(predictions, dataset)
        print(f"hf squad metric: EM: {hf['exact_match']:.2f}  F1: {hf['f1']:.2f}")

    output = args.output or Path(args.predictions).with_name(Path(args.predictions).stem.replace("_predictions", "") + "_metrics.json")
    save_json(metrics, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
