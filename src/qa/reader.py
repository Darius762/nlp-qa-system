import argparse
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.qa.data import load_squad
from src.qa.evaluate import evaluate_predictions, save_json


def tokenize_pairs(tokenizer, questions, contexts, max_length, stride, **kwargs):
    return tokenizer(
        [q.lstrip() for q in questions],
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        **kwargs,
    )


def context_bounds(sequence_ids):
    start = sequence_ids.index(1)
    end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
    return start, end


def prepare_train_features(examples, tokenizer, max_length=384, stride=128):
    encoded = tokenize_pairs(tokenizer, examples["question"], examples["context"], max_length, stride)
    sample_map = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        cls_index = encoded["input_ids"][i].index(tokenizer.cls_token_id)
        answers = examples["answers"][sample_map[i]]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        context_start, context_end = context_bounds(encoded.sequence_ids(i))

        # windows that do not fully contain the answer are labeled with the cls token
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        token_start = context_start
        while token_start <= context_end and offsets[token_start][0] <= start_char:
            token_start += 1
        token_end = context_end
        while token_end >= context_start and offsets[token_end][1] >= end_char:
            token_end -= 1
        start_positions.append(token_start - 1)
        end_positions.append(token_end + 1)

    encoded["start_positions"] = start_positions
    encoded["end_positions"] = end_positions
    return encoded


def prepare_eval_features(examples, tokenizer, max_length=384, stride=128):
    encoded = tokenize_pairs(tokenizer, examples["question"], examples["context"], max_length, stride)
    sample_map = encoded.pop("overflow_to_sample_mapping")
    encoded["example_id"] = []
    for i in range(len(encoded["input_ids"])):
        sequence_ids = encoded.sequence_ids(i)
        encoded["example_id"].append(examples["id"][sample_map[i]])
        encoded["offset_mapping"][i] = [
            offset if sequence_ids[k] == 1 else None for k, offset in enumerate(encoded["offset_mapping"][i])
        ]
    return encoded


def best_span(start_logits, end_logits, offsets, n_best=20, max_answer_length=30):
    best = None
    start_indexes = np.argsort(start_logits)[-1 : -n_best - 1 : -1]
    end_indexes = np.argsort(end_logits)[-1 : -n_best - 1 : -1]
    for s in start_indexes:
        for e in end_indexes:
            if offsets[s] is None or offsets[e] is None or e < s or e - s + 1 > max_answer_length:
                continue
            score = float(start_logits[s] + end_logits[e])
            if best is None or score > best[0]:
                best = (score, offsets[s][0], offsets[e][1])
    return best


def predict_logits(model, features, batch_size=32, device=None):
    device = device or next(model.parameters()).device
    model_inputs = [c for c in ("input_ids", "attention_mask", "token_type_ids") if c in features.column_names]
    loader = DataLoader(features.select_columns(model_inputs).with_format("torch"), batch_size=batch_size)
    start_logits, end_logits = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(**{k: v.to(device) for k, v in batch.items()})
            start_logits.append(outputs.start_logits.float().cpu().numpy())
            end_logits.append(outputs.end_logits.float().cpu().numpy())
    return np.concatenate(start_logits), np.concatenate(end_logits)


def predict_dataset(model, tokenizer, dataset, max_length=384, stride=128, n_best=20, max_answer_length=30, batch_size=32):
    features = dataset.map(
        lambda batch: prepare_eval_features(batch, tokenizer, max_length, stride),
        batched=True,
        remove_columns=dataset.column_names,
    )
    start_logits, end_logits = predict_logits(model, features, batch_size)

    features_per_example = defaultdict(list)
    for i, example_id in enumerate(features["example_id"]):
        features_per_example[example_id].append(i)
    offset_mapping = list(features["offset_mapping"])

    predictions = {}
    for example in dataset:
        candidates = [
            best_span(start_logits[i], end_logits[i], offset_mapping[i], n_best, max_answer_length)
            for i in features_per_example[example["id"]]
        ]
        candidates = [c for c in candidates if c is not None]
        if candidates:
            _, start, end = max(candidates)
            predictions[example["id"]] = example["context"][start:end]
        else:
            predictions[example["id"]] = ""
    return predictions, len(features)


class QAReader:
    def __init__(self, model_name_or_path, max_length=384, stride=128, n_best=20, max_answer_length=30, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path).to(self.device).eval()
        self.max_length = max_length
        self.stride = stride
        self.n_best = n_best
        self.max_answer_length = max_answer_length

    def answer(self, question, context):
        encoded = tokenize_pairs(self.tokenizer, [question], [context], self.max_length, self.stride, return_tensors="pt")
        offset_mapping = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping")
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in encoded.items()})
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        best = None
        for i in range(len(offset_mapping)):
            sequence_ids = encoded.sequence_ids(i)
            offsets = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])]
            candidate = best_span(start_logits[i], end_logits[i], offsets, self.n_best, self.max_answer_length)
            if candidate is not None and (best is None or candidate[0] > best[0]):
                best = candidate
        if best is None:
            return {"answer": "", "score": float("-inf"), "start": 0, "end": 0}
        score, start, end = best
        return {"answer": context[start:end], "score": score, "start": start, "end": end}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned extractive QA reader on SQuAD v1.1")
    parser.add_argument("--model", required=True, help="hf hub id or local path")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--max-answer-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    dataset = load_squad(args.split, args.n_samples, args.seed, args.cache_dir)

    predictions, n_features = predict_dataset(
        model, tokenizer, dataset, args.max_length, args.stride, args.n_best, args.max_answer_length, args.batch_size
    )
    metrics = evaluate_predictions(predictions, dataset)
    metrics["n_features"] = n_features
    metrics["config"] = vars(args)

    print(f"examples: {metrics['count']} ({n_features} windows)  EM: {metrics['exact_match']:.2f}  "
          f"F1: {metrics['f1']:.2f}  ({time.time() - t0:.0f}s on {device})")
    for name, group in metrics["by_question_type"].items():
        print(f"  {name:<10} n={group['count']:<5} EM={group['exact_match']:6.2f}  F1={group['f1']:6.2f}")

    prefix = args.output_prefix or f"results/reader_{args.model.rstrip('/').split('/')[-1]}"
    save_json(predictions, f"{prefix}_predictions.json")
    save_json(metrics, f"{prefix}_metrics.json")
    print(f"saved: {prefix}_predictions.json, {prefix}_metrics.json")


if __name__ == "__main__":
    main()
