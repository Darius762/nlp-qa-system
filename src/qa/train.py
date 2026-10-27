import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from src.qa.data import load_squad
from src.qa.evaluate import evaluate_predictions, save_json
from src.qa.reader import predict_dataset, prepare_train_features


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an extractive QA reader on SQuAD v1.1")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1, help="ratio of total steps")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--max-answer-length", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--resume-from-hub", action="store_true", help="resume from last-checkpoint pushed to the hub")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def resolve_checkpoint(args):
    if not args.resume_from_hub:
        return None
    local = snapshot_download(args.hub_model_id, allow_patterns=["last-checkpoint/*"], local_dir=Path(args.output_dir) / "hub")
    checkpoint = Path(local) / "last-checkpoint"
    return str(checkpoint) if checkpoint.exists() else None


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = f"{args.model_name.split('/')[-1]}-squad"
    args.output_dir = args.output_dir or f"checkpoints/{run_name}"
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--push-to-hub needs --hub-model-id")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    train_raw = load_squad("train", args.train_samples, args.seed, args.cache_dir)
    eval_raw = load_squad("validation", args.eval_samples, args.seed, args.cache_dir)
    train_features = train_raw.map(
        lambda batch: prepare_train_features(batch, tokenizer, args.max_length, args.stride),
        batched=True,
        remove_columns=train_raw.column_names,
    )
    print(f"train: {len(train_raw)} examples -> {len(train_features)} windows")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        seed=args.seed,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    train_result = trainer.train(resume_from_checkpoint=resolve_checkpoint(args))
    trainer.save_model(args.output_dir)
    print(f"training done ({time.time() - t0:.0f}s)")

    predictions, n_features = predict_dataset(
        trainer.model, tokenizer, eval_raw, args.max_length, args.stride, args.n_best, args.max_answer_length,
        args.eval_batch_size,
    )
    metrics = evaluate_predictions(predictions, eval_raw)
    metrics["n_features"] = n_features
    metrics["train"] = train_result.metrics
    metrics["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    metrics["total_seconds"] = round(time.time() - t0)
    metrics["config"] = vars(args)
    print(f"eval: {metrics['count']} examples  EM: {metrics['exact_match']:.2f}  F1: {metrics['f1']:.2f}")

    predictions_path = Path(args.results_dir) / f"reader_{run_name}_predictions.json"
    metrics_path = Path(args.results_dir) / f"reader_{run_name}_metrics.json"
    save_json(predictions, predictions_path)
    save_json(metrics, metrics_path)
    print(f"saved: {predictions_path}, {metrics_path}")

    if args.push_to_hub:
        trainer.push_to_hub(commit_message=f"final model, EM {metrics['exact_match']:.2f} F1 {metrics['f1']:.2f}")
        api = HfApi()
        for path in (predictions_path, metrics_path):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"results/{path.name}", repo_id=args.hub_model_id)
        print(f"pushed: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
