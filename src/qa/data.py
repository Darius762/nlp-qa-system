import argparse
from collections import defaultdict

from datasets import load_dataset

DATASET_NAME = "rajpurkar/squad"


def load_squad(split="validation", n_samples=None, seed=42, cache_dir="data/hf_cache"):
    dataset = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)
    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    return dataset


def build_corpus(*datasets):
    paragraphs = []
    paragraph_ids = {}
    question_to_paragraph = {}
    for dataset in datasets:
        for qid, title, context in zip(dataset["id"], dataset["title"], dataset["context"]):
            key = (title, context)
            if key not in paragraph_ids:
                paragraph_ids[key] = len(paragraphs)
                paragraphs.append({"paragraph_id": len(paragraphs), "title": title, "text": context})
            question_to_paragraph[qid] = paragraph_ids[key]
    return paragraphs, question_to_paragraph


def group_by_article(paragraphs):
    articles = defaultdict(list)
    for paragraph in paragraphs:
        articles[paragraph["title"]].append(paragraph["paragraph_id"])
    return dict(articles)


def parse_args():
    parser = argparse.ArgumentParser(description="Load SQuAD v1.1 and build the paragraph corpus")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_squad(args.split, args.n_samples, args.seed, args.cache_dir)
    paragraphs, question_to_paragraph = build_corpus(dataset)
    articles = group_by_article(paragraphs)

    print(f"split: {args.split}")
    print(f"questions: {len(dataset)}")
    print(f"articles: {len(articles)}")
    print(f"paragraphs: {len(paragraphs)}")
    print(f"paragraphs per article: {len(paragraphs) / len(articles):.2f}")
    print(f"questions per paragraph: {len(question_to_paragraph) / len(paragraphs):.2f}")

    example = dataset[0]
    print("\nexample:")
    print(f"  title: {example['title']}")
    print(f"  question: {example['question']}")
    print(f"  answers: {example['answers']['text']}")
    print(f"  answer_start: {example['answers']['answer_start']}")
    print(f"  paragraph_id: {question_to_paragraph[example['id']]}")
    print(f"  context: {example['context'][:300]}...")


if __name__ == "__main__":
    main()
