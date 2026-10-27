import argparse
import re
import string

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktTokenizer, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

WH_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
PUNCTUATION = set(string.punctuation)

_stemmer = PorterStemmer()
_sentence_tokenizer = None


def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def get_sentence_tokenizer():
    global _sentence_tokenizer
    if _sentence_tokenizer is None:
        ensure_nltk_resources()
        _sentence_tokenizer = PunktTokenizer("english")
    return _sentence_tokenizer


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def split_sentences(text):
    return [(start, end, text[start:end]) for start, end in get_sentence_tokenizer().span_tokenize(text)]


def tokenize_for_tfidf(text, remove_stopwords=True):
    tokens = word_tokenize(text.lower(), preserve_line=True)
    return [
        _stemmer.stem(token)
        for token in tokens
        if any(ch.isalnum() for ch in token) and not (remove_stopwords and token in ENGLISH_STOP_WORDS)
    ]


def question_type(question):
    tokens = [t.lower() for t in word_tokenize(question, preserve_line=True)]
    for i, token in enumerate(tokens):
        if token == "how" and i + 1 < len(tokens) and tokens[i + 1] in ("many", "much"):
            return f"how {tokens[i + 1]}"
        if token in WH_WORDS:
            return token
    return "other"


def span_entity_label(start, end, entities):
    best_label, best_overlap = "NO_ENTITY", 0
    for ent_start, ent_end, label in entities:
        overlap = min(end, ent_end) - max(start, ent_start)
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return best_label if best_overlap >= 0.5 * (end - start) else "NO_ENTITY"


def parse_args():
    parser = argparse.ArgumentParser(description="Show preprocessing steps on a few SQuAD examples")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    return parser.parse_args()


def main():
    from src.qa.data import load_squad

    args = parse_args()
    dataset = load_squad(args.split, args.n_samples, args.seed, args.cache_dir)
    for example in dataset:
        sentences = split_sentences(example["context"])
        print(f"question:        {example['question']}")
        print(f"question type:   {question_type(example['question'])}")
        print(f"tfidf tokens:    {tokenize_for_tfidf(example['question'])}")
        print(f"answers:         {example['answers']['text']}")
        print(f"normalized:      {[normalize_answer(a) for a in example['answers']['text']]}")
        print(f"sentences:       {len(sentences)}")
        for start, end, sentence in sentences[:3]:
            print(f"  [{start}:{end}] {sentence[:100]}")
        print()


if __name__ == "__main__":
    main()
