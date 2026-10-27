import argparse
import time
from collections import Counter

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from src.qa.data import build_corpus, load_squad
from src.qa.evaluate import evaluate_predictions, save_json
from src.qa.preprocess import question_type, split_sentences, tokenize_for_tfidf

ENTITY_RULES = {
    "when": {"DATE", "TIME"},
    "how many": {"CARDINAL", "QUANTITY"},
    "how much": {"MONEY", "CARDINAL", "PERCENT", "QUANTITY"},
    "who": {"PERSON", "ORG", "NORP"},
    "whom": {"PERSON", "ORG", "NORP"},
    "whose": {"PERSON", "ORG", "NORP"},
    "where": {"GPE", "LOC", "FAC", "ORG"},
}


def fit_vectorizer(texts):
    sentences = [sentence for text in texts for _, _, sentence in split_sentences(text)]
    vectorizer = TfidfVectorizer(tokenizer=tokenize_for_tfidf, lowercase=False, token_pattern=None)
    vectorizer.fit(sentences)
    return vectorizer, len(sentences)


def in_question(text, question_stems):
    stems = tokenize_for_tfidf(text, remove_stopwords=False)
    return bool(stems) and all(stem in question_stems for stem in stems)


class SentenceBaseline:
    def __init__(self, vectorizer, top_sentences=1):
        self.vectorizer = vectorizer
        self.top_sentences = top_sentences

    def rank_sentences(self, question, sentences):
        question_vector = self.vectorizer.transform([question])
        sentence_vectors = self.vectorizer.transform([s for _, _, s in sentences])
        # tf-idf rows are l2 normalized, so the dot product is the cosine similarity
        scores = (sentence_vectors @ question_vector.T).toarray().ravel()
        return sorted(range(len(sentences)), key=lambda i: -scores[i])

    def answer(self, question, context, doc):
        sentences = split_sentences(context)
        ranking = self.rank_sentences(question, sentences)
        question_stems = set(tokenize_for_tfidf(question, remove_stopwords=False))
        labels = ENTITY_RULES.get(question_type(question))

        if labels:
            for index in ranking[: self.top_sentences]:
                start, end, _ = sentences[index]
                for ent in doc.ents:
                    if start <= ent.start_char < end and ent.label_ in labels and not in_question(ent.text, question_stems):
                        return {"answer": ent.text, "start": ent.start_char, "sentence": index, "method": "entity"}

        start, end, sentence = sentences[ranking[0]]
        for chunk in doc.noun_chunks:
            if start <= chunk.start_char < end and not in_question(chunk.text, question_stems):
                return {"answer": chunk.text, "start": chunk.start_char, "sentence": ranking[0], "method": "noun_chunk"}
        return {"answer": sentence, "start": start, "sentence": ranking[0], "method": "sentence"}


def sentence_contains_answer(context, sentence_index, answer_starts):
    start, end, _ = split_sentences(context)[sentence_index]
    return any(start <= s < end for s in answer_starts)


def parse_args():
    parser = argparse.ArgumentParser(description="Classical QA baseline: TF-IDF sentence selection + spaCy NER rules")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--top-sentences", type=int, default=1)
    parser.add_argument("--output-prefix", default="results/baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    train_paragraphs, _ = build_corpus(load_squad("train", cache_dir=args.cache_dir))
    vectorizer, n_sentences = fit_vectorizer(p["text"] for p in train_paragraphs)
    print(f"tf-idf fit on {n_sentences} train sentences, vocab {len(vectorizer.vocabulary_)} ({time.time() - t0:.0f}s)")

    dataset = load_squad(args.split, args.n_samples, args.seed, args.cache_dir)
    paragraphs, question_to_paragraph = build_corpus(dataset)
    nlp = spacy.load(args.spacy_model, disable=["lemmatizer"])
    docs = dict(zip((p["paragraph_id"] for p in paragraphs), nlp.pipe((p["text"] for p in paragraphs), batch_size=64)))
    print(f"spacy on {len(paragraphs)} paragraphs ({time.time() - t0:.0f}s)")

    baseline = SentenceBaseline(vectorizer, args.top_sentences)
    predictions, methods, sentence_hits = {}, Counter(), 0
    for example in dataset:
        result = baseline.answer(example["question"], example["context"], docs[question_to_paragraph[example["id"]]])
        predictions[example["id"]] = result["answer"]
        methods[result["method"]] += 1
        sentence_hits += sentence_contains_answer(example["context"], result["sentence"], example["answers"]["answer_start"])

    metrics = evaluate_predictions(predictions, dataset)
    metrics["sentence_accuracy"] = round(100 * sentence_hits / len(dataset), 2)
    metrics["answer_method"] = dict(methods)
    metrics["config"] = vars(args)

    print(f"examples: {metrics['count']}  EM: {metrics['exact_match']:.2f}  F1: {metrics['f1']:.2f}  "
          f"sentence acc: {metrics['sentence_accuracy']:.2f}  ({time.time() - t0:.0f}s)")
    for name, group in metrics["by_question_type"].items():
        print(f"  {name:<10} n={group['count']:<5} EM={group['exact_match']:6.2f}  F1={group['f1']:6.2f}")
    print(f"answer method: {dict(methods)}")

    save_json(predictions, f"{args.output_prefix}_predictions.json")
    save_json(metrics, f"{args.output_prefix}_metrics.json")
    print(f"saved: {args.output_prefix}_predictions.json, {args.output_prefix}_metrics.json")


if __name__ == "__main__":
    main()
