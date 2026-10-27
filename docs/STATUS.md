# STATUS

Ultima actualizare: 2026-10-27

## Faza curenta
Faza 3 — reader BERT. Codul e gata si validat local; urmeaza antrenarea pe Colab (o face utilizatorul).

## Gata
- Setup, docs, `data.py`, `eda.ipynb` (7 figuri), `preprocess.py`, `evaluate.py` (EM/F1 identic cu HF)
- `baseline.py`: validation EM 18.98 / F1 24.68 (top-1), EM 19.76 / F1 25.56 (top-2)
- `src/qa/reader.py`: features train/eval cu sliding window + offset mapping, postprocesare n-best, `predict_dataset`, clasa `QAReader` (pentru pipeline/demo), CLI de evaluare. Validat: aliniere etichete 99,4%, model public -> EM 78,6 / F1 86,3 pe 500 ex (publicat 79,1 / 86,9)
- `src/qa/train.py`: `Trainer` HF, push pe Hub cu checkpoint-uri (`--resume-from-hub`), evaluare finala, metrici + predictii urcate in `results/` pe Hub. Smoke test local OK
- `notebooks/train_colab.ipynb` + `requirements-colab.txt`

## In lucru
- Antrenarea pe Colab (utilizatorul): DistilBERT apoi BERT-base -> `dariurs/distilbert-base-uncased-squad`, `dariurs/bert-base-uncased-squad`
  - inainte: commit + push pe GitHub (notebook-ul cloneaza repo-ul)
  - in Colab: runtime T4, secret `HF_TOKEN` (iconita cheie din stanga), Run All

## Urmatorii pasi
1. Dupa Colab: descarca metricile de pe Hub (`results/` din repo-urile modelelor) in `results/`, trece EM/F1/durata in EXPERIMENTS.md, comparatie cu baseline-ul
2. Optional: rulare locala `python -m src.qa.reader --model dariurs/distilbert-base-uncased-squad --n-samples 200` ca verificare ca modelul de pe Hub se incarca
3. Faza 4 — retriever: `src/qa/retriever.py` (TF-IDF sklearn + BM25 rank-bm25 peste cele 2067 paragrafe validation, Recall@k) — de propus planul

Imbunatatiri baseline propuse (optionale): "what year/date" -> DATE, "what percentage" -> PERCENT, entitatea cea mai apropiata de cuvintele din intrebare.

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale; pentru Colab exista `requirements-colab.txt`.
- Warning HF pe Windows despre symlinks; se poate ignora.
