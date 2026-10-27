# STATUS

Ultima actualizare: 2026-10-27

## Faza curenta
Faza 3 — reader BERT (gata). Urmeaza Faza 4 — retriever (open-domain).

## Gata
- Setup, docs, `data.py`, `eda.ipynb` (7 figuri), `preprocess.py`, `evaluate.py` (EM/F1 identic cu HF)
- `baseline.py`: validation EM 18.98 / F1 24.68
- `reader.py` + `train.py` + `notebooks/train_colab.ipynb` (repo GitHub e public acum, clone fara token)
- Modele antrenate pe Colab T4, publice pe HF Hub:
  - `dariurs/distilbert-base-uncased-squad`: EM 77.61 / F1 85.55 (29 min)
  - `dariurs/bert-base-uncased-squad`: EM 81.22 / F1 88.54 (63 min)
- `results/`: metrici + predictii pentru baseline si ambele modele (cele de reader copiate de pe Hub)
- `notebooks/results.ipynb`: tabel comparativ global / pe tip de intrebare / pe lungime pasaj + figura `results_f1_by_question_type.png`
- Verificat: modelul de pe Hub se incarca si raspunde local pe CPU (~0,3 s/intrebare)

## In lucru
- nimic

## Urmatorii pasi (de propus planul si de confirmat inainte de implementare)
1. Faza 4 — `src/qa/retriever.py`: TF-IDF (sklearn, tokenizer NLTK din preprocess) si BM25 (`rank-bm25`, deja instalat) peste cele 2067 paragrafe din validation; Recall@1/5/10/20 (paragraful corect in top-k), si la nivel de articol; comparatie TF-IDF vs BM25
2. Faza 5 — `src/qa/pipeline.py`: intrebare -> top-k paragrafe -> reader pe fiecare -> raspunsul cu scorul maxim; EM/F1 open-domain pentru k = 1, 5, 10
3. Faza 6 — demo Gradio `app/app.py`; Faza 7 — analiza erorilor in `notebooks/results.ipynb`

Imbunatatiri propuse (optionale): stergerea folderului `last-checkpoint/` (optimizer ~0,5-1 GB) din repo-urile HF dupa ce nu mai e nevoie de reluare; scos `GH_TOKEN` din `train_colab.ipynb` (repo public); reguli baseline "what year/date" -> DATE.

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale; pentru Colab exista `requirements-colab.txt`.
- Warning HF pe Windows despre symlinks; se poate ignora.
