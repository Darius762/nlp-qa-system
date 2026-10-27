# STATUS

Ultima actualizare: 2026-10-27

## Faza curenta
Faza 5 — pipeline open-domain (gata). Urmeaza Faza 6 — demo Gradio.

## Gata
- Setup, docs, `data.py`, `eda.ipynb` (7 figuri), `preprocess.py`, `evaluate.py` (EM/F1 identic cu HF)
- `baseline.py`: validation EM 18.98 / F1 24.68
- `reader.py` (+ `QAReader.answer_many` pentru mai multe paragrafe intr-un batch) + `train.py` + `notebooks/train_colab.ipynb`
- Modele pe HF Hub (publice): `dariurs/distilbert-base-uncased-squad` EM 77.61 / F1 85.55; `dariurs/bert-base-uncased-squad` EM 81.22 / F1 88.54
- `retriever.py`: BM25 R@1 75.05 / R@5 91.37 / R@20 96.52; TF-IDF R@1 72.07 / R@5 89.91
- `pipeline.py`: `OpenDomainQA` (BM25 top-k -> reader -> scor maxim); 500 intrebari, DistilBERT: EM/F1 57.2/65.8 (k=1), 62.8/69.8 (k=10), paragraf corect 74.6/83.6
- `notebooks/results.ipynb`: reader, retriever, pipeline + 3 figuri `results_*.png`

## In lucru
- nimic (stergerea `last-checkpoint/` de pe HF e optionala)

## Urmatorii pasi (de propus planul si de confirmat inainte de implementare)
1. Faza 6 — demo Gradio `app/app.py`: tab "pasaj dat" (pasaj + intrebare -> raspuns evidentiat in pasaj, alegere DistilBERT/BERT) si tab "open-domain" (intrebare -> raspuns + top-k paragrafe cu scoruri); exemple predefinite; capturi de ecran in `docs/figures/`
2. Faza 7 — analiza erorilor in `notebooks/results.ipynb` (exemple concrete: baseline vs BERT, erori de granita why/how, erori pipeline din paragraf gresit)
3. Faza 8 — README final (How to run) din DEPLOYMENT.md + documentatia finala din DOCUMENTATION_NOTES.md

Imbunatatiri propuse (optionale): scor combinat reader + retriever in pipeline (ar trebui sa reduca alegerea paragrafului gresit); pipeline cu BERT-base pe Colab; reguli baseline "what year/date" -> DATE.

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale; pentru Colab exista `requirements-colab.txt`.
- Warning HF pe Windows despre symlinks; se poate ignora.
