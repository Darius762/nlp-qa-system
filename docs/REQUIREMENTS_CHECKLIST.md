# REQUIREMENTS CHECKLIST

Cerintele temei si unde sunt acoperite in proiect. De verificat inainte de predare.

| Cerinta | Unde | Stare |
|---------|------|-------|
| Model extractiv bazat pe BERT | `src/qa/reader.py`, `src/qa/train.py` | de facut |
| Raspuns pe baza unui pasaj | reader + demo, mod pasaj dat | de facut |
| Data loading | `src/qa/data.py` | gata |
| Preprocessing | `src/qa/preprocess.py` | de facut |
| Model | baseline, reader, retriever | de facut |
| Evaluation (EM, F1, Recall@k) | `src/qa/evaluate.py` | de facut |
| NLTK folosit vizibil | _de completat (tokenizare/sentence split)_ | de facut |
| spaCy folosit vizibil | baseline: sentence split + NER | de facut |
| scikit-learn folosit vizibil | TF-IDF (baseline + retriever) | de facut |
| Hugging Face folosit vizibil | `datasets` (gata), `transformers`, HF Hub | partial |
| Comparatie baseline vs DistilBERT vs BERT | `docs/EXPERIMENTS.md` | de facut |
| Analiza erorilor | notebook / `evaluate.py` | de facut |
| Demo | `app/app.py` (Gradio) | de facut |
| Rulabil pe laptopul profesorului | `docs/DEPLOYMENT.md` | partial |
| Documentatie | `docs/DOCUMENTATION_NOTES.md` -> document final | in lucru |
| Cod scris de la zero | tot `src/qa/` | in lucru |
