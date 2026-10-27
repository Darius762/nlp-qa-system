# REQUIREMENTS CHECKLIST

Cerintele temei si unde sunt acoperite in proiect. De verificat inainte de predare.

| Cerinta | Unde | Stare |
|---------|------|-------|
| Model extractiv bazat pe BERT | `src/qa/reader.py`, `src/qa/train.py` | de facut |
| Raspuns pe baza unui pasaj | reader + demo, mod pasaj dat | de facut |
| Data loading | `src/qa/data.py` | gata |
| Preprocessing | `src/qa/preprocess.py` | gata |
| Model | baseline (gata), reader, retriever | partial |
| Evaluation (EM, F1, Recall@k) | `src/qa/evaluate.py`: EM/F1 gata (validat cu HF), Recall@k de facut | partial |
| Analiza exploratorie a datelor | `notebooks/eda.ipynb`, figuri in `docs/figures/` | gata |
| NLTK folosit vizibil | Punkt sentence split, `word_tokenize`, PorterStemmer (EDA + baseline) | gata |
| spaCy folosit vizibil | NER + noun chunks (EDA + baseline) | gata |
| scikit-learn folosit vizibil | TF-IDF baseline (gata), retriever (de facut) | partial |
| Hugging Face folosit vizibil | `datasets` (gata), `transformers`, HF Hub | partial |
| Comparatie baseline vs DistilBERT vs BERT | `docs/EXPERIMENTS.md` (baseline gata) | partial |
| Analiza erorilor | notebook / `evaluate.py` | de facut |
| Demo | `app/app.py` (Gradio) | de facut |
| Rulabil pe laptopul profesorului | `docs/DEPLOYMENT.md` | partial |
| Documentatie | `docs/DOCUMENTATION_NOTES.md` -> document final | in lucru |
| Cod scris de la zero | tot `src/qa/` | in lucru |
