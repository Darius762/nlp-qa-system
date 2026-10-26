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
| Analiza exploratorie a datelor | `notebooks/eda.ipynb`, figuri in `docs/figures/` | gata |
| NLTK folosit vizibil | EDA: `word_tokenize`, Punkt sentence split (gata); baseline: de facut | partial |
| spaCy folosit vizibil | EDA: NER pe pasaje (gata); baseline: NER + reguli (de facut) | partial |
| scikit-learn folosit vizibil | TF-IDF (baseline + retriever) | de facut |
| Hugging Face folosit vizibil | `datasets` (gata), `transformers`, HF Hub | partial |
| Comparatie baseline vs DistilBERT vs BERT | `docs/EXPERIMENTS.md` | de facut |
| Analiza erorilor | notebook / `evaluate.py` | de facut |
| Demo | `app/app.py` (Gradio) | de facut |
| Rulabil pe laptopul profesorului | `docs/DEPLOYMENT.md` | partial |
| Documentatie | `docs/DOCUMENTATION_NOTES.md` -> document final | in lucru |
| Cod scris de la zero | tot `src/qa/` | in lucru |
