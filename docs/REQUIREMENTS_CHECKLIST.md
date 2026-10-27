# REQUIREMENTS CHECKLIST

Cerintele temei si unde sunt acoperite in proiect. De verificat inainte de predare.

| Cerinta | Unde | Stare |
|---------|------|-------|
| Model extractiv bazat pe BERT | `src/qa/reader.py`, `src/qa/train.py`; DistilBERT 77,6/85,6 si BERT-base 81,2/88,5 pe HF Hub | gata |
| Raspuns pe baza unui pasaj | reader + demo, mod pasaj dat | de facut |
| Data loading | `src/qa/data.py` | gata |
| Preprocessing | `src/qa/preprocess.py` | gata |
| Model | baseline, reader (DistilBERT, BERT), retriever (TF-IDF, BM25), pipeline open-domain (`src/qa/pipeline.py`) | gata |
| Evaluation (EM, F1, Recall@k) | EM/F1 `src/qa/evaluate.py` (validat cu HF), Recall@k + MRR `src/qa/retriever.py` | gata |
| Analiza exploratorie a datelor | `notebooks/eda.ipynb`, figuri in `docs/figures/` | gata |
| NLTK folosit vizibil | Punkt sentence split, `word_tokenize`, PorterStemmer (EDA + baseline) | gata |
| spaCy folosit vizibil | NER + noun chunks (EDA + baseline) | gata |
| scikit-learn folosit vizibil | `TfidfVectorizer` in baseline si retriever, stopwords | gata |
| Hugging Face folosit vizibil | `datasets` (gata), `transformers`, HF Hub | partial |
| Comparatie baseline vs DistilBERT vs BERT | `docs/EXPERIMENTS.md`, `notebooks/results.ipynb`, figura in `docs/figures/` | gata |
| Analiza erorilor | notebook / `evaluate.py` | de facut |
| Demo | `app/app.py` (Gradio) | de facut |
| Rulabil pe laptopul profesorului | `docs/DEPLOYMENT.md` | partial |
| Documentatie | `docs/DOCUMENTATION_NOTES.md` -> document final | in lucru |
| Cod scris de la zero | tot `src/qa/` | in lucru |
