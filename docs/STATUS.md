# STATUS

Ultima actualizare: 2026-10-26

## Faza curenta
Faza 1 — data loading + EDA. `data.py` gata, urmeaza EDA.

## Gata
- venv local in `.venv` (Python, CPU only), dependente instalate, `requirements.txt` generat
- Structura proiectului: `src/qa/` (module goale: data, preprocess, baseline, reader, retriever, train, evaluate, pipeline), `app/app.py`, `notebooks/`, `results/`
- `.gitignore` exclude `.venv/`, `data/`, `models/`, `checkpoints/`, `*.safetensors`, `*.bin`
- Git initializat, primul commit (`chore: initial project structure`), remote `origin` -> github.com/Darius762/nlp-qa-system
- `docs/` cu STATUS, DECISIONS, EXPERIMENTS
- `requirements.txt` convertit din UTF-16 in UTF-8
- `src/qa/data.py`: `load_squad` (subset cu seed), `build_corpus` (paragrafe unice + maparea intrebare -> paragraph_id), `group_by_article`; CLI `python -m src.qa.data --split validation --n-samples 200`. Verificat: validation = 10570 intrebari, 48 articole, 2067 paragrafe. Cache HF in `data/hf_cache`.

## In lucru
- nimic

## Urmatorii pasi
1. `notebooks/eda.ipynb`: lungimi pasaje/intrebari/raspunsuri (NLTK), tipuri de intrebari, tipuri de entitati in raspunsuri (spaCy NER), paragrafe per articol
2. `src/qa/preprocess.py` + `src/qa/baseline.py`

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale (ipykernel, debugpy). Pe Colab trebuie instalat fara torch CPU (de rezolvat la `train_colab.ipynb`).
- Warning HF pe Windows despre symlinks (cache-ul merge, doar ocupa mai mult spatiu); se poate ignora.
