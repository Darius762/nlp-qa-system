# STATUS

Ultima actualizare: 2026-10-26

## Faza curenta
Faza 0 — setup (gata). Urmeaza Faza 1 — data loading + EDA.

## Gata
- venv local in `.venv` (Python, CPU only), dependente instalate, `requirements.txt` generat
- Structura proiectului: `src/qa/` (module goale: data, preprocess, baseline, reader, retriever, train, evaluate, pipeline), `app/app.py`, `notebooks/`, `results/`
- `.gitignore` exclude `.venv/`, `data/`, `models/`, `checkpoints/`, `*.safetensors`, `*.bin`
- Git initializat, primul commit (`chore: initial project structure`), remote `origin` -> github.com/Darius762/nlp-qa-system
- `docs/` cu STATUS, DECISIONS, EXPERIMENTS

## In lucru
- nimic

## Urmatorii pasi
1. `src/qa/data.py`: incarcare SQuAD v1.1 prin `datasets`, subset pentru test local, grupare paragrafe pe articole (pentru retriever)
2. `notebooks/eda.ipynb`: statistici dataset (lungimi pasaje/intrebari/raspunsuri, tipuri de intrebari, nr. paragrafe per articol)

## Probleme deschise
- `requirements.txt` e salvat in UTF-16 (generat cu `pip freeze >` din PowerShell) si contine `torch==2.14.1+cpu` + pachete specifice local (ipykernel, debugpy). De verificat inainte de primul run pe Colab: re-salvare in UTF-8 si o varianta fara torch CPU pentru Colab.
