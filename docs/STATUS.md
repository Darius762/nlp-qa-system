# STATUS

Ultima actualizare: 2026-10-26

## Faza curenta
Faza 1 — data loading + EDA (gata). Urmeaza Faza 2 — preprocessing + baseline clasic.

## Gata
- Setup: venv `.venv` (Python 3.13.1, CPU), structura `src/qa/`, git + remote github.com/Darius762/nlp-qa-system, `requirements.txt` in UTF-8
- `CLAUDE.md` scos din git (in `.gitignore`), ramane doar local
- `docs/`: STATUS, DECISIONS, EXPERIMENTS, DOCUMENTATION_NOTES, DEPLOYMENT, REQUIREMENTS_CHECKLIST, REFERENCES, `figures/`
- `src/qa/data.py`: `load_squad`, `build_corpus` (paragrafe unice + intrebare -> paragraph_id), `group_by_article`; validation = 10570 intrebari, 48 articole, 2067 paragrafe
- `notebooks/eda.ipynb`: rulat, cu output salvat; 7 figuri in `docs/figures/`, concluziile sunt in notebook si in DOCUMENTATION_NOTES

## In lucru
- nimic

## Urmatorii pasi
1. `src/qa/preprocess.py`: normalizare text si raspuns (pentru EM/F1), tipul intrebarii (functia din EDA mutata in modul), sentence split
2. `src/qa/evaluate.py`: EM si F1 (maxim peste referinte), ca sa putem evalua baseline-ul
3. `src/qa/baseline.py`: TF-IDF pe propozitii + reguli NER (when -> DATE, how many -> CARDINAL, who -> PERSON/ORG, how much -> MONEY, where -> GPE/LOC/FAC) + fallback pentru what/why/how

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale (ipykernel, debugpy). Pe Colab trebuie instalat fara torch CPU (de rezolvat la `train_colab.ipynb`).
- Warning HF pe Windows despre symlinks; se poate ignora.
