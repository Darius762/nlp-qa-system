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

## Urmatorii pasi — PLAN PROPUS, ASTEAPTA CONFIRMARE
Sesiunea din 2026-10-26 s-a oprit aici. Planul de mai jos a fost prezentat utilizatorului, dar nu e confirmat inca.
Cand utilizatorul zice "continua": reaminteste pe scurt planul si intreaba decizia deschisa (sau, daca o confirma direct, implementeaza).

Decizie deschisa: TF-IDF din baseline se antreneaza pe propozitiile din **train** (recomandat, nu atinge datele de evaluare) sau pe paragrafele din validation?

1. `src/qa/preprocess.py`
   - `normalize_answer` (normalizarea oficiala SQuAD: lowercase, fara punctuatie, fara a/an/the, spatii)
   - `question_type` mutata din `notebooks/eda.ipynb` (notebook-ul o importa apoi de aici; la fel `answer_entity` daca e util)
   - `split_sentences` cu NLTK Punkt + offset-uri de caractere (pentru highlight in demo)
   - `tokenize_for_tfidf`: NLTK `word_tokenize` + PorterStemmer
   - CLI care afiseaza cateva exemple procesate
2. `src/qa/evaluate.py`
   - `exact_match`, `f1_score` pe tokeni, maxim peste raspunsurile de referinta
   - defalcare pe tip de intrebare si lungime pasaj
   - verificare o data contra metricii `squad` din HF `evaluate` (deja instalat)
   - CLI `--predictions <fisier>`, salveaza `results/<nume>_metrics.json`
3. `src/qa/baseline.py`
   - sentence split -> TF-IDF (sklearn, tokenizer NLTK + stemming) -> propozitia cu cosine maxim
   - spaCy NER cu reguli din EDA: when -> DATE; how many -> CARDINAL/QUANTITY; who -> PERSON/ORG; how much -> MONEY/CARDINAL; where -> GPE/LOC/FAC/ORG; elimina entitatile care apar in intrebare
   - fallback (what/why/how sau fara entitate): primul noun chunk care nu apare in intrebare
   - metrica extra: sentence accuracy (propozitia aleasa contine raspunsul)
   - argparse: `--split`, `--n-samples`, `--seed`, `--top-sentences`, `--output`
4. Test pe 200 exemple, apoi validation complet (~2-3 min CPU); rezultatele in EXPERIMENTS.md. Asteptare: EM ~10-20%, F1 ~20-30%.

Imbunatatiri EDA propuse (optionale, nefacute): overlap lexical intrebare / propozitia cu raspuns (limita superioara pentru TF-IDF), comparatie distributii train vs validation, scos warning-ul HF_TOKEN din notebook.

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale (ipykernel, debugpy). Pe Colab trebuie instalat fara torch CPU (de rezolvat la `train_colab.ipynb`).
- Warning HF pe Windows despre symlinks; se poate ignora.
