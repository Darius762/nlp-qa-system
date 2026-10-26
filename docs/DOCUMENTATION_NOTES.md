# DOCUMENTATION NOTES

Material strans pe parcurs pentru documentatia finala. Se completeaza la fiecare etapa; la final se transforma in documentatie.
Figurile se salveaza in `docs/figures/` (PNG, nume descriptive, ex. `eda_context_length.png`) si se listeaza mai jos.

## 1. Introducere
- Tema 15: Question Answering System, extractiv, bazat pe BERT.
- Problema: dat un pasaj si o intrebare, extrage din pasaj fragmentul care raspunde. Extindere open-domain: gasirea pasajului intr-o colectie, apoi extragerea raspunsului.

## 2. Dataset
- SQuAD v1.1 (Rajpurkar et al., 2016), incarcat din Hugging Face (`rajpurkar/squad`).
- Train: 87.599 intrebari. Validation: 10.570 intrebari, 48 articole, 2.067 paragrafe unice (~43 paragrafe/articol, ~5,1 intrebari/paragraf).
- Validation are pana la 3 raspunsuri de referinta per intrebare; EM/F1 se calculeaza fata de cel mai bun dintre ele.
- Statistici EDA: _de completat dupa `notebooks/eda.ipynb`_

## 3. Pipeline (data loading -> preprocessing -> model -> evaluation)
- Data loading: `src/qa/data.py`, cu subset reproductibil (seed) si corpus de paragrafe unice cu `paragraph_id`, plus maparea intrebare -> paragraf corect.
- Preprocessing: _de completat_
- Modele: _de completat_
- Evaluare: _de completat_

## 4. Metode
### 4.1 Baseline clasic
_de completat_
### 4.2 Reader (DistilBERT, BERT-base)
_de completat_
### 4.3 Retriever (TF-IDF, BM25)
_de completat_
### 4.4 Pipeline open-domain
_de completat_

## 5. Rezultate
Cifrele oficiale sunt in `docs/EXPERIMENTS.md`; aici se pun tabelele finale si interpretarea.

## 6. Analiza erorilor
_de completat: pe tip de intrebare, lungime pasaj, exemple concrete de greseli_

## 7. Demo
_de completat: capturi de ecran din Gradio (ambele moduri) in `docs/figures/`_

## 8. Concluzii si limitari
_de completat_

## Figuri

| Fisier | Ce arata | Sectiune | Generat de |
|--------|----------|----------|------------|
| — | — | — | — |

## Observatii de mentionat in documentatie
- Antrenarea s-a facut pe Colab (T4); local doar CPU, deci local s-a testat pe subseturi mici.
- Corpusul open-domain foloseste paragrafele din validation (vezi `docs/DECISIONS.md`).
