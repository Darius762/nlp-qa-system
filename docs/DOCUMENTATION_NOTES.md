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
- EDA (`notebooks/eda.ipynb`, validation complet, tokenizare NLTK `word_tokenize`, propozitii NLTK Punkt, NER spaCy `en_core_web_sm`):
  - lungimi (mediana): pasaj 127 tokeni (max 700, ~5 propozitii), intrebare 11, raspuns 2; 73,3% din raspunsuri au <= 3 tokeni
  - 53,2% din intrebari au mai multe raspunsuri de referinta distincte -> EM/F1 = maxim peste referinte
  - intrebare + pasaj in tokeni DistilBERT: mediana 163, p95 299, max 819; doar 1,73% > 384 -> sliding window necesar, dar afecteaza putine exemple
  - tipuri de intrebari: what 57,0%, who 10,5%, when 6,7%, which 6,6%, how many 6,0%, where 4,5%, how 4,1%, why 1,5%, how much 1,5%
  - entitatea raspunsului (NER pe pasaj, entitatea care acopera >= 50% din raspuns): fara entitate 52,8%, DATE 10,0%, ORG 9,3%, PERSON 9,2%, CARDINAL 5,6%, GPE 3,5%
  - tip intrebare -> entitate: when -> DATE 79%, how many -> CARDINAL 68%, who -> PERSON 48% + ORG 16%, how much -> MONEY 34%, where -> GPE/LOC/FAC ~33% + ORG 13%; what -> fara entitate 66%, why -> 97%
  - raspunsul e in prima propozitie in 32,4% din cazuri; trece granita de propozitie doar in 0,19% din cazuri (justifica baseline-ul pe propozitii)
  - corpus retriever: 21-98 paragrafe per articol (medie 43)

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
| `eda_lengths.png` | histograme lungime pasaj / intrebare / raspuns (tokeni NLTK) | 2. Dataset | `notebooks/eda.ipynb` |
| `eda_bert_length.png` | lungime intrebare+pasaj in tokeni DistilBERT, linie la 384 | 2. Dataset / 4.2 Reader | `notebooks/eda.ipynb` |
| `eda_question_types.png` | distributia tipurilor de intrebari | 2. Dataset / 6. Erori | `notebooks/eda.ipynb` |
| `eda_answer_entities.png` | tipul entitatii spaCy din raspuns | 2. Dataset / 4.1 Baseline | `notebooks/eda.ipynb` |
| `eda_qtype_entity_heatmap.png` | % entitate raspuns per tip de intrebare (baza regulilor baseline) | 4.1 Baseline | `notebooks/eda.ipynb` |
| `eda_answer_sentence_position.png` | in a cata propozitie e raspunsul | 4.1 Baseline | `notebooks/eda.ipynb` |
| `eda_paragraphs_per_article.png` | paragrafe per articol in corpusul retriever | 4.3 Retriever | `notebooks/eda.ipynb` |

## Observatii de mentionat in documentatie
- Antrenarea s-a facut pe Colab (T4); local doar CPU, deci local s-a testat pe subseturi mici.
- Corpusul open-domain foloseste paragrafele din validation (vezi `docs/DECISIONS.md`).
