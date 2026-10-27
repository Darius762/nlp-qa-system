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
- Preprocessing: `src/qa/preprocess.py` — normalizarea oficiala SQuAD a raspunsurilor, sentence split NLTK Punkt cu offset-uri de caractere, tokenizare NLTK + PorterStemmer pentru TF-IDF, tipul intrebarii (primul cuvant interogativ, cu "how many"/"how much" separat), eticheta NER a unui span.
- Modele: _de completat_
- Evaluare: `src/qa/evaluate.py` — EM si F1 pe tokeni dupa normalizare, maxim peste raspunsurile de referinta; defalcare pe tip de intrebare si lungime pasaj (<100, 100-149, 150-199, >=200 cuvinte). Validat: rezultate identice cu metrica `squad` din HF `evaluate`.

## 4. Metode
### 4.1 Baseline clasic
- Pasul 1, selectia propozitiei: TF-IDF (sklearn) pe tokeni NLTK cu stemming, antrenat pe 93.596 propozitii din train; propozitia cu cosine maxim fata de intrebare.
- Pasul 2, extragerea raspunsului: tipul intrebarii -> etichete NER spaCy acceptate (when -> DATE/TIME, how many -> CARDINAL/QUANTITY, how much -> MONEY/CARDINAL/PERCENT/QUANTITY, who/whom/whose -> PERSON/ORG/NORP, where -> GPE/LOC/FAC/ORG); se ia prima entitate potrivita care nu apare deja in intrebare.
- Fallback (what/which/how/why sau fara entitate): primul noun chunk spaCy din propozitie care nu apare in intrebare.
- Varianta top-k: entitatea se cauta in primele k propozitii (k=1 referinta, k=2 testat).
- Rezultat (validation): EM 18,98, F1 24,68; propozitia corecta aleasa in 76,1% din cazuri.
- Interpretare: selectia propozitiei merge relativ bine, pierderea e la extragerea span-ului; regulile NER functioneaza pe tipurile cu entitate clara (when EM 56, how many 43, who 35), iar what/how/why (63% din date) raman slabe (EM 4-11). Asta motiveaza reader-ul BERT, care invata direct span-ul.
### 4.2 Reader (DistilBERT, BERT-base)
- Formulare: QA extractiv = prezicerea pozitiei de start si de end a raspunsului in pasaj; peste encoder se pune un strat liniar (`qa_outputs`) care da doua scoruri per token.
- Intrare: `[CLS] intrebare [SEP] pasaj [SEP]`, max_length 384. Pasajele mai lungi se impart in ferestre care se suprapun cu 128 tokeni (stride), ca raspunsul sa nu fie taiat la granita.
- Etichete la antrenare: offset mapping-ul tokenizer-ului transforma pozitia raspunsului din caractere in tokeni; ferestrele care nu contin raspunsul complet sunt etichetate pe [CLS]. Verificat: 99,4% din ferestre decodeaza exact raspunsul, restul sunt adnotari in mijlocul unui cuvant.
- Inferenta: pentru fiecare fereastra, top-20 start x top-20 end, doar combinatii valide (in pasaj, end >= start, <= 30 tokeni), scor = logit start + logit end; se alege cel mai bun span din toate ferestrele si se mapeaza inapoi in text cu offset-urile.
- Validarea codului: pe un model public antrenat (distilbert-base-cased-distilled-squad) obtinem EM 78,6 / F1 86,3 pe 500 exemple, practic scorurile publicate (79,1 / 86,9).
- Antrenare: `Trainer` HF, 2 epoci, lr 3e-5, warmup 10%, weight decay 0,01, fp16, Colab T4. Modele: `dariurs/distilbert-base-uncased-squad`, `dariurs/bert-base-uncased-squad`.
- Rezultate (validation): DistilBERT EM 77,61 / F1 85,55 (29 min pe T4); BERT-base EM 81,22 / F1 88,54 (63 min). Ambele egale cu valorile din articole (77,7 / 85,8 si 80,8 / 88,5).
### 4.3 Retriever (TF-IDF, BM25)
- Scop: pentru open-domain, gasirea paragrafului relevant dintr-o colectie (2.067 paragrafe din 48 de articole Wikipedia, validation SQuAD).
- Tokenizare comuna: NLTK `word_tokenize` + PorterStemmer, fara stopwords (aceeasi ca la baseline).
- TF-IDF (sklearn): unigrame + bigrame, tf sublinear (1 + log tf), similaritate cosine intre intrebare si paragraf.
- BM25 (rank-bm25, Okapi): saturatia frecventei termenului (k1 = 1,5) si normalizare dupa lungimea documentului (b = 0,75).
- Evaluare: Recall@k (paragraful sursa al intrebarii in top-k), MRR, si Recall@k la nivel de articol.
- Rezultate: BM25 R@1 75,1 / R@5 91,4 / R@20 96,5; TF-IDF 1-2 grame 72,1 / 89,9 / 96,1; TF-IDF unigrame 69,7 / 89,4 / 96,2.
- Interpretare: BM25 e mai bun mai ales la k mic; bigramele ajuta TF-IDF (+2,4 R@1). Articolul corect e gasit in top-1 in ~93% din cazuri, deci greselile sunt intre paragrafe din acelasi articol (lexical foarte asemanatoare). Limitare: intrebarile SQuAD au fost scrise privind paragraful, deci suprapunerea lexicala e mai mare decat in open-domain real.
### 4.4 Pipeline open-domain
- Arhitectura retriever-reader (ca in DrQA, Chen et al., 2017): intrebare -> BM25 top-k paragrafe -> reader-ul citeste toate paragrafele intr-un batch -> raspunsul cu scorul maxim al reader-ului.
- `src/qa/pipeline.py`: clasa `OpenDomainQA` (folosita si in demo), CLI pentru o intrebare (`--question`) sau evaluare.
- Rezultate (500 intrebari, DistilBERT): EM/F1 57,2/65,8 (k=1), 61,2/69,0 (k=3), 62,8/69,8 (k=10); cu paragraful corect dat: 74,6/83,6.
- Interpretare: recall-ul BM25 creste de la 76% la 94% (k 1 -> 10), dar raspunsul vine din paragraful corect doar in ~75% din cazuri la orice k. Scorurile reader-ului nu sunt calibrate intre paragrafe, iar paragrafele din acelasi articol contin span-uri plauzibile (exemplu: "Which NFL team won Super Bowl 50?" -> paragraful clasat primul raspunde "Carolina Panthers", al doilea "Denver"). Pierderea totala fata de limita superioara: ~14 F1, jumatate din retrieval (k=1) si jumatate din alegerea gresita intre paragrafe.

## 5. Rezultate
Cifrele oficiale sunt in `docs/EXPERIMENTS.md`; aici se pun tabelele finale si interpretarea. Tabelele se regenereaza cu `notebooks/results.ipynb`.

| Model | EM | F1 |
|---|---|---|
| Baseline (TF-IDF + NER) | 18,98 | 24,68 |
| DistilBERT | 77,61 | 85,55 |
| BERT-base | 81,22 | 88,54 |

- Salt enorm de la baseline la reader (+59 EM): baseline-ul alege bine propozitia (76%), dar nu stie sa extraga span-ul; BERT invata direct granitele raspunsului.
- BERT-base vs DistilBERT: +3,6 EM / +3,0 F1, la cost ~2x (parametri, timp de antrenare 63 vs 29 min).
- Pe tipuri: baseline-ul e competitiv relativ doar unde regulile NER se aplica (when F1 60, how many 49); pe what/which/how/why e sub 20 F1, iar reader-ul e peste 72 peste tot.
- why/how sunt cele mai grele si pentru BERT (EM 50-68, F1 73-83): raspunsuri lungi, explicative, granite ambigue.
- Lungimea pasajului nu conteaza (variatie < 1,5 F1) -> sliding window-ul nu pierde informatie.

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
| `results_f1_by_question_type.png` | F1 baseline vs DistilBERT vs BERT pe tip de intrebare | 5. Rezultate | `notebooks/results.ipynb` |
| `results_retriever_recall.png` | Recall@k TF-IDF (1-2 grame, unigrame) vs BM25 | 4.3 Retriever / 5. Rezultate | `notebooks/results.ipynb` |
| `results_pipeline_topk.png` | pipeline BM25 + DistilBERT: Recall@k, EM, F1 vs k + limita superioara | 4.4 Pipeline / 5. Rezultate | `notebooks/results.ipynb` |

## Observatii de mentionat in documentatie
- Antrenarea s-a facut pe Colab (T4); local doar CPU, deci local s-a testat pe subseturi mici.
- Corpusul open-domain foloseste paragrafele din validation (vezi `docs/DECISIONS.md`).
