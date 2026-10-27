# DECISIONS

Jurnal de decizii. Format: data — decizie — motiv — alternative respinse.

## 2026-10-26 — Arhitectura: varianta B + D (baseline clasic + retriever/reader)
- **Decizie:** baseline clasic (spaCy/NLTK sentence split + TF-IDF cosine + reguli NER pe tipul intrebarii), reader extractiv DistilBERT si BERT-base fine-tuned, retriever TF-IDF/BM25 pentru open-domain, pipeline complet retriever -> reader, demo Gradio.
- **Motiv:** acopera cerinta (model extractiv bazat pe BERT) si foloseste vizibil NLTK, spaCy, scikit-learn si Hugging Face; baseline-ul clasic da un punct de comparatie, iar retriever-ul extinde proiectul dincolo de "pasaj dat".
- **Alternative respinse:** doar reader fine-tuned (prea putin, fara baseline si fara folosire reala a sklearn/NLTK); retriever dens (DPR) — cost de antrenare prea mare pentru resursele disponibile.

## 2026-10-26 — Dataset: SQuAD v1.1
- **Decizie:** SQuAD v1.1 (train pentru fine-tuning, validation pentru evaluare), incarcat prin Hugging Face `datasets`.
- **Motiv:** standard pentru QA extractiv, fiecare intrebare are raspuns in pasaj, metrici EM/F1 consacrate; paragrafele grupate pe articole Wikipedia permit evaluarea retriever-ului cu Recall@k.
- **Alternative respinse:** SQuAD v2.0 (intrebari fara raspuns complica reader-ul si evaluarea, nu sunt cerute de tema); Natural Questions (prea mare, preprocesare grea).

## 2026-10-26 — Antrenare pe Colab, modele pe Hugging Face Hub
- **Decizie:** fine-tuning pe Google Colab (GPU T4) prin `notebooks/train_colab.ipynb`, care cloneaza repo-ul si ruleaza `python -m src.qa.train`. Modelele antrenate se urca pe HF Hub, niciodata in git.
- **Motiv:** fine-tuning BERT pe SQuAD nu e fezabil pe CPU; HF Hub tine modelele versionate si usor de incarcat in evaluare/demo; repo-ul ramane mic.
- **Alternative respinse:** antrenare locala (prea lenta); modele in git / Git LFS (fisiere mari, limite de stocare); Google Drive (mai greu de incarcat reproductibil din cod).

## 2026-10-26 — Mediu local: CPU only
- **Decizie:** local (Windows, i5-1135G7, 16GB RAM, fara GPU) se dezvolta codul si se testeaza doar pe subseturi mici (~200 exemple); torch instalat in varianta CPU.
- **Motiv:** fara GPU local; subseturile mici permit verificarea rapida a codului inainte de rularea pe Colab.
- **Alternative respinse:** rulari complete locale (ore/zile pe CPU).

## 2026-10-26 — Corpus retriever: paragrafele din SQuAD validation
- **Decizie:** corpusul pentru open-domain contine paragrafele unice din split-ul validation (48 articole, 2067 paragrafe). `build_corpus` accepta mai multe split-uri, deci se poate extinde ulterior cu train.
- **Motiv:** rapid pe CPU, intrebarile de evaluare au paragraful corect in corpus, setup folosit frecvent in proiecte similare.
- **Alternative respinse (deocamdata):** train + validation (~20k paragrafe), mai realist, dar mai lent; ramane optiune de extindere.

## 2026-10-27 — Baseline: TF-IDF antrenat pe propozitiile din train
- **Decizie:** vectorizer-ul TF-IDF al baseline-ului se antreneaza pe propozitiile paragrafelor din train (93.596 propozitii), tokenizare NLTK + PorterStemmer, stopwords sklearn.
- **Motiv:** nimic din datele de evaluare nu influenteaza modelul; IDF calculat pe propozitii, aceeasi unitate pe care o clasificam.
- **Alternative respinse:** fit pe paragrafele din validation (vocabular/IDF din datele de evaluare).

## 2026-10-27 — Baseline: reguli NER din EDA + fallback noun chunk
- **Decizie:** tipul intrebarii (primul cuvant interogativ) decide etichetele NER acceptate; entitatile care apar deja in intrebare sunt eliminate; daca nu exista entitate potrivita, raspunsul e primul noun chunk din propozitia aleasa care nu apare in intrebare. Referinta oficiala ramane varianta cu top-1 propozitie.
- **Motiv:** regulile vin direct din heatmap-ul din EDA (when -> DATE 79%, how many -> CARDINAL 68% etc.); top-1 ramane referinta ca sa nu alegem configuratia privind setul de evaluare.
- **Alternative respinse:** clasificator de tip de raspuns antrenat (iese din ideea de baseline cu reguli); intoarcerea propozitiei intregi (F1 foarte mic, raspunsurile au mediana 2 tokeni).

## 2026-10-27 — Reader: configuratie de antrenare
- **Decizie:** `distilbert-base-uncased` si `bert-base-uncased`; 2 epoci, lr 3e-5, batch 16 (DistilBERT) / 12 (BERT), weight decay 0.01, warmup 10%, max_length 384, stride 128, fp16, seed 42; postprocesare n-best 20, lungime maxima raspuns 30 tokeni. Modele pe Hub: `dariurs/distilbert-base-uncased-squad`, `dariurs/bert-base-uncased-squad`.
- **Motiv:** hiperparametrii standard pentru SQuAD (articolul BERT si exemplele HF), comparabili cu literatura; uncased e varianta clasica din articole; 384/128 acopera 98,3% din perechi intr-o singura fereastra (EDA).
- **Alternative respinse:** cautare de hiperparametri (cost GPU, nu e scopul temei); modele cased (rezultate apropiate, comparatie mai putin directa cu literatura).

## 2026-10-27 — Antrenare: checkpoint-uri pe Hub si requirements separat pentru Colab
- **Decizie:** `hub_strategy="checkpoint"` (ultimul checkpoint urcat pe Hub in timpul antrenarii, reluare cu `--resume-from-hub`); predictiile si metricile se urca in `results/` din repo-ul modelului; Colab instaleaza `requirements-colab.txt` (doar bibliotecile HF fixate + nltk/sklearn), nu `requirements.txt`.
- **Motiv:** sesiunile Colab se pot deconecta; `requirements.txt` local are `torch+cpu` care ar strica GPU-ul pe Colab.
- **Alternative respinse:** checkpoint-uri pe Google Drive (mai greu de reluat din cod); un singur requirements cu marker-e de platforma (fragil).

## 2026-10-27 — Validarea reader-ului cu un model public
- **Decizie:** inainte de antrenare, pipeline-ul de evaluare (`reader.py`: ferestre, offset mapping, n-best) se verifica pe `distilbert/distilbert-base-cased-distilled-squad`, deja antrenat.
- **Motiv:** un bug in postprocesare ar strica toate rezultatele fara sa fie evident; daca obtinem scorurile publicate ale modelului, codul e corect.
- **Rezultat:** EM 78,60 / F1 86,34 pe 500 exemple, fata de 79,1 / 86,9 raportat pe tot setul -> corect.

## 2026-10-27 — Metrici: implementare proprie EM/F1, validata cu HF
- **Decizie:** `src/qa/evaluate.py` implementeaza EM/F1 dupa scriptul oficial SQuAD (normalizare + maxim peste referinte), cu defalcare pe tip de intrebare si lungime pasaj; `--check-hf` compara cu metrica `squad` din HF `evaluate`.
- **Motiv:** avem nevoie de scoruri per exemplu pentru analiza erorilor; validarea arata ca cifrele sunt identice cu cele oficiale.
- **Alternative respinse:** doar HF `evaluate` (nu da scoruri per exemplu).
