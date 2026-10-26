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
