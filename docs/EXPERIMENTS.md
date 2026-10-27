# EXPERIMENTS

Fiecare antrenare/evaluare: data, model, hiperparametri, EM, F1, Recall@k, link HF Hub, observatii.

## Sumar

### Reader (SQuAD v1.1 validation)

| Data | Model | EM | F1 | HF Hub |
|------|-------|----|----|--------|
| 2026-10-27 | Baseline clasic (TF-IDF + NER, top-1 propozitie) | 18.98 | 24.68 | — |
| 2026-10-27 | Baseline clasic (TF-IDF + NER, top-2 propozitii) | 19.76 | 25.56 | — |
| 2026-10-27 | DistilBERT (distilbert-base-uncased, 2 ep) | 77.61 | 85.55 | [dariurs/distilbert-base-uncased-squad](https://huggingface.co/dariurs/distilbert-base-uncased-squad) |
| 2026-10-27 | BERT-base (bert-base-uncased, 2 ep) | 81.22 | 88.54 | [dariurs/bert-base-uncased-squad](https://huggingface.co/dariurs/bert-base-uncased-squad) |

### Retriever

| Data | Metoda | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|------|--------|----------|----------|-----------|-----------|
| — | — | — | — | — | — |

### Pipeline open-domain

| Data | Retriever | Reader | top-k | EM | F1 |
|------|-----------|--------|-------|----|----|
| — | — | — | — | — | — |

## Detalii

### 2026-10-27 — Reader DistilBERT si BERT-base (Colab T4)
- Comanda: `notebooks/train_colab.ipynb` -> `python -m src.qa.train --model-name <model> --epochs 2 --lr 3e-5 --batch-size 16|12 --max-length 384 --stride 128 --seed 42 --fp16 --push-to-hub`
- Date: train complet (87.599 intrebari -> 88.524 ferestre), evaluare validation complet (10.570 intrebari -> 10.784 ferestre)
- Hiperparametri comuni: lr 3e-5, warmup 10%, weight decay 0,01, fp16, n-best 20, max answer 30 tokeni

| | DistilBERT | BERT-base |
|---|---|---|
| batch / pasi | 16 / 11.066 | 12 / 14.754 |
| timp antrenare | 1.764 s (~29 min) | 3.776 s (~63 min) |
| timp total (cu evaluare + upload) | 1.901 s | 4.000 s |
| train loss mediu | 1,317 | 1,111 |
| EM / F1 | **77,61 / 85,55** | **81,22 / 88,54** |
| literatura (referinta) | 77,7 / 85,8 (Sanh et al., 2019) | 80,8 / 88,5 (Devlin et al., 2019) |

- F1 pe tip de intrebare (Baseline / DistilBERT / BERT): what 16,1 / 85,1 / 88,1; who 41,6 / 89,2 / 91,5; when 60,1 / 91,0 / 94,2; which 18,6 / 86,9 / 87,7; how many 49,2 / 89,4 / 92,5; where 29,4 / 83,5 / 86,1; how 8,8 / 75,2 / 82,7; why 10,1 / 72,8 / 76,0; how much 41,4 / 87,5 / 90,4
- F1 pe lungime pasaj (<100 / 100-149 / 150-199 / >=200 cuvinte): DistilBERT 85,3 / 85,8 / 85,1 / 86,1; BERT 88,0 / 89,2 / 88,6 / 87,7
- Fisiere: `results/reader_*_metrics.json`, `results/reader_*_predictions.json` (copiate de pe Hub); figura `docs/figures/results_f1_by_question_type.png` (`notebooks/results.ipynb`)
- Verificare locala: modelul DistilBERT de pe Hub se incarca in ~16 s si raspunde in ~0,3 s/intrebare pe CPU
- Observatii: ambele modele reproduc scorurile din literatura. BERT-base castiga +3,6 EM / +3,0 F1 fata de DistilBERT, cel mai mult pe "how" (+7,5 F1). why/how raman cele mai grele (raspunsuri lungi, granite ambigue: EM mult sub F1). Lungimea pasajului nu influenteaza scorul -> sliding window-ul functioneaza.

### 2026-10-27 — Verificari reader (fara antrenare reala)
- Aliniere etichete (`prepare_train_features`, distilbert-base-uncased, 3000 exemple train -> 3030 ferestre): 2991/3008 ferestre cu raspuns decodeaza exact raspunsul; cele 17 diferente sunt raspunsuri adnotate in mijlocul unui cuvant (ex. "x. Men" din "sex. Men") sau artefacte de decodare ("qur ' an"); 22 ferestre fara raspuns -> [CLS].
- Model public `distilbert/distilbert-base-cased-distilled-squad`, `python -m src.qa.reader --n-samples 500`: EM 78,60 / F1 86,34 (raportat pe tot setul: 79,1 / 86,9) -> postprocesarea e corecta. 141 s pe CPU.
- Smoke test `src.qa.train` local: 200 exemple train, 10 pasi, CPU, ~2 min, loss 5.9 -> pipeline-ul train + eval + salvare ruleaza cap-coada.

### 2026-10-27 — Baseline clasic (TF-IDF sentence selection + spaCy NER rules)
- Model: `src/qa/baseline.py`, fara deep learning
- Date: TF-IDF fit pe 93.596 propozitii din paragrafele train (vocab 74.093 stem-uri); evaluare pe validation complet (10.570 intrebari)
- Configuratie: tokenizer NLTK `word_tokenize` + PorterStemmer, stopwords sklearn; spaCy `en_core_web_sm`; reguli NER: when -> DATE/TIME, how many -> CARDINAL/QUANTITY, how much -> MONEY/CARDINAL/PERCENT/QUANTITY, who/whom/whose -> PERSON/ORG/NORP, where -> GPE/LOC/FAC/ORG; fallback primul noun chunk care nu apare in intrebare
- Comanda: `python -m src.qa.baseline` (top-1), `python -m src.qa.baseline --top-sentences 2 --output-prefix results/baseline_top2`
- Rezultate top-1: EM 18.98, F1 24.68, sentence accuracy 76.13% (`results/baseline_metrics.json`)
- Rezultate top-2: EM 19.76, F1 25.56, sentence accuracy 76.93% (`results/baseline_top2_metrics.json`)
- Pe tip de intrebare (top-1, EM / F1): when 56.2 / 60.1, how many 42.6 / 49.2, who 35.4 / 41.6, how much 30.5 / 41.4, where 22.9 / 29.4, which 12.0 / 18.6, what 10.6 / 16.1, why 5.2 / 10.1, how 3.7 / 8.8
- Metoda de raspuns (top-1): noun chunk 8.120, entitate 2.399, propozitie intreaga 51
- Verificare: EM/F1 identice cu metrica `squad` din HF `evaluate` (18.98 / 24.68)
- Mediu: local CPU, ~90 s per rulare (30 s fit TF-IDF, 25 s spaCy, restul inferenta)
- Observatii: propozitia corecta e gasita in 76% din cazuri, dar EM e doar 19% -> pierderea mare e la extragerea span-ului, nu la selectia propozitiei. Regulile NER merg bine (when/how many/who); what/how/why (63% din date) cad pe fallback-ul noun chunk si sunt slabe. Top-2 ajuta putin (+0.8 EM) fiindca permite cautarea entitatii si in a doua propozitie. Raman ca referinta top-1 (top-2 a fost ales privind validation).

<!--
### YYYY-MM-DD — <model / experiment>
- Model: 
- Date: train / eval (subset?)
- Hiperparametri: lr, epochs, batch size, max_length, doc_stride, seed
- Rezultate: EM, F1 / Recall@k
- HF Hub: 
- Mediu: Colab T4 / local CPU, durata
- Observatii: 
-->
