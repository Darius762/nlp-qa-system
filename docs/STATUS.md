# STATUS

Ultima actualizare: 2026-10-27

## Faza curenta
Faza 2 — preprocessing + evaluare + baseline clasic (gata). Urmeaza Faza 3 — reader BERT.

## Gata
- Setup: venv `.venv` (Python 3.13.1, CPU), structura `src/qa/`, git + remote github.com/Darius762/nlp-qa-system, `requirements.txt` in UTF-8; `CLAUDE.md` doar local (in `.gitignore`)
- `docs/`: STATUS, DECISIONS, EXPERIMENTS, DOCUMENTATION_NOTES, DEPLOYMENT, REQUIREMENTS_CHECKLIST, REFERENCES, `figures/` (7 figuri EDA)
- `src/qa/data.py`: incarcare SQuAD v1.1, corpus de paragrafe, grupare pe articole
- `notebooks/eda.ipynb`: rulat, importa `question_type` / `span_entity_label` din `preprocess.py`
- `src/qa/preprocess.py`: normalizare SQuAD, sentence split NLTK cu offset-uri, tokenizare NLTK + stemming, tipul intrebarii, eticheta NER a unui span
- `src/qa/evaluate.py`: EM/F1 (maxim peste referinte), defalcare pe tip intrebare si lungime pasaj, `--check-hf` (identic cu HF `squad`)
- `src/qa/baseline.py`: TF-IDF (fit pe train) + reguli NER + fallback noun chunk. Validation: EM 18.98 / F1 24.68 (top-1), EM 19.76 / F1 25.56 (top-2); rezultate in `results/`

## In lucru
- nimic

## Urmatorii pasi (de propus planul si de confirmat inainte de implementare)
1. `src/qa/reader.py` + `src/qa/train.py`: preprocesare pentru BERT (tokenizare question+context, max_length 384, stride 128, offset mapping, pozitii start/end), fine-tuning cu `Trainer`, postprocesare (n-best start/end, max answer length), evaluare cu `evaluate.py`
2. Test local pe ~200 exemple (cativa pasi de antrenare, doar ca sa verificam ca merge), apoi `notebooks/train_colab.ipynb` pentru antrenarea completa pe T4 si upload pe HF Hub
3. De rezolvat odata cu Colab: instalare fara `torch==...+cpu` din `requirements.txt`

Imbunatatiri baseline propuse (optionale, nefacute): "what year/what date" -> regula DATE, "what percentage" -> PERCENT; alegerea entitatii celei mai apropiate de cuvintele din intrebare in loc de prima; analiza erorilor pe 20-30 exemple.

## Probleme deschise
- `requirements.txt` contine `torch==2.14.1+cpu` si pachete doar locale (ipykernel, debugpy).
- Warning HF pe Windows despre symlinks; se poate ignora.
