# DEPLOYMENT

Cum se ruleaza proiectul pe alt calculator (ex. laptopul profesorului). Se actualizeaza la fiecare modul nou; la final devine sectiunea "How to run" din README.

## Cerinte
- Python 3.13 (dezvoltat pe 3.13.1, Windows 10)
- Git
- Conexiune la internet la prima rulare (descarca SQuAD si modelele de pe HF Hub)
- Nu e nevoie de GPU pentru inferenta, evaluare si demo
- Spatiu pe disc: _de completat (dataset + modele)_

## Instalare

Windows (PowerShell):
```powershell
git clone https://github.com/Darius762/nlp-qa-system.git
cd nlp-qa-system
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:
```bash
git clone https://github.com/Darius762/nlp-qa-system.git
cd nlp-qa-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note:
- `requirements.txt` fixeaza `torch==2.14.1+cpu`; daca pip nu il gaseste, se instaleaza separat: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, apoi restul.
- Modelul spaCy `en_core_web_sm` e inclus in `requirements.txt` ca link direct.
- Resurse NLTK: `punkt_tab` (descarcat automat in cod cu `nltk.download("punkt_tab", quiet=True)`, ajunge in `%APPDATA%\nltk_data` pe Windows). Manual: `python -c "import nltk; nltk.download('punkt_tab')"`
- Tokenizer-ul `distilbert-base-uncased` se descarca automat in `data/hf_cache` la prima rulare a EDA.

## Date
Nu sunt in git. Se descarca automat la prima rulare in `data/hf_cache`:
```
python -m src.qa.data --split validation
```

## Modele
Nu sunt in git. Se descarca automat de pe Hugging Face Hub:
- DistilBERT fine-tuned: https://huggingface.co/dariurs/distilbert-base-uncased-squad (_dupa antrenarea pe Colab_)
- BERT-base fine-tuned: https://huggingface.co/dariurs/bert-base-uncased-squad (_dupa antrenarea pe Colab_)
- Modelele se descarca in cache-ul HF implicit (`~/.cache/huggingface`, pe Windows `C:\Users\<user>\.cache\huggingface`), ~260 MB DistilBERT, ~420 MB BERT-base.

## Antrenare (doar pe Colab, nu e necesara pentru rulare)
`notebooks/train_colab.ipynb`: cloneaza repo-ul de pe GitHub, instaleaza `requirements-colab.txt`, login HF (secret `HF_TOKEN`), ruleaza `python -m src.qa.train` pentru ambele modele si le urca pe Hub. Codul trebuie sa fie push-uit pe GitHub inainte, notebook-ul ia ultima versiune de acolo.

## Rulare pe module

| Pas | Comanda | Durata pe CPU | Output |
|-----|---------|---------------|--------|
| Data loading | `python -m src.qa.data --split validation` | < 1 min | statistici in consola |
| EDA | deschide `notebooks/eda.ipynb` in VS Code/Jupyter, kernel din `.venv`, Run All | ~1-2 min | grafice in `docs/figures/` |
| Preprocessing (exemple) | `python -m src.qa.preprocess --n-samples 3` | < 30 s | exemple procesate in consola |
| Baseline | `python -m src.qa.baseline` (test rapid: `--n-samples 200 --output-prefix results/baseline_test`) | ~1,5 min (200 ex: ~35 s) | `results/baseline_predictions.json`, `results/baseline_metrics.json` |
| Evaluare predictii | `python -m src.qa.evaluate --predictions results/baseline_predictions.json [--check-hf]` | < 30 s | `results/baseline_metrics.json` |
| Reader (evaluare) | `python -m src.qa.reader --model dariurs/distilbert-base-uncased-squad` (test rapid: `--n-samples 200`) | DistilBERT: ~0,3 s/intrebare (200 ex: ~1 min; tot setul ~45-60 min) | `results/reader_<model>_predictions.json`, `_metrics.json` |
| Train (smoke test) | `python -m src.qa.train --train-samples 200 --eval-samples 100 --max-steps 10 --batch-size 8` | ~2-3 min | `checkpoints/` (ignorat de git) |
| Retriever | _de completat_ | | |
| Pipeline | _de completat_ | | |
| Demo | _de completat_ | | |

## Probleme cunoscute
- Pe Windows, HF afiseaza un warning despre symlinks; se poate ignora.
- Primul acces la HF Hub poate afisa un warning ca lipseste `HF_TOKEN`; nu e necesar pentru dataset si modele publice.
- Prima rulare a baseline-ului descarca si split-ul train (~30 MB) pentru fit-ul TF-IDF.
- `--check-hf` din `src.qa.evaluate` descarca o data metrica `squad` din HF `evaluate`.

## Verificat pe
| Data | Sistem | Python | Rezultat |
|------|--------|--------|----------|
| 2026-10-26 | Windows 10, CPU (dezvoltare) | 3.13.1 | `src.qa.data` OK, `eda.ipynb` OK |
| 2026-10-27 | Windows 10, CPU (dezvoltare) | 3.13.1 | `src.qa.preprocess`, `src.qa.baseline`, `src.qa.evaluate --check-hf` OK |
