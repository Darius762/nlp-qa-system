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
- Resurse NLTK: _de completat (ex. punkt) cand se folosesc_

## Date
Nu sunt in git. Se descarca automat la prima rulare in `data/hf_cache`:
```
python -m src.qa.data --split validation
```

## Modele
Nu sunt in git. Se descarca automat de pe Hugging Face Hub:
- DistilBERT fine-tuned: _link HF Hub_
- BERT-base fine-tuned: _link HF Hub_

## Rulare pe module

| Pas | Comanda | Durata pe CPU | Output |
|-----|---------|---------------|--------|
| Data loading | `python -m src.qa.data --split validation` | < 1 min | statistici in consola |
| Baseline | _de completat_ | | |
| Reader (evaluare) | _de completat_ | | |
| Retriever | _de completat_ | | |
| Pipeline | _de completat_ | | |
| Demo | _de completat_ | | |

## Probleme cunoscute
- Pe Windows, HF afiseaza un warning despre symlinks; se poate ignora.
- Primul acces la HF Hub poate afisa un warning ca lipseste `HF_TOKEN`; nu e necesar pentru dataset si modele publice.

## Verificat pe
| Data | Sistem | Python | Rezultat |
|------|--------|--------|----------|
| 2026-10-26 | Windows 10, CPU (dezvoltare) | 3.13.1 | `src.qa.data` OK |
