# NLP QA System — Retriever + Reader

Proiect universitar NLP, tema 15: extractive Question Answering.
Pipeline: data loading -> preprocessing -> model -> evaluation.
Biblioteci obligatorii folosite: NLTK, spaCy, scikit-learn, Hugging Face.

## Arhitectura
- baseline: sentence split (spaCy/NLTK) + TF-IDF cosine (sklearn) + reguli pe tipul intrebarii (spaCy NER)
- reader: DistilBERT / BERT-base fine-tuned pe SQuAD v1.1 (start/end span)
- retriever: TF-IDF si BM25 peste paragrafele SQuAD grupate pe articole
- evaluare: Exact Match + F1 pentru reader, Recall@k pentru retriever, analiza erorilor pe tip de intrebare
- demo: Gradio in app/app.py

## Mediu
- local: Windows, CPU only (i5, 16GB RAM), fara GPU. Nu rula antrenari complete local.
- antrenarea se face pe Colab (T4) prin notebooks/train_colab.ipynb, care cloneaza repo-ul si ruleaza src/qa/train.py
- modelele antrenate se urca pe Hugging Face Hub, nu in git
- local se testeaza pe subseturi mici (ex. 200 exemple) pentru debug

## Reguli de cod
- Fara comentarii in cod. Daca un comentariu e absolut necesar: litere mici, fara diacritice.
- Cod modular in src/qa/, fiecare script rulabil cu python -m src.qa.<modul>
- Hiperparametrii si caile se dau ca argumente CLI (argparse), nu hardcodate
- Seed fix pentru reproductibilitate
- Nu adauga dependente noi fara sa intrebi

## Git
- Commit-uri mici, mesaje cu prefix: feat, fix, data, exp, docs, chore
- Nu face commit la data/, models/, checkpoints sau fisiere .env
- Nu face push fara confirmare

## Comunicare
- Raspunde in romana, codul si numele de variabile in engleza