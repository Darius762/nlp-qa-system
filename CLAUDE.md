# NLP QA System — Retriever + Reader

## Despre proiect
Proiect final la cursul de NLP (master, Facultatea ACE Craiova), tema 15: Question Answering System.
Cerinta: model extractiv (bazat pe BERT) care raspunde la intrebari pe baza unui pasaj de text.
Pipeline obligatoriu: data loading -> preprocessing -> model -> evaluation.
Biblioteci care trebuie folosite vizibil: NLTK, spaCy, scikit-learn, Hugging Face.
Profesorul vrea un proiect construit de la zero, nu cod reciclat din alte proiecte.

## Arhitectura (varianta aleasa: B + D)
1. Baseline clasic, fara deep learning: sentence split (spaCy/NLTK) + TF-IDF cosine (sklearn) pentru alegerea propozitiei + reguli pe tipul intrebarii cu spaCy NER (who -> PERSON, when -> DATE etc.)
2. Reader: DistilBERT si BERT-base fine-tuned pe SQuAD v1.1, predictie start/end span, sliding window cu stride si offset mapping
3. Retriever (open-domain): TF-IDF si BM25 peste paragrafele SQuAD grupate pe articole Wikipedia; evaluat cu Recall@k
4. Pipeline complet: intrebare -> retriever -> top-k paragrafe -> reader -> raspuns
5. Evaluare: Exact Match + F1 pentru reader, Recall@k pentru retriever, comparatie baseline vs DistilBERT vs BERT, analiza erorilor pe tip de intrebare si lungime pasaj
6. Demo: Gradio in app/app.py (mod pasaj dat + mod open-domain)

## Mediu
- Local: Windows, PowerShell, venv in .venv, CPU only (i5-1135G7, 16GB RAM), fara GPU
- NU rula antrenari complete local. Local se testeaza doar pe subseturi mici (ex. 200 exemple)
- Antrenarea se face pe Google Colab (T4), prin extensia Colab din VS Code sau direct pe Colab
- notebooks/train_colab.ipynb cloneaza repo-ul de pe GitHub si ruleaza python -m src.qa.train
- Modelele antrenate se urca pe Hugging Face Hub, niciodata in git
- Repo: github.com/Darius762/nlp-qa-system

## Retinerea contextului (OBLIGATORIU)
Fisierele din docs/ sunt memoria proiectului intre sesiuni.
- docs/STATUS.md: faza curenta, ce e gata, ce e in lucru, urmatorii pasi, probleme deschise
- docs/DECISIONS.md: jurnal de decizii (data, decizie, motiv, alternative respinse)
- docs/EXPERIMENTS.md: fiecare antrenare/evaluare (data, model, hiperparametri, EM, F1, Recall@k, link HF Hub, observatii)

La inceputul fiecarei sesiuni: citeste STATUS.md si DECISIONS.md inainte de orice altceva.
La finalul fiecarui task: actualizeaza STATUS.md; daca s-a luat o decizie, adaug-o in DECISIONS.md; daca s-au obtinut rezultate, adauga-le in EXPERIMENTS.md.
Pastreaza STATUS.md scurt si la zi, rescrie-l, nu il transforma in jurnal.

## Git (REGULI STRICTE)
- Nu rula niciodata git add, git commit, git push, git reset --hard sau git rebase. Commit-urile si push-ul le face utilizatorul din interfata VS Code.
- Ai voie doar cu comenzi read-only: git status, git diff, git log.
- La finalul fiecarui task, propune un singur mesaj de commit scurt, pe un rand, cu prefix: feat, fix, data, exp, docs, chore, refactor. Exemplu: feat: tf-idf sentence baseline
- Nu crea fisiere in data/, models/ sau checkpoints/ care ar trebui urcate in git.

## Reguli de cod
- Fara comentarii in cod. Daca un comentariu e absolut necesar: litere mici, fara diacritice.
- Codul si numele de variabile in engleza.
- Cod modular in src/qa/, fiecare modul rulabil cu python -m src.qa.<modul>
- Hiperparametrii si caile se dau prin argparse, nu hardcodate.
- Seed fix pentru reproductibilitate.
- Nu adauga dependente noi fara sa intrebi; daca se adauga, actualizeaza requirements.txt.
- Inainte de taskuri mari, prezinta un plan scurt si asteapta confirmarea.

## Comunicare
- Raspunde in romana.
- Explica pe scurt ce ai facut si de ce, fara rezumate lungi.