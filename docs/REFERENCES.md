# REFERENCES

Surse de citat in documentatia finala. Se adauga pe masura ce se folosesc.

## Articole
- Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP. arXiv:1606.05250
- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL. arXiv:1810.04805
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108
- Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.
- Chen, D., Fisch, A., Weston, J., Bordes, A. (2017). Reading Wikipedia to Answer Open-Domain Questions (DrQA). ACL. arXiv:1704.00051 — arhitectura retriever-reader folosita in pipeline

## Biblioteci
- Hugging Face `transformers`, `datasets`, `huggingface_hub`
- spaCy (`en_core_web_sm`)
- NLTK
- scikit-learn
- rank-bm25
- Gradio

## Dataset / modele
- SQuAD v1.1: https://huggingface.co/datasets/rajpurkar/squad
- Modele de baza: `distilbert-base-uncased`, `bert-base-uncased`
- Model public folosit doar pentru validarea codului de evaluare: `distilbert/distilbert-base-cased-distilled-squad`
- Modelele noastre: `dariurs/distilbert-base-uncased-squad`, `dariurs/bert-base-uncased-squad`
