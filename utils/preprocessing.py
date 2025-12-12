import re
from utils.config import nlp, stopwords_set
from tqdm import tqdm

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_set]
    return " ".join(tokens)

def lemmatize_list(texts):
    print("Lemmatizando textos...")
    docs = nlp.pipe(texts, batch_size=200, n_process=1)
    out = []
    for doc in tqdm(docs, total=len(texts)):
        lemmas = [t.lemma_ for t in doc if not t.is_stop]
        out.append(" ".join(lemmas))
    return out
