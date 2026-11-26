"""Lightweight text analytics: lexicon sentiment and optional TF-IDF topic discovery."""
import re
from collections import Counter
import pandas as pd

_POS = set("amazing awesome beautiful best breakthrough exciting excellent fantastic good great innovative love impressive launch powerful popular success useful win".split())
_NEG = set("bad broken fail failure hate issue problem risk scam slow terrible worst bug".split())

def sentiment_analysis(df, text_col="title"):
    if text_col not in df.columns: raise ValueError(f"Missing text column: {text_col}")
    out=df.copy(); tokens=out[text_col].fillna("").astype(str).str.lower().map(lambda s: re.findall(r"[a-z']+",s))
    out["sentiment_positive"] = tokens.map(lambda t: sum(w in _POS for w in t))
    out["sentiment_negative"] = tokens.map(lambda t: sum(w in _NEG for w in t))
    out["sentiment_score"] = out["sentiment_positive"]-out["sentiment_negative"]
    out["sentiment_label"] = out["sentiment_score"].map(lambda x: "positive" if x>0 else ("negative" if x<0 else "neutral"))
    return out

def discover_topics(df, text_col="title", n_topics=5, max_features=1500):
    """Discover title themes with TF-IDF + NMF; requires scikit-learn."""
    try:
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc: raise ImportError("Install scikit-learn to use topic discovery: pip install scikit-learn") from exc
    texts=df[text_col].fillna("").astype(str)
    if len(texts)<2: raise ValueError("At least two titles are required")
    vectorizer=TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1,2), min_df=1)
    matrix=vectorizer.fit_transform(texts)
    count=max(1,min(int(n_topics),matrix.shape[0],matrix.shape[1]))
    model=NMF(n_components=count, init="nndsvda", random_state=42, max_iter=400)
    weights=model.fit_transform(matrix); terms=vectorizer.get_feature_names_out()
    labels=[" | ".join(terms[i] for i in comp.argsort()[-6:][::-1]) for comp in model.components_]
    result=df.copy(); result["topic_id"]=weights.argmax(axis=1); result["topic_terms"]=result["topic_id"].map(dict(enumerate(labels)))
    return result, pd.DataFrame({"topic_id":range(count),"keywords":labels,"stories":[(result.topic_id==i).sum() for i in range(count)]})
