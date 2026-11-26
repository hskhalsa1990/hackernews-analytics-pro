"""Leakage-conscious baseline score modelling with chronological holdout."""
import numpy as np
import pandas as pd

def train_score_model(df, min_rows=30):
    """Train a RandomForest on metadata/title features; returns model and metrics.

    Requires scikit-learn. Uses the earliest 80% for training and latest 20% for
    testing when timestamps are available. Score/comments are never predictors.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc: raise ImportError("Install scikit-learn: pip install scikit-learn") from exc
    data=df.copy(); data["score"]=pd.to_numeric(data.score,errors="coerce"); data=data.dropna(subset=["score"])
    if len(data)<min_rows: raise ValueError(f"Need at least {min_rows} scored stories; found {len(data)}")
    features=pd.DataFrame({"title":data.title.fillna(""),"domain":data.domain.fillna("unknown"),"type":data.type.fillna("unknown"),"hour":data.hour_utc.fillna(-1).astype(int)})
    order=data.created_at.argsort(kind="stable") if "created_at" in data and data.created_at.notna().any() else pd.Series(np.arange(len(data)),index=data.index).sort_values().index
    features=features.loc[order]; target=np.log1p(data.loc[order,"score"].clip(lower=0))
    split=max(1,int(len(features)*.8)); split=min(split,len(features)-1)
    prep=ColumnTransformer([("title",TfidfVectorizer(max_features=1200,ngram_range=(1,2),stop_words="english"),"title"),("cats",OneHotEncoder(handle_unknown="ignore"),["domain","type"]),("hour","passthrough",["hour"])])
    model=Pipeline([("features",prep),("forest",RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1))])
    model.fit(features.iloc[:split],target.iloc[:split]); pred=model.predict(features.iloc[split:]); actual=target.iloc[split:]
    metrics={"train_rows":split,"test_rows":len(actual),"mae_log_score":float(mean_absolute_error(actual,pred)),"rmse_log_score":float(mean_squared_error(actual,pred)**.5),"r2_log_score":float(r2_score(actual,pred)) if len(actual)>1 else None,"target":"log1p(score)","split":"chronological 80/20"}
    return model,metrics
