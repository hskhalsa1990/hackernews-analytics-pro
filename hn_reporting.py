"""Reproducible multi-artifact analytics report export."""
import json
from pathlib import Path
from hn_analytics_core import build_kpis, analyze_stories
from hn_advanced import engagement_analysis, trend_analysis, statistical_summary
from hn_nlp import sentiment_analysis

def export_report(df, output_dir="reports", threshold=200):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    enriched=sentiment_analysis(df); analysis=analyze_stories(enriched); engagement=engagement_analysis(enriched)
    tables={"cleaned_stories":enriched,"daily_trends":trend_analysis(enriched),"domains":analysis["domains"],"authors":engagement["authors"],"sentiment":enriched.groupby("sentiment_label").size().rename("stories").reset_index()}
    for name,table in tables.items(): table.to_csv(out/f"{name}.csv",index=False)
    summary={"kpis":build_kpis(enriched,threshold),"statistics":statistical_summary(enriched),"sentiment_counts":enriched.sentiment_label.value_counts().to_dict(),"artifacts":list(tables)}
    (out/"executive_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    return summary
