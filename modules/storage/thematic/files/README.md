# Sentiment analysis exports

This folder contains machine-readable exports produced by `notebooks/04_sentiment_analysis.ipynb`.

## Files

- **`sentiment_records.csv`**
  - Main table (one row per policy/document after filtering).
  - Includes the original text plus metadata (university, year, IPEDS, etc.), grouping variables (e.g., `school_type`, `policy_typology`), and sentiment outputs from both approaches:
    - VADER: `vader_compound`, `vader_label`
    - RoBERTa: `roberta_score`, `roberta_label`

- **`sentiment_records.parquet`** (optional)
  - Same content as the CSV, stored in Parquet for faster loading in Python/R/AI pipelines.
  - May be missing if the runtime does not have a Parquet engine installed (e.g., `pyarrow`).

- **`data_dictionary.csv`**
  - Lightweight schema table listing each column name, dtype, null counts, and a sample value.

## Notes on key columns

- **Identifiers / metadata**
  - `IPEDS`: Institution ID used for merging with IPEDS characteristics data.
  - `university`, `state`, `year`, `rank`, `link`: Document metadata.

- **Grouping variables**
  - `school_type`: Derived from IPEDS `CONTROL` (Public / Private nonprofit / Private for-profit). If IPEDS merge is missing, a simple name heuristic is used as a fallback.
  - `policy_typology`: Imported from `data/processed/clean_df_with_typology.csv` (e.g., Restrictive / Innovation-Forward / Administrative).

- **Sentiment outputs**
  - `vader_compound`: Continuous score in [-1, 1] from NLTK VADER.
  - `vader_label`: Discrete label derived from `vader_compound` using thresholds (>= 0.05 Positive; <= -0.05 Negative; else Neutral).
  - `roberta_score`: Continuous score in approximately [-1, 1], computed by chunking long documents and averaging chunk-level predictions from `cardiffnlp/twitter-roberta-base-sentiment-latest`.
  - `roberta_label`: Discrete label derived from `roberta_score` using the same thresholds as above.

## Generated timestamp

- Generated on: 2026-03-09 07:05:20
