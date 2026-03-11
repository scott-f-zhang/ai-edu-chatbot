# EDA Module Data

Place **`clean_df.csv`** here for the EDA preset charts. Same format as the main project processed corpus:

- Required columns: `university`, `text`, `state` (optional for choropleth)
- Optional columns: `year`, `rank`, `token_count`, `policy_typology` (if present, used; otherwise computed from `text`)

Charts mirror the analyses in `notebooks/07_eda.ipynb` (corpus overview, document length, term frequency, policies by state).
