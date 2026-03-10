# computational_analysis_data — Data Export for AI / LLM Context

This folder contains outputs from the computational analysis of AI-in-higher-education policy documents: **policy typology** (Restrictive, Innovation-Forward, Administrative Risk-Management, Faculty-Autonomy Centered) and related density/count metrics. Use these files as context when answering questions about how policies were classified and how institutions differ by typology.

---

## Governance word lists

- **governance_word_lists.json**  
  The four keyword lists used to score each document: `restrictive_words`, `innovation_words`, `admin_risk_words`, `faculty_words`. Each document is scored by counting how many of these terms appear (case-insensitive, word-boundary); typology is assigned to the dimension with the highest **density** (count / token_count). Use this to understand how typology labels were derived and to replicate or adjust the logic.

---

## Document-level typology and densities

- **df_with_typology_and_densities.csv**  
  One row per document (university × year). Columns include: `university`, `year`, `state`, `rank`, `IPEDS`, `policy_typology`, `token_count`, and for each dimension the raw **count** and **density** (e.g. `restrictive_count`, `restrictive_density`). Also `max_density`, `second_density`, `density_gap`. Use to compare institutions, years, or regions by typology, or to feed document-level typology into downstream models.

---

## Typology distribution

- **typology_distribution.csv**  
  One row per policy typology category; columns: `policy_typology`, `count`. Use to summarize how many documents fall into each type (e.g. “57 Restrictive, 54 Innovation-Forward”).

---

## Density gap (optional)

- **density_gap_summary.json**  
  Summary statistics (min, max, mean, std, etc.) of the `density_gap` (difference between the top and second-highest dimension density per document). Use to understand how “clear” the winning typology is per document.

---

## Summary for LLM

- **computational_analysis_summary_for_llm.txt**  
  Short plain-text summary: output path, number of typology categories, list of categories, and list of exported files. Use as the first piece of context so the model knows what is available in this folder.
