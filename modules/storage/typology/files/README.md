# topic_modeling_data — Data Export for AI / LLM Context

This folder contains topic modeling outputs from the AI-in-higher-education policy analysis. Use these files as context when answering questions about topic structure, university-level topic prevalence, and cross-tabulations by rank, school type, and policy typology.

---

## LDA (Latent Dirichlet Allocation)

- **lda_config.json**  
  Chosen number of topics and coherence metric. Keys: `best_num_topics`, `coherence_metric` (e.g. `c_v`). Use to know how many LDA topics exist and how the model was selected.

- **lda_coherence_scores.csv**  
  One row per tested topic count. Columns: `num_topics`, `coherence`. Use to see the coherence curve and why a given K was chosen.

- **lda_topic_terms.json**  
  Top 15 words per LDA topic. Keys: `Topic_0`, `Topic_1`, …; values: list of token strings. Use to interpret what each LDA topic is about (e.g. "Topic_5 = course, instructor, assignment, chatgpt, academic_integrity").

- **df_with_lda_topics.csv**  
  One row per document (university × year). Columns: `university`, `year`, `rank`, `state`, and `Topic_0` … `Topic_K` (probability that the document belongs to each topic). Use to compare topic prevalence across institutions or years, or to feed document-level topic distributions into downstream models.

---

## Topic × Rank (university ranking)

- **topic_by_rank_means.csv**  
  Rows = LDA topics; columns = mean topic probability in "High-rank (better)" vs "Low-rank (worse)" groups, and often a difference column. Use to answer: do higher-ranked universities emphasize certain topics more or less than lower-ranked ones?

---

## Topic × School type (public / private)

- **topic_by_school_type_means.csv**  
  Rows = LDA topics; columns = mean topic probability by school type (e.g. Public, Private nonprofit, Private for-profit), and possibly a public-vs-private difference. Use to compare topic emphasis across institution types.

---

## Topic × Policy typology

- **df_topics_typology.csv**  
  One row per document with policy typology. Columns: `university`, `year`, `policy_typology`, and `Topic_0` … `Topic_K`. Use to link topic distributions to policy categories (e.g. Innovation-Forward, Faculty-Autonomy Centered).

- **topic_by_policy_typology_means.csv**  
  Rows = LDA topics; columns = mean topic probability per policy typology. Use to summarize which topics are more prevalent in which policy type.

---

## BERTopic (neural topic model)

- **bertopic_topic_info.csv**  
  One row per BERTopic topic. Columns typically include topic id, count, name. Topic -1 is usually "outliers/noise". Use to get an overview of how many BERTopic topics there are and how many docs per topic.

- **bertopic_topic_terms.json**  
  Top 15 (word, weight) pairs per BERTopic topic. Keys = integer topic ids. Use to interpret BERTopic topics (semantic labels).

- **df_with_bertopic.csv**  
  One row per document. Columns: `university`, `year`, `bertopic_topic` (assigned topic id), and optionally LDA `Topic_*` columns. Use to analyze or compare BERTopic assignments by institution or year.

---

## Structural topic model / DMR (if present)

- **dmr_topic_labels.json**  
  Human- or model-derived labels per topic (e.g. "Topic_0: researcher, output, human, …"). Same idea as LDA topic terms but for the DMR/STM run. Use to interpret DMR topics.

- **df_stm_with_topics.csv**  
  Document-level DMR/STM topic probabilities. One row per document; columns include `university`, `year`, and `Topic_0` … `Topic_K`. Use like `df_with_lda_topics.csv` but for the structural model.

---

## Summary for LLM

- **topic_analysis_summary_for_llm.txt**  
  Short plain-text summary: output path, number of LDA topics, which cross-tabs were exported, and a list of exported files. Use as the first piece of context so the model knows what is available in this folder.
