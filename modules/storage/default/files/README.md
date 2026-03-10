# Default Module

This is the fallback module used when the app starts.

Users can begin chatting immediately in this module without selecting a specialized analysis area first.

Add general reference files here if you want the default conversation path to use local documents.

`default_module_corpus.txt` is generated from `/Users/scott/repos/ai-in-higher-education/data/processed/clean_df.csv` by grouping rows on `university` and concatenating each group's `text` values into one section per university.
