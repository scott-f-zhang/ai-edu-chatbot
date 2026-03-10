# AI in Higher Education Chatbot

A Chainlit-based chatbot for analyzing higher education AI policy materials with module-specific RAG workflows, notebook-backed preset charts, and a default general-purpose policy corpus.

## Features

- Default chat experience on app start without forcing module selection
- Retrieval-augmented responses over local module files
- Specialized modules for:
  - Explanatory Computational Analysis
  - Typology Analysis
  - Thematic Analysis
- Notebook-backed preset Plotly charts that bypass LLM chart generation
- LLM-generated charts from module CSV or Excel files
- Separate RAG service with streamed responses
- Slash commands for module switching, rebuilds, config changes, and history clearing

## Default Module

The app starts in a `default` module so users can begin chatting immediately.

The default module corpus is built from:

`/Users/scott/repos/ai-in-higher-education/data/processed/clean_df.csv`

It groups rows by `university`, concatenates each university's `text`, and stores the combined output in:

`modules/storage/default/files/default_module_corpus.txt`

## Modules

- `default`: General cross-university AI policy questions
- `explanatory_computational`: Computational and typology-oriented analysis
- `typology`: Topic-modeling and typology comparisons
- `thematic`: Sentiment and thematic analysis assets

## Running Locally

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and fill in any required secrets:

```bash
cp .env.example .env
```

4. Start the app:

```bash
chainlit run app.py -w
```

## Commands

- `/switch`: Switch to another module
- `/images`: Show images in the current module
- `/rebuild`: Rebuild the current module index
- `/config`: View or update LLM configuration
- `/clear`: Clear conversation history
- `/help`: Show available commands

## Deployment Notes

For short-lived demos, Railway is a good fit for this project. If you deploy it, remember to set the required environment variables and rebuild the default module index after the first launch if needed.
