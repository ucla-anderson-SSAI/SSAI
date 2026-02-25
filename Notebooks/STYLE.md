# Notebook Style Guide

## Markdown & Section Structure
- Each notebook starts with the course name as plain bold text, then `#` heading for the week title
- Use a numbered hierarchy for sections: `#` for major sections (1, 2, 3…), `#` for subsections (1.1, 1.2…), no numbering for sub-subsections
- Separate major sections with `---` horizontal rules
- Before each major code block, include a markdown cell with a `####` heading that summarizes what the code below does in 1–3 plain sentences
- Keep summaries concise and conversational — no bullet points, no bold syntax callouts, no listing individual function names or method calls
- Focus on what is happening in the pipeline and why it matters, not on Python syntax
- Mention well-known packages by name where relevant (`pandas`, `numpy`, `sklearn`, `matplotlib`) but weave them into the sentence naturally
- Sub-subsection markdown cells (e.g., Lasso, Ridge descriptions) also use `####` with 1–2 sentences

## Scope
- This style guide applies to all notebooks in the course, not just Week 1
- Other weeks cover topics like bandits, Keras-style neural networks, XGBoost, and more — the same markdown structure, summary style, and code philosophy apply regardless of the ML technique being taught
- When introducing new packages or frameworks (e.g., `tensorflow`/`keras`, `xgboost`), mention them naturally in the `####` summary cells the same way `sklearn` is referenced in Week 1

## Code Philosophy
- Keep code compact and minimal — avoid unnecessary abstractions
- Use default parameters whenever possible; only specify non-defaults
- Prefer explicit, inline code over functions unless reuse is clear (e.g., `split_and_scale`)
- Don't wrap simple sequential logic in functions (e.g., feature engineering)
- No for-loops for small repetitive assignments — write them out explicitly

## Naming & Terminology
- Use **weights** (not coefficients)
- Use **λ (lambda)** for Lasso/Ridge penalty strength
- Use **α (alpha)** for Elastic Net mixing parameter
- Don't reference L1 or L2 — just say Lasso, Ridge, Elastic Net

## Comments & Documentation
- Add short inline comments at key locations (model fitting, feature definitions, etc.)
- Keep comments brief — describe *what* the line does, not how
- Don't over-comment obvious code
- No references to external apps or tools

## Output & Display
- Use `pd.set_option('display.max_columns', None)` so all columns are visible
- Use simple `print()` statements for model results — one line per model
- Avoid DataFrames just for display purposes
- Include minimal matplotlib plots where they help visualize results (e.g., hyperparameter sweeps, training curves, model comparisons) — keep plotting code to 2–4 lines

## Features & Variables
- Define feature lists explicitly — no dynamic column filtering
- Group features clearly: engineered, months, colors, patterns
- When using CV models, define the penalty grid once and pass it to all models symmetrically

## Data Handling
- Filter to a single product type before analysis
- Use `fillna(0)` and `replace([np.inf, -np.inf], 0)` after engineering
- Train/test split by product ID (80/20)
- Standardize features with `StandardScaler`
