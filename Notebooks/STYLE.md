# Notebook Style Guide

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
- No matplotlib plots unless explicitly requested

## Features & Variables
- Define feature lists explicitly — no dynamic column filtering
- Group features clearly: engineered, months, colors, patterns
- When using CV models, define the penalty grid once and pass it to all models symmetrically

## Data Handling
- Filter to a single product type before analysis
- Use `fillna(0)` and `replace([np.inf, -np.inf], 0)` after engineering
- Train/test split by product ID (80/20)
- Standardize features with `StandardScaler`
