# Domain Docs

This project uses a single-context layout: one root `GLOSSARY.md` and `docs/adr/`.

## Before exploring

Read `GLOSSARY.md` for project terminology and ADRs in `docs/adr/` that apply to the area being changed. If a file or directory is absent, proceed silently. The domain-modeling skill creates documentation when terms or decisions are resolved.

## File structure

- `GLOSSARY.md`: project terms and definitions.
- `docs/adr/NNNN-short-title.md`: architecture decision records.

## Use the glossary's vocabulary

Use glossary terms when naming domain concepts in issues, code, plans, and explanations. If a concept is missing, reconsider the wording or note the gap for domain-modeling.

## Surface decision conflicts

When a proposal contradicts an existing ADR, identify the ADR and explain why the decision should be revisited.
