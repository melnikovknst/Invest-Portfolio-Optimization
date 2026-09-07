# Article source and reproduction

The complete English article is `../Portfolio optimization.docx`; `manuscript.md` is its readable Markdown counterpart. The current edition contains an abstract, related literature, data and model definitions, evaluation protocol, results, discussion, conclusion, references and implementation appendices.

The 7 September repository review preserves the original experiment results, marks the disputed equal-weight quality exposure as N/V in Table 6, and clarifies the historical/current code distinction in Appendix B. See `../REVIEW.md`. Original article QA is archived in `../artifacts/review/original_article_quality_checks.json`; `quality_checks.json` describes the revised 16-page document.

## Files

- `manuscript.template.md`: editable prose, TeX equation counterparts, figure references and table markers.
- `manuscript.md`: generated prose with the saved numerical tables inserted; edit the template instead.
- `references.json`: numbered bibliographic records and primary-source links.
- `figures/`: publication PNG and vector PDF figures generated from saved experiment outputs.
- `article_manifest.json`: source, result-table, experiment-manifest, frozen-specification and DOCX SHA-256 fingerprints, plus document inventory.
- `quality_checks.json`: table reconciliation, equation/reference inventory, notebook output checks, test results and final page review.
- `rendered/`: ignored local rendering and visual-review intermediates.

## Rebuild

From the repository root, using an environment with the research and optional document dependencies:

```sh
.venv/bin/python -m pip install -r requirements-docs.txt
.venv/bin/python scripts/build_article_figures.py
.venv/bin/python scripts/build_article.py
```

The figure builder requires the scientific packages in `requirements.txt`. The document builder uses `python-docx` and `lxml`; it does not refit models or execute notebooks. Its display and inline equations are editable Office Math objects. If the equations change, update both their TeX representation in the prose template and the Office Math definitions in the builder.

In Codex, the delivered DOCX was authored and rendered using its bundled document runtime, independently of the research virtual environment. After editing, export every page with a DOCX renderer and visually inspect the full document for pagination, equations, tables and figure readability. Rendering intermediates are deliberately excluded from version control.

## Evidence boundaries

Numerical tables are assembled directly from `artifacts/tables/`. The split table and fixed settings also document the frozen protocol; update these only alongside a newly validated experiment. Figures use the same saved series, feature panel and predictions. `native_equations` in the article manifest counts the 12 numbered display equations; inline mathematical symbols are additional Office Math objects.

The article reports the corrected common accounting engine. It does not reuse superseded baseline numbers, reselect parameters on the test, or treat explanatory controls as prespecified selection candidates. Its limitations include the fixed survivor universe, earlier visibility of the test period, imperfect historical statement reconstruction, idealized execution and uncertainty of tail metrics. Author affiliations, funding declarations and journal-specific submission metadata have not been invented.
