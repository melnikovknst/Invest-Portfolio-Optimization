# Submission checklist

Target journal: *Corporate Finance* / *Korportivnye Finansy* (HSE University).

## Completed in the repository

- English and Russian titles each stay below the journal's 10-word ceiling.
- Structured article follows an introduction–literature–data–methods–results–discussion–conclusion sequence.
- The structured English abstract contains 235 words and the Russian abstract 247 words.
- Seven semicolon-separated keywords are supplied in both languages, together with four JEL codes.
- Main manuscript is below 40,000 characters with spaces, including generated tables and references.
- DOCX body style is Times New Roman, 12 pt, single-spaced, with 6 pt after paragraphs.
- Tables and figures are numbered, have bilingual titles, are cited in the text, and are generated from frozen outputs.
- Figure source series and four native editable Excel charts are supplied separately in `figure_data.xlsx`.
- The 24 academic references use continuous Vancouver order and stable DOI or primary-source links.
- Code, exact dependency lock, deterministic seeds, source/split hashes, executed notebooks, daily returns, weights, and an automated audit are supplied.

## Author input still required before submission

The repository cannot infer personal metadata. Add the following to the journal submission form or a separate title page:

- full author names in the journal's required languages;
- affiliations, positions, academic degrees, city, and country;
- corresponding-author email and phone number if requested by the form;
- ORCID for every author;
- funding statement, acknowledgements, conflicts of interest, and author-contribution statement where applicable;

## Equation format

The DOCX contains native editable Microsoft Office Math equations with right-aligned numbers. This matches the current Russian author-information page, which permits MathType **or the MS Word equation editor** and excludes the obsolete Microsoft Equation format. No Cyrillic symbols occur inside formulas.

## Scientific limitations disclosed in the manuscript

- complete-history selection creates survivorship bias and is not a historical S&P 500 constituent reconstruction;
- the provider's adjusted prices are not independently reconstructed;
- SEC aggregate snapshots are externally mutable, although the exact local archive is hash-verified;
- close execution, proportional costs, and no market-impact/capacity model are simplifying assumptions;
- the original project had already displayed the 2021–2026 period, so the research program is exploratory rather than preregistered;
- paired bootstrap intervals are conditional on the selected model family and do not correct for every research choice.
