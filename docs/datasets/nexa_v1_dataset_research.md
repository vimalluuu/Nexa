# Nexa V1 Dataset Research

Date: 2026-08-13

This is a research-only report. No external datasets were downloaded, no raw
dataset directories were created, no tokenizer was retrained, and no model
training was started.

This report is not legal advice. It records a technical/license review for
dataset selection and marks uncertain cases explicitly.

## 1. Research Methodology

The review prioritized official sources over third-party summaries:

- Official dataset pages, repositories, dump indexes, and help pages.
- Official license pages or license deeds where a dataset uses a standard
  license.
- Current availability and terms, checked on 2026-08-13.
- English-first datasets suitable for a tiny, from-scratch Nexa V1 model.
- Manageable subsets rather than full multi-GB or TB-scale corpora.

Scoring uses 0-5 for each dimension:

- License clarity: explicitness and ease of compliance.
- Training permission: whether reuse/adaptation for model training appears
  allowed.
- Text quality: prose quality, editorial quality, and OCR/noise level.
- Text diversity: range of topics/styles.
- Language suitability: English relevance for Nexa V1.
- Data cleanliness: likely preprocessing burden.
- Reproducibility: official URLs, stable versions, checksums, or dump dates.
- Size suitability: whether a small, controlled subset is practical.

## 2. Candidate Datasets

### 1. Project Gutenberg

- Official source: Project Gutenberg
- Official URL: https://www.gutenberg.org/
- Download URL: https://www.gutenberg.org/ebooks/
- Version: Rolling collection; use per-ebook IDs and retrieval date
- License: Project Gutenberg License / public-domain status varies by country
- License URL: https://www.gutenberg.org/policy/license.html
- Language: Primarily English for the intended subset
- Approximate size: Full collection is large; recommended subset 10-50 MB
- Approximate document count: Official page reports about 79k ebooks
- Text type: Public-domain books, fiction, essays, reference works
- Known limitations: U.S.-centric public-domain status; boilerplate removal
  required; older language style can bias the model.
- Training use permitted?: YES for U.S. public-domain texts, subject to license
  and jurisdiction review
- Redistribution permitted?: YES, with Project Gutenberg trademark/license
  conditions if redistributing Project Gutenberg editions
- Attribution required?: Usually license/trademark notice preservation matters;
  attribution depends on redistribution format
- Commercial use permitted?: YES for unrestricted public-domain content, but
  comply with Project Gutenberg trademark rules when redistributing their files
- Personal-data concerns?: LOW
- Copyright concerns?: MEDIUM outside the U.S. or if selected works are not
  public domain in the relevant jurisdiction
- Why useful for Nexa: High-quality long-form prose and stories; ideal primary
  corpus for a tiny model.
- License confidence: HIGH for carefully selected U.S. public-domain works;
  MEDIUM internationally
- Score: license clarity 4, training permission 5, text quality 4,
  text diversity 4, language suitability 5, data cleanliness 3,
  reproducibility 4, size suitability 5. Total: 34/40.

### 2. Standard Ebooks

- Official source: Standard Ebooks
- Official URL: https://standardebooks.org/
- Download URL: https://standardebooks.org/ebooks
- Version: Rolling collection; use ebook repository/tag/date
- License: Public domain / CC0 for Standard Ebooks-produced content
- License URL: https://standardebooks.org/about/standard-ebooks-and-the-public-domain
- Language: English
- Approximate size: UNKNOWN; recommended subset 10-50 MB
- Approximate document count: UNKNOWN from official source in this pass
- Text type: Proofread public-domain ebooks
- Known limitations: Smaller than Project Gutenberg; EPUB/XHTML extraction
  needed; mostly classic literature.
- Training use permitted?: YES
- Redistribution permitted?: YES
- Attribution required?: NO for CC0/public-domain content, though source
  attribution is recommended for provenance
- Commercial use permitted?: YES
- Personal-data concerns?: LOW
- Copyright concerns?: LOW for U.S. use; still review individual works for
  non-U.S. distribution
- Why useful for Nexa: Very clean long-form prose with less boilerplate and
  better formatting consistency than raw Gutenberg text.
- License confidence: HIGH
- Score: license clarity 5, training permission 5, text quality 5,
  text diversity 3, language suitability 5, data cleanliness 4,
  reproducibility 4, size suitability 5. Total: 36/40.

### 3. Simple English Wikipedia

- Official source: Wikimedia Dumps
- Official URL: https://dumps.wikimedia.org/simplewiki/latest/
- Download URL:
  https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
- Version: latest dump checked 2026-08-13; file dated 2026-08-04
- License: CC BY-SA 4.0 and GFDL for original text, with possible public-domain
  or other licensing for some contributions
- License URL: https://dumps.wikimedia.org/legal.html
- Language: Simple English
- Approximate size: pages-articles XML bzip2 about 354 MB; multistream XML
  about 384 MB compressed
- Approximate document count: UNKNOWN from official dump index
- Text type: Encyclopedic general knowledge
- Known limitations: Wiki markup extraction required; attribution/share-alike
  compliance needed; simplified style may underrepresent natural prose.
- Training use permitted?: YES, if CC BY-SA/GFDL obligations are satisfied
- Redistribution permitted?: YES, under license terms
- Attribution required?: YES
- Commercial use permitted?: YES under CC BY-SA, with obligations
- Personal-data concerns?: LOW to MEDIUM due to public article/revision context
- Copyright concerns?: LOW to MEDIUM due to contribution and attribution
  complexity
- Why useful for Nexa: Small, manageable general-knowledge source with clear
  official dumps.
- License confidence: HIGH for license identity; MEDIUM for model-output and
  attribution compliance details
- Score: license clarity 4, training permission 4, text quality 4,
  text diversity 4, language suitability 5, data cleanliness 3,
  reproducibility 5, size suitability 4. Total: 33/40.

### 4. English Wikibooks

- Official source: Wikimedia Dumps
- Official URL: https://dumps.wikimedia.org/enwikibooks/latest/
- Download URL:
  https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2
- Version: latest dump checked 2026-08-13; file dated 2026-08-04
- License: CC BY-SA 4.0 and GFDL for original text, with possible public-domain
  or other licensing for some contributions
- License URL: https://dumps.wikimedia.org/legal.html
- Language: English
- Approximate size: pages-articles XML bzip2 about 203 MB
- Approximate document count: UNKNOWN from official dump index
- Text type: Educational textbooks and manuals
- Known limitations: Uneven quality and completeness; wiki markup cleanup
  required; attribution/share-alike obligations.
- Training use permitted?: YES, if CC BY-SA/GFDL obligations are satisfied
- Redistribution permitted?: YES, under license terms
- Attribution required?: YES
- Commercial use permitted?: YES under CC BY-SA, with obligations
- Personal-data concerns?: LOW
- Copyright concerns?: MEDIUM due to mixed community contributions and
  attribution compliance
- Why useful for Nexa: Educational style and procedural explanations diversify
  book-heavy training data.
- License confidence: HIGH for license identity; MEDIUM for compliance details
- Score: license clarity 4, training permission 4, text quality 3,
  text diversity 4, language suitability 5, data cleanliness 3,
  reproducibility 5, size suitability 4. Total: 32/40.

### 5. English Wikisource

- Official source: Wikimedia Dumps
- Official URL: https://dumps.wikimedia.org/enwikisource/latest/
- Download URL:
  https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2
- Version: latest dump checked 2026-08-13; file dated 2026-08-04
- License: CC BY-SA 4.0 and GFDL for Wikimedia-contributed text; underlying
  source works may be public domain or otherwise licensed
- License URL: https://dumps.wikimedia.org/legal.html
- Language: English
- Approximate size: pages-articles XML bzip2 about 3.4 GB
- Approximate document count: UNKNOWN from official dump index
- Text type: Source texts, books, historical documents
- Known limitations: Full dump is too large for the first Nexa V1 pass;
  source-work license verification can be complex; wiki markup cleanup needed.
- Training use permitted?: YES for compliant works, but requires careful
  filtering
- Redistribution permitted?: YES for compliant works, under relevant terms
- Attribution required?: YES for Wikimedia contribution layer; underlying
  public-domain works may not require attribution
- Commercial use permitted?: YES for CC BY-SA/public-domain works, with terms
- Personal-data concerns?: LOW
- Copyright concerns?: MEDIUM because underlying work status must be respected
- Why useful for Nexa: Broad public-domain style diversity, but best used as a
  later curated subset.
- License confidence: MEDIUM
- Score: license clarity 3, training permission 4, text quality 4,
  text diversity 5, language suitability 5, data cleanliness 2,
  reproducibility 5, size suitability 2. Total: 30/40.

### 6. OpenStax Textbooks

- Official source: OpenStax
- Official URL: https://openstax.org/
- Download URL: Per-book downloads from OpenStax book pages
- Version: Per-textbook edition
- License: CC BY-NC-SA 4.0 according to current OpenStax licensing help
- License URL:
  https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks
- Language: English, with some multilingual offerings
- Approximate size: UNKNOWN; per-book PDFs/HTML are manageable
- Approximate document count: UNKNOWN from official source in this pass
- Text type: Peer-reviewed educational textbooks
- Known limitations: NonCommercial restriction makes commercial usage not
  permitted; share-alike attribution obligations; per-book download workflow.
- Training use permitted?: LIKELY YES for noncommercial research/education;
  LICENSE UNCERTAIN for any future commercial use
- Redistribution permitted?: YES for noncommercial redistribution under
  CC BY-NC-SA
- Attribution required?: YES
- Commercial use permitted?: NO under current CC BY-NC-SA
- Personal-data concerns?: LOW
- Copyright concerns?: LOW if license terms are followed
- Why useful for Nexa: Excellent educational prose, science/math grounding,
  and high editorial quality.
- License confidence: HIGH for current noncommercial license; MEDIUM for
  suitability if Nexa later has commercial goals
- Score: license clarity 4, training permission 3, text quality 5,
  text diversity 4, language suitability 5, data cleanliness 4,
  reproducibility 4, size suitability 5. Total: 34/40.

### 7. BCcampus OpenEd Textbooks

- Official source: BCcampus OpenEd
- Official URL: https://open.bccampus.ca/
- Download URL: Per-book downloads from BCcampus/Pressbooks pages
- Version: Per-textbook edition
- License: Mostly CC BY 4.0 for BCcampus-published OER, with exceptions
- License URL:
  https://open.bccampus.ca/create-open-textbooks/creative-commons-open-licences-for-authors/
- Language: English
- Approximate size: UNKNOWN; per-book subset likely 10-100 MB
- Approximate document count: UNKNOWN
- Text type: Open textbooks and educational resources
- Known limitations: Must verify each book's license and third-party exceptions;
  corpus assembly is more manual than single dumps.
- Training use permitted?: YES for CC BY materials
- Redistribution permitted?: YES for CC BY materials
- Attribution required?: YES
- Commercial use permitted?: YES for CC BY materials
- Personal-data concerns?: LOW
- Copyright concerns?: LOW to MEDIUM due to per-book exceptions
- Why useful for Nexa: High-quality explanatory educational text under more
  permissive terms than noncommercial textbook sources.
- License confidence: MEDIUM to HIGH, depending on per-book verification
- Score: license clarity 4, training permission 5, text quality 5,
  text diversity 4, language suitability 5, data cleanliness 4,
  reproducibility 3, size suitability 5. Total: 35/40.

### 8. Federal Register Bulk XML

- Official source: Office of the Federal Register / GovInfo / Data.gov
- Official URL:
  https://www.federalregister.gov/reader-aids/developer-resources/bulk-data
- Download URL: https://www.govinfo.gov/bulkdata/FR/
- Version: Annual/daily XML files; select specific years
- License: U.S. federal government public-domain material; Federal Register
  reproduction is unrestricted under 1 CFR 2.6
- License URL:
  https://www.federalregister.gov/reader-aids/government-policy-and-ofr-procedures/about-this-site
- Language: English
- Approximate size: UNKNOWN for selected subset; one year likely manageable
- Approximate document count: About 250 issues per year according to NARA
  dataset description
- Text type: Rules, notices, executive documents, agency prose
- Known limitations: Legal/regulatory style is dense; XML cleanup required;
  may include incorporated or submitted material that needs care.
- Training use permitted?: YES for U.S. government-authored Federal Register
  material
- Redistribution permitted?: YES
- Attribution required?: NO copyright attribution required, but provenance
  should be kept
- Commercial use permitted?: YES
- Personal-data concerns?: MEDIUM; notices can mention individuals/entities
- Copyright concerns?: LOW to MEDIUM for incorporated third-party material
- Why useful for Nexa: Public-domain contemporary formal prose and government
  domain knowledge.
- License confidence: HIGH for core Federal Register material
- Score: license clarity 5, training permission 5, text quality 3,
  text diversity 3, language suitability 5, data cleanliness 3,
  reproducibility 5, size suitability 4. Total: 33/40.

### 9. GovInfo Congressional Bills / Congressional Record Bulk Data

- Official source: U.S. Government Publishing Office GovInfo
- Official URL: https://www.govinfo.gov/
- Download URL: https://www.govinfo.gov/bulkdata/
- Version: Collection/year/congress-specific
- License: U.S. federal government public-domain material, with caveats for
  non-government or third-party material
- License URL:
  https://ask.gpo.gov/s/article/What-are-the-copyright-and-use-policies-of-govinfo-content
- Language: English
- Approximate size: UNKNOWN; select years or bill-status collections to control
  size
- Approximate document count: UNKNOWN
- Text type: Legislation, reports, congressional proceedings
- Known limitations: Highly formal and repetitive; XML cleanup needed; some
  collections include scanned/OCR or non-government inserts.
- Training use permitted?: YES for U.S. government works
- Redistribution permitted?: YES for U.S. government works
- Attribution required?: NO copyright attribution required, but source
  provenance should be retained
- Commercial use permitted?: YES for public-domain government works
- Personal-data concerns?: MEDIUM for hearings/records naming individuals
- Copyright concerns?: MEDIUM for non-government inserts or contractor material
- Why useful for Nexa: Reliable public-domain civic/legal text; useful as a
  small formal-prose supplement.
- License confidence: HIGH for government-authored works; MEDIUM for mixed
  collections
- Score: license clarity 4, training permission 5, text quality 3,
  text diversity 3, language suitability 5, data cleanliness 3,
  reproducibility 5, size suitability 4. Total: 32/40.

### 10. PMC Open Access Subset, Commercial-Use-Allowed Portion

- Official source: PubMed Central / NIH National Library of Medicine
- Official URL: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- Download URL: https://pmc.ncbi.nlm.nih.gov/tools/ftp/ and AWS OA paths
- Version: Rolling; use current file list/date and article license metadata
- License: Per-article licenses; NIH notes commercial usage is limited to the
  `oa_comm` directory, which includes CC BY and CC0 articles
- License URL: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
- Language: Mostly English biomedical/scientific text
- Approximate size: Millions of articles; full subset is far too large; a
  curated CC0/CC BY sample could be 10-100 MB
- Approximate document count: Millions overall; exact current count UNKNOWN
- Text type: Biomedical and life-science research articles
- Known limitations: License varies per article; avoid noncommercial and custom
  license groups for V1; domain is too specialized for a tiny general V1 if
  overused.
- Training use permitted?: YES for articles whose license permits reuse; avoid
  NonCommercial and NoDerivatives materials for V1 unless reviewed
- Redistribution permitted?: YES for CC0/CC BY, subject to terms
- Attribution required?: YES for CC BY/CC BY-SA
- Commercial use permitted?: YES only for the commercial-use-allowed set
- Personal-data concerns?: LOW to MEDIUM in biomedical case reports
- Copyright concerns?: MEDIUM due to per-article licenses and medical privacy
  review
- Why useful for Nexa: Adds precise scientific/technical prose, but should be a
  small secondary slice.
- License confidence: MEDIUM
- Score: license clarity 3, training permission 4, text quality 5,
  text diversity 3, language suitability 5, data cleanliness 3,
  reproducibility 4, size suitability 3. Total: 30/40.

### 11. Caselaw Access Project

- Official source: Harvard Law School Library Innovation Lab
- Official URL: https://case.law/
- Download URL: https://case.law/bulk/
- Version: Rolling/bulk release; use release date
- License: CC0 1.0 Universal for CAP data according to CAP terms
- License URL: https://case.law/terms/
- Language: English
- Approximate size: Very large; official announcements describe about 40M pages
  and 6.5M cases
- Approximate document count: About 6.5M cases
- Text type: U.S. court opinions
- Known limitations: Legal style is narrow and difficult; full corpus too large;
  OCR and citation cleanup may be needed.
- Training use permitted?: YES
- Redistribution permitted?: YES
- Attribution required?: NO under CC0, but source attribution recommended
- Commercial use permitted?: YES
- Personal-data concerns?: MEDIUM; legal opinions can contain names and
  sensitive facts
- Copyright concerns?: LOW for opinions/CC0 corpus, but privacy sensitivity
  remains
- Why useful for Nexa: Clear licensing and formal reasoning prose; use only as
  a small secondary if legal text is desired.
- License confidence: HIGH
- Score: license clarity 5, training permission 5, text quality 3,
  text diversity 2, language suitability 5, data cleanliness 3,
  reproducibility 4, size suitability 2. Total: 29/40.

### 12. Chronicling America OCR

- Official source: Library of Congress
- Official URL: https://www.loc.gov/collections/chronicling-america/
- Download URL: https://www.loc.gov/collections/chronicling-america/datasets/
- Version: Rolling collection/dataset batches
- License: Public domain or no known copyright restrictions according to LOC
  rights statement
- License URL:
  https://www.loc.gov/collections/chronicling-america/about-this-collection/rights-and-access/
- Language: English and some multilingual U.S. newspapers
- Approximate size: Millions of pages; full OCR too large; small state/year
  subset could be manageable
- Approximate document count: Millions of newspaper pages
- Text type: Historical newspapers, OCR text
- Known limitations: OCR noise; historical newspaper language; may include
  personal names and sensitive historical material.
- Training use permitted?: YES for public-domain/no-known-restrictions content
- Redistribution permitted?: YES, subject to LOC rights guidance
- Attribution required?: NO copyright attribution required, but source citation
  recommended
- Commercial use permitted?: YES where public domain/no known restrictions
  applies
- Personal-data concerns?: MEDIUM
- Copyright concerns?: LOW to MEDIUM because LOC says it believes content is
  public domain or has no known restrictions, not a blanket warranty
- Why useful for Nexa: Diverse public-domain prose, but OCR quality makes it a
  later-stage candidate, not primary V1.
- License confidence: MEDIUM
- Score: license clarity 4, training permission 4, text quality 2,
  text diversity 5, language suitability 4, data cleanliness 1,
  reproducibility 4, size suitability 2. Total: 26/40.

### 13. Stack Exchange Data Dump

- Official source: Stack Exchange / Internet Archive mirrors
- Official URL: https://stackoverflow.com/help/data-dumps
- Download URL: User/account-gated current dumps; historical mirrors exist
- Version: Quarterly/historical dumps
- License: Mixed CC BY-SA versions by date and site
- License URL: https://stackoverflow.com/help/licensing
- Language: Mostly English, technical Q&A
- Approximate size: Large; per-site archives vary
- Approximate document count: UNKNOWN
- Text type: Questions, answers, comments, code snippets
- Known limitations: Current official data-dump access requires affirming no
  LLM-training intent; attribution for many users is complex; code licenses are
  not uniform; comments include personal data.
- Training use permitted?: NO for current official dump workflow if it requires
  no-LLM-training affirmation
- Redistribution permitted?: YES under CC BY-SA terms for content already
  obtained lawfully, but compliance is complex
- Attribution required?: YES
- Commercial use permitted?: CC BY-SA permits commercial use, but current access
  terms make this unsuitable
- Personal-data concerns?: MEDIUM
- Copyright concerns?: HIGH
- Why useful for Nexa: Technical Q&A would help coding/technical fluency, but
  current access and compliance risks are too high.
- License confidence: HIGH that it should be rejected for Nexa V1
- Score: license clarity 2, training permission 0, text quality 4,
  text diversity 4, language suitability 5, data cleanliness 2,
  reproducibility 2, size suitability 3. Total: 22/40.

### 14. The Pile

- Official source: EleutherAI
- Official URL: https://github.com/EleutherAI/the-pile
- Download URL: Historical release locations vary
- Version: Released 2020-12-31; subsequent access/removal issues
- License: Mixed component datasets, some with unclear or disputed copyright
  status
- License URL: UNKNOWN
- Language: Mostly English
- Approximate size: About 886 GB
- Approximate document count: UNKNOWN
- Text type: Mixed web, books, papers, code, forums, etc.
- Known limitations: Far too large; contains components with licensing concerns;
  not appropriate for the explicit licensing-safety goal.
- Training use permitted?: LICENSE UNCERTAIN
- Redistribution permitted?: LICENSE UNCERTAIN
- Attribution required?: UNKNOWN
- Commercial use permitted?: LICENSE UNCERTAIN
- Personal-data concerns?: MEDIUM to HIGH
- Copyright concerns?: HIGH
- Why useful for Nexa: Diversity, but outweighed by licensing and size risks.
- License confidence: HIGH that it should be rejected for Nexa V1
- Score: license clarity 1, training permission 1, text quality 3,
  text diversity 5, language suitability 5, data cleanliness 2,
  reproducibility 2, size suitability 0. Total: 19/40.

## 3. Official Sources

- Project Gutenberg: https://www.gutenberg.org/ and
  https://www.gutenberg.org/policy/license.html
- Standard Ebooks: https://standardebooks.org/ and
  https://standardebooks.org/about/standard-ebooks-and-the-public-domain
- Wikimedia dumps: https://dumps.wikimedia.org/ and
  https://dumps.wikimedia.org/legal.html
- Simple English Wikipedia dump:
  https://dumps.wikimedia.org/simplewiki/latest/
- English Wikibooks dump:
  https://dumps.wikimedia.org/enwikibooks/latest/
- English Wikisource dump:
  https://dumps.wikimedia.org/enwikisource/latest/
- OpenStax licensing:
  https://help.openstax.org/s/article/Licensing-information-of-OpenStax-textbooks
- BCcampus open licenses:
  https://open.bccampus.ca/create-open-textbooks/creative-commons-open-licences-for-authors/
- Federal Register bulk data:
  https://www.federalregister.gov/reader-aids/developer-resources/bulk-data
- Federal Register copyright policy:
  https://www.federalregister.gov/reader-aids/government-policy-and-ofr-procedures/about-this-site
- GovInfo bulk data: https://www.govinfo.gov/bulkdata/
- GovInfo copyright policy:
  https://ask.gpo.gov/s/article/What-are-the-copyright-and-use-policies-of-govinfo-content
- PMC OA Subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- PMC AWS licensing note: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
- Caselaw Access Project: https://case.law/ and https://case.law/terms/
- Chronicling America datasets:
  https://www.loc.gov/collections/chronicling-america/datasets/
- Chronicling America rights:
  https://www.loc.gov/collections/chronicling-america/about-this-collection/rights-and-access/
- Stack Exchange data dumps: https://stackoverflow.com/help/data-dumps
- Common Pile reference/replacement direction:
  https://huggingface.co/common-pile and https://arxiv.org/abs/2506.05209

## 4. License Analysis

Best license clarity:

- Standard Ebooks: CC0/public-domain-oriented, high confidence.
- Caselaw Access Project: CC0, high confidence.
- Federal Register: official reproduction rule and U.S. government public-domain
  basis, high confidence.
- Project Gutenberg: high confidence for U.S. public-domain works, but use
  careful per-work selection and preserve license/trademark constraints.

Usable with obligations:

- Simple English Wikipedia, English Wikibooks, and English Wikisource are useful
  but carry CC BY-SA/GFDL attribution and share-alike obligations.
- BCcampus CC BY materials are strong, but each book must be checked for
  exceptions.
- PMC OA can be useful only after filtering by article-level license.

Not recommended for V1:

- OpenStax is high quality but currently NonCommercial/ShareAlike according to
  the official licensing help page. It may be acceptable for private
  noncommercial learning, but it is not ideal for a clean future-proof V1
  training set.
- Stack Exchange is rejected because the current official download workflow
  includes a no-LLM-training affirmation.
- The Pile is rejected for licensing uncertainty and size.

## 5. Size Analysis

For rough planning, plain English text often ranges around 4-6 characters per
token. These are planning estimates only; Nexa's tokenizer may differ.

| Raw text target | Approx tokens | Storage implication | Training implication |
|---:|---:|---|---|
| 10 MB | 1.7M-2.5M | Tiny; easy to inspect | Good smoke-test corpus, likely overfit quickly |
| 50 MB | 8M-12M | Still manageable | Good first serious V1 range |
| 100 MB | 17M-25M | Manageable on local disk | Reasonable upper initial target |
| 500 MB | 85M-125M | More preprocessing time | Likely excessive for 803K parameters at first |
| 1 GB | 170M-250M | Heavy for iteration | Not recommended until model/hardware scale up |

The current model is about 803K parameters. For the first real training corpus,
50-100 MB of cleaned text is a realistic target. It is large enough to improve
over toy data but small enough to inspect, deduplicate, split, tokenize, and
iterate without turning Phase 8 into an infrastructure project.

## 6. Quality Analysis

Highest quality:

- Standard Ebooks: carefully proofread, consistent formatting.
- BCcampus/OpenStax textbooks: high educational quality, but OpenStax license
  limits commercial reuse.
- Project Gutenberg: strong content, but edition quality and boilerplate vary.

Good but needs cleanup:

- Simple English Wikipedia and English Wikibooks require wiki markup extraction.
- Federal Register/GovInfo XML needs structural extraction and filtering.
- PMC OA needs article-license filtering and removal of references/tables where
  needed.

Lower first-pass quality:

- Chronicling America OCR is historically valuable but noisy.
- Legal corpora are clean enough structurally but too domain-specific.

## 7. Risk Analysis

- License obligations: CC BY and CC BY-SA sources require attribution tracking.
- Share-alike uncertainty: CC BY-SA model-training implications are debated;
  use only with clear attribution manifests and avoid as the sole primary.
- NonCommercial restriction: OpenStax and CK-12-like sources are poor choices if
  Nexa might later be used commercially.
- Personal data: legal, biomedical, historical newspaper, and Q&A sources can
  contain names or sensitive details.
- OCR noise: Chronicling America and some historical/legal sources need heavy
  validation.
- Domain imbalance: legal/scientific/government text can make a tiny model
  overly formal if overrepresented.

## 8. Candidate Scoring

| Rank | Dataset | License | Score |
|---:|---|---|---:|
| 1 | Standard Ebooks | Public domain / CC0 | 36 |
| 2 | BCcampus OpenEd Textbooks | Mostly CC BY 4.0, per-book exceptions | 35 |
| 3 | Project Gutenberg | PG License / public-domain works | 34 |
| 3 | OpenStax Textbooks | CC BY-NC-SA 4.0 | 34 |
| 5 | Simple English Wikipedia | CC BY-SA 4.0 / GFDL | 33 |
| 5 | Federal Register Bulk XML | Public domain / 1 CFR 2.6 | 33 |
| 7 | English Wikibooks | CC BY-SA 4.0 / GFDL | 32 |
| 7 | GovInfo Congressional Bulk Data | U.S. government public domain, caveats | 32 |
| 9 | English Wikisource | Mixed public domain + CC BY-SA/GFDL layer | 30 |
| 9 | PMC OA Commercial-Use-Allowed | Per-article CC0/CC BY/etc. | 30 |
| 11 | Caselaw Access Project | CC0 | 29 |
| 12 | Chronicling America OCR | Public domain/no known restrictions | 26 |
| 13 | Stack Exchange Data Dump | Mixed CC BY-SA plus current access limits | 22 |
| 14 | The Pile | Mixed/uncertain | 19 |

## 9. Recommended Primary Dataset

Primary: Standard Ebooks, 25-40 MB cleaned subset.

Why:

- Highest combined score.
- Excellent prose quality and low cleanup burden.
- Clear public-domain/CC0 posture.
- Manageable for a tiny 803K-parameter model.
- Strong fit for early language modeling from scratch.

Fallback primary if Standard Ebooks subset is too small: Project Gutenberg,
carefully selected public-domain English works with boilerplate removed and
per-work metadata retained.

## 10. Recommended Secondary Datasets

Recommended secondary sources:

- Project Gutenberg, 20-40 MB cleaned subset, to widen author/style coverage.
- Simple English Wikipedia, 10-20 MB cleaned subset, for concise general
  knowledge.
- BCcampus CC BY textbooks, 5-15 MB verified subset, for educational prose.
- Federal Register, 2-5 MB cleaned subset, only if a small amount of formal
  contemporary government prose is desired.

Not recommended for V1 primary:

- OpenStax because of NonCommercial restrictions.
- PMC OA because article-level filtering and domain imbalance make it a later
  secondary.
- Caselaw Access Project because legal prose is too narrow for a tiny general
  model.
- Chronicling America because OCR noise would dominate cleaning work.

## 11. Recommended Final Dataset Composition

For the first real Nexa V1 corpus, target about 75 MB cleaned text:

- 45% Standard Ebooks: clean literary/general prose.
- 25% Project Gutenberg: broader public-domain books and essays.
- 20% Simple English Wikipedia: concise general knowledge.
- 10% BCcampus CC BY textbook chapters: educational explanatory prose.

Justification:

- The majority stays high-quality long-form human prose.
- General knowledge is present but not dominant.
- Educational text adds clarity and expository structure.
- All selected sources can be kept small and independently identifiable.

Do not include multilingual text in V1. A tiny 803K-parameter model has limited
capacity, and multilingual data would increase vocabulary pressure and reduce
English quality. Revisit multilingual data after the tokenizer and model scale.

## 12. Estimated Total Size

Recommended first target: about 75 MB cleaned UTF-8 text.

Acceptable range: 50-100 MB.

Avoid 500 MB or 1 GB at this phase unless hardware and model size are scaled.

## 13. Estimated Token Count

For 75 MB cleaned text:

- Approximate token range: 12M-19M tokens.
- Planning midpoint: about 15M tokens.

This is enough for many epochs/steps on an 803K-parameter model and should make
overfitting/data-quality problems visible without requiring enormous storage.

## 14. Expected Storage Requirements

For a 75 MB cleaned target, expect approximately:

- Raw source archives/files: 100-500 MB depending on XML/EPUB/plain-text source.
- Cleaned text: about 75 MB.
- Deduplicated text: about 60-75 MB.
- Train/validation splits: about 60-75 MB total.
- Tokenized processed files: roughly 50-150 MB depending on integer encoding.
- Manifests/reports: small text files.

These are local artifacts and must not be committed to Git except metadata,
manifests, reports, scripts, docs, and tiny permitted samples.

## 15. Expected Training Implications

- 10 MB: Useful for validating the full pipeline; model will likely overfit.
- 50 MB: Good first real training run.
- 75 MB: Recommended balance for Nexa V1.
- 100 MB: Reasonable upper bound if preprocessing is smooth.
- 500 MB+: Better deferred until model and training infrastructure scale.
- 1 GB: Not appropriate for the current tiny model as the first real dataset.

The first run should emphasize reproducibility and loss/evaluation behavior,
not raw scale. Keep validation split document-level where possible.

## 16. Rejected Datasets and Reasons

- Stack Exchange Data Dump: rejected for V1 because current official access
  requires affirming no LLM-training intent; attribution and code-license
  complexity is high.
- The Pile: rejected for V1 due to size and mixed/uncertain licensing.
- C4/Common Crawl-derived corpora: not selected because source pages are web
  scraped with unclear copyright/license status for Nexa's license-safety goal.
- BookCorpus: not selected because copyright/license provenance is unclear.
- OpenWebText: not selected because it reconstructs web-linked Reddit content
  and does not provide sufficiently clear licensing for the underlying pages.
- OpenStax: not rejected completely for private noncommercial research, but not
  recommended for the V1 dataset because current official licensing is
  NonCommercial ShareAlike.
- Chronicling America: deferred because OCR noise creates too much cleanup risk
  for the first V1 dataset.
- Caselaw Access Project: deferred because it is too domain-specific and very
  large, despite excellent CC0 licensing.

## License Decision

Selected for recommendation:

- Standard Ebooks: selected as primary. The official Standard Ebooks public
  domain/CC0 posture permits training, redistribution, and commercial use with
  low restriction. Keep provenance anyway.
- Project Gutenberg: selected as secondary. Use only works verified as
  public-domain in the relevant jurisdiction and preserve Project Gutenberg
  license/trademark constraints when redistributing their editions.
- Simple English Wikipedia: selected as secondary with obligations. License
  permits sharing/adaptation, including commercial use, but attribution and
  share-alike compliance must be tracked.
- BCcampus CC BY textbooks: selected as secondary only after per-book license
  verification. CC BY permits redistribution/adaptation/commercial use with
  attribution, but exceptions must be excluded.

Explicitly uncertain or restricted:

- OpenStax: high educational value, but current CC BY-NC-SA license makes it
  unsuitable if Nexa has any future commercial path.
- PMC OA: use later only after filtering article-level licenses; for any
  commercial-compatible path, limit to the official `oa_comm` CC0/CC BY set.
- Wikimedia share-alike sources: usable, but model-output/share-alike legal
  interpretation should not be overstated.

## Final Recommendation

Recommended V1 research outcome:

- Primary dataset: Standard Ebooks.
- Secondary datasets: Project Gutenberg, Simple English Wikipedia, and a small
  verified BCcampus CC BY subset.
- Initial cleaned target: about 75 MB.
- Estimated tokens: about 15M tokens.
- Language strategy: English only for V1.
- Dataset stages: preserve raw, cleaned, deduplicated, splits, processed,
  manifests, and reports separately when downloading is later approved.

Stop here until explicit approval is given to download sources.
