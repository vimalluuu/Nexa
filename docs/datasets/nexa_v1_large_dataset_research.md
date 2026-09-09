# Nexa V1 Large Dataset Research

This is a research-only decision record. No candidate dataset was downloaded,
extracted, cleaned, tokenized, or used for training. No model, tokenizer, or
training-loop code was changed.

## 1. Methodology

This review was performed on 2026-09-09. It prioritizes original dataset
cards, source repositories, papers, and first-party terms over summaries. A
dataset-level license is evaluated separately from rights in the documents it
contains. When a primary source does not establish a fact, the report says
UNKNOWN rather than inferring it.

The selection rule is deliberately stricter than "publicly downloadable":

1. Nexa may use text data, but must not import pretrained model weights,
   pretrained tokenizers, pretrained embeddings, external LLM APIs, or
   intentionally AI-generated training data.
2. A corpus packaged under ODC-By, Apache-2.0, MIT, or CC0 does not
   automatically grant rights in third-party webpages, books, papers, or
   posts within it.
3. A Common Crawl-derived corpus is a conditional candidate, not a
   license-cleared candidate. Common Crawl says individual content can carry
   separate owner terms, requires compliance with applicable law, and places
   downstream AI-system risk on the user.[^1]
4. Provider token counts are not Nexa BPE token counts. They use each
   provider's tokenizer or counting rule and are useful only for comparison.
   The final Nexa count must be measured after the selected normalized corpus
   is available and Nexa's own BPE tokenizer has been trained.

The candidate scores use 0 through 5:

| Score | Meaning |
| --- | --- |
| 5 | Strong fit, clear evidence, and practical now |
| 4 | Good fit with manageable obligations |
| 3 | Usable only with meaningful caveats or extra review |
| 2 | Major uncertainty or poor fit for this phase |
| 1 | Material blocker |
| 0 | Not usable for the stated Nexa V1 constraints |

The checked-out repository is at adc7e73, while the supplied brief names
caca4cc. This report records the observed checkout rather than treating the
two commits as interchangeable. The working tree was clean before this
research artifact was created.

## 2. Hardware Information

| Item | Observed result | Implication |
| --- | --- | --- |
| CPU | AMD Ryzen 5 5600G, 6 cores / 12 logical processors, 3.9 GHz reported maximum | Capable of small streaming and preprocessing jobs; CPU-only transformer training will be slow. |
| RAM | 15.37 GiB | Do not materialize large datasets in memory; stream and shard future preprocessing. |
| GPU | AMD Radeon(TM) Graphics; Windows reports 0.5 GiB dedicated memory | This is not a practical dedicated training GPU for the planned model. |
| PyTorch | 2.13.0+cpu | CPU build only. |
| CUDA | torch.cuda.is_available() is False; nvidia-smi is unavailable | No CUDA acceleration or CUDA VRAM is available. |
| Workspace disk | F: has 336.78 GiB free | Enough for controlled subsets and working space, not full web-scale corpora. |
| System disk | C: has 3.36 GiB free | Future dataset caches and temporary files must be directed to F:, not left at defaults on C:. |

The repository's default model configuration describes a roughly 41.6M
parameter Nexa-Small configuration with a 32K vocabulary, 512 hidden width,
6 layers, and 1024-token context. The supplied brief separately describes an
approximately 803K-parameter toy model. This report does not change or resolve
that configuration/checkpoint distinction. It matters because 10M unique
tokens are a sensible pipeline/data pilot, not a compute-optimal training
budget for a 41.6M-parameter model.

### Training-scale feasibility

The following estimates concern a selected subset only, not a full candidate
corpus. They assume normalized English text at roughly 4 to 5 bytes per token,
compressed source artifacts at roughly 2 to 4 bytes per token, uint16 token
IDs plus document offsets at 2 to 3 bytes per token, and temporary
clean/deduplication space. A 32K Nexa vocabulary fits uint16 IDs; using
int32 IDs raises the token-storage portion. Model checkpoints and the Python
environment are excluded.

| Unique-data scale | Raw selected source | Cleaned text | Token IDs + index | Conservative peak workspace | Current-machine assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| 10M tokens | 0.03-0.08 GiB | 0.04-0.05 GiB | 0.02-0.03 GiB | 0.11-0.17 GiB | Storage and preprocessing feasible; appropriate first training/pipeline pilot. |
| 50M tokens | 0.10-0.20 GiB | 0.20-0.25 GiB | 0.10-0.15 GiB | 0.55-0.85 GiB | Storage feasible; CPU-only training must first be benchmarked on 10M. |
| 100M tokens | 0.20-0.40 GiB | 0.40-0.50 GiB | 0.20-0.30 GiB | 1.1-1.7 GiB | Storage feasible; defer until a 10M run has measured throughput and memory. |
| 500M tokens | 1-2 GiB | 2-2.5 GiB | 1-1.5 GiB | 5.5-8.5 GiB | Disk can hold a selected subset, but CPU-only training is not a realistic next experiment. |
| 1B tokens | 2-4 GiB | 4-5 GiB | 2-3 GiB | 11-17 GiB | Disk may hold the subset, but leaves little RAM/disk margin and is deferred pending a suitable GPU and a staged data pipeline. |

These are storage estimates, not measured training times. GPU availability,
actual batch size, context length, precision support, and the eventual model
size determine training feasibility. The data scale is not the immediate
storage constraint; CPU-only training is.

## 3. Candidate Datasets

### 3.1 FineWeb

- Official source: [Hugging Face FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
- License stated by publisher: [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), with use also subject to the [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use).
- Size: more than 18.5T provider-reported tokens of cleaned, deduplicated
  English web text; full on-disk size is multi-terabyte and is not feasible
  locally.[^2]
- Composition and provenance: webpages from Common Crawl dumps since 2013.
  The release documents its crawl lineage and processing code.
- Quality: main-text extraction, URL filtering, heuristic filtering, and
  deduplication are documented. FineWeb deliberately avoided ML-based
  quality/toxicity filtering; the card still warns that toxic content and web
  biases remain.[^2]
- Duplicates: deduplicated per crawl rather than globally by design. Therefore
  cross-crawl repetition remains a consideration for a locally sampled
  subset.[^2]
- Language and code: English; the card says code is not prevalent.
- Synthetic-data status: UNKNOWN. A broad web crawl cannot prove that all
  post-generative-AI webpages are human-authored.
- Training fit: excellent quality/reproducibility and easy subsetting, but
  not clear per-document training rights. It is **conditionally deferred**,
  not recommended under the current strict rights gate.

### 3.2 Dolma v1.7

- Official sources: [Dolma repository](https://github.com/allenai/dolma),
  [Dolma dataset card](https://huggingface.co/datasets/allenai/dolma), and
  [AI2's v1.7 description](https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d).
- License stated by publisher: ODC-By for the dataset. The repository
  separately carries Apache-2.0 for toolkit code; the two must not be
  conflated.[^3]
- Size: the general Dolma documentation describes 3T tokens; AI2's v1.7
  experimental mixture is described as about 2.3T tokens.[^3][^4]
- Composition: web text, academic publications, code, books, encyclopedic
  material, and other source-specific subsets.
- Quality: source-specific filtering and documented toolkit support for
  Bloom-filter deduplication. Diversity is strong, but exact quality and
  licensing vary by source.
- Independence concern: v1.7 has a FLAN instruction-data component; AI2's
  own comparison work exposes a "no Flan" ablation.[^5] The documentation
  inspected does not establish that the complete v1.7 mixture meets Nexa's
  no-AI-generated-training-data rule. Its status is therefore UNKNOWN rather
  than assumed compliant.
- Training fit: **deferred**. It requires a source-level recipe excluding
  disallowed/uncertain components and still inherits web-source rights issues.

### 3.3 RedPajama-Data-v2

- Official source: [Together Computer's repository](https://github.com/togethercomputer/RedPajama-Data).
- License stated by repository: Apache-2.0 for repository code. The project
  explicitly directs dataset users to Common Crawl's terms; that is not a
  per-document content license.[^6]
- Size: 20.8B deduplicated annotated head-middle documents, estimated at
  30.4T tokens: 20.5T English, with German, French, Spanish, and Italian
  also present.[^6]
- Provenance: 84 Common Crawl snapshots, processed using CCNet; 30B documents
  have quality signals and 20B have deduplication information.
- Quality: useful quality and MinHash signals are supplied, but no final
  universally safe/high-quality slice is preselected for Nexa.
- Synthetic-data status: UNKNOWN for web material.
- Training fit: **deferred**. It is far beyond local scale and inherits the
  Common Crawl ownership/terms issue.

### 3.4 The Pile

- Official source: [EleutherAI's The Pile repository](https://github.com/EleutherAI/the-pile).
- License: mixed by component; no single content license establishes training
  rights for all 22 constituents.
- Size: roughly 825 GiB in the official release description; a single
  authoritative corpus-wide token total was not verified in the official
  package documentation, so token count is UNKNOWN for this report.
- Composition: 22 sources across web, books, academic papers, code, forums,
  legal/government text, and reference material.[^7]
- Quality and duplicates: broad and well documented, but quality and
  duplication vary by component. The official repository's component table
  includes sources with material outside Nexa's safe V1 policy, including
  Books3.
- Training fit: **rejected for Nexa V1** because mixed/uncertain underlying
  rights, code and non-English/source mixture control, and local size.

### 3.5 C4 English

- Official source: [C4 dataset card](https://huggingface.co/datasets/allenai/c4)
  and [original C4 paper](https://arxiv.org/abs/1910.10683).
- License stated by publisher: ODC-By, additionally subject to Common Crawl
  terms.[^8]
- Size: English configuration is 305 GB with about 364.9M training documents;
  the original paper reports approximately 156B tokens.
- Composition and provenance: English-only text filtered from Common Crawl,
  with URL, text, and timestamp fields.
- Quality: language identification, boilerplate/natural-language heuristics,
  and extensive deduplication are documented. The documented bad-word filter
  is a known quality/bias concern.
- Synthetic-data status: lower historical exposure than recent crawls, but
  still UNKNOWN at document level.
- Training fit: **deferred**. It can be streamed and subsetted, but it retains
  the same substantive Common Crawl rights caveat as FineWeb.

### 3.6 SlimPajama-627B

- Official source: [Cerebras SlimPajama dataset card](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
  and [Cerebras release note](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama).
- License: component-dependent. A documented downstream split explicitly
  tells users to follow licenses of the selected subsets, including Common
  Crawl, C4, GitHub, books, and arXiv terms.[^9]
- Size: 627B reported tokens; an openly documented split is about 883 GB
  compressed.[^9]
- Composition: a cleaned/deduplicated RedPajama mixture spanning Common Crawl,
  C4, GitHub, books, arXiv, Wikipedia, and StackExchange-derived material.
- Quality: cleaning and deduplication are strengths, but they do not resolve
  the source-rights differences.
- Training fit: **deferred**. It is too large locally and lacks a single
  license/training-clearance answer.

### 3.7 RefinedWeb public extract

- Official source: [RefinedWeb paper and datasheet](https://proceedings.neurips.cc/paper_files/paper/2023/file/fa3ed726cc5073b9c31e3e49a807789c-Supplemental-Datasets_and_Benchmarks.pdf).
- License stated by publisher: ODC-By 1.0 plus Common Crawl terms.[^10]
- Size: the full English corpus is about 5T tokens; the public release is a
  600B-token random extract.
- Composition and provenance: Common Crawl pages through June 2023, with
  source-page URL and crawl-origin metadata.
- Quality: heuristic filters, adult URL filtering, language identification,
  strict exact/fuzzy deduplication, and substring deduplication are
  documented. The datasheet says toxic content remains comparable to The
  Pile.[^10]
- Synthetic-data status: UNKNOWN.
- Training fit: **deferred**. Documentation is unusually strong, but public
  data remains Common Crawl-derived and is far beyond local scale.

### 3.8 OSCAR 23.01

- Official sources: [OSCAR documentation](https://oscar-project.github.io/documentation/quickstart/)
  and [OSCAR 23.01 dataset card](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301).
- License: the project places CC0 on packaging, metadata, and annotations;
  it expressly says it does not own the crawled text.[^11]
- Size: 3.57 TB on the dataset card; a single corpus-wide token count for
  OSCAR 23.01 was not verified here, so it is UNKNOWN.
- Composition: Common Crawl WET text, document-oriented, across more than
  150 languages.
- Quality: original and deduplicated language variants exist, but the card
  warns that filtering/quality varies and personal/sensitive information may
  be present.
- Access and training use: access is gated and the card states that access
  was temporarily suspended; it also distinguishes CC0 metadata from content
  rights.[^11]
- Training fit: **rejected for Nexa V1** due to gated availability,
  multilingual scope, source ownership uncertainty, and local scale.

### 3.9 ROOTS

- Official source: [The BigScience ROOTS Corpus paper](https://papers.nips.cc/paper_files/paper/2022/file/ce9e92e3de2372a4b93353eb7f3dc0bd-Paper-Datasets_and_Benchmarks.pdf).
- License and access: a preliminary subset was released gated behind
  commitment to the BigScience ethical charter; the corpus joins hundreds of
  constituent datasets with source-level governance requirements.[^12]
- Size: 1.6 TB; 59 languages, comprising 46 natural and 13 programming
  languages. A verified corpus-wide token count is UNKNOWN for this report.
- Composition/provenance: 62% from a documented community-selected source
  list and 38% from OSCAR-derived web text.[^12]
- Quality: unusually thoughtful documentation, source review, cleaning, and
  cross-pipeline deduplication.
- Training fit: **deferred**. It is a good research reference but poorly
  matched to English-first, CPU-only Nexa V1 and carries a complex
  source-by-source governance burden.

### 3.10 Wikimedia Wikipedia English

- Official sources: [Wikimedia's processed Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia),
  [Wikimedia dumps legal page](https://dumps.wikimedia.org/legal.html), and
  [Wikimedia reuse guidance](https://foundation.wikimedia.org/wiki/Legal:Wikimedia_Developer_App_Guidelines).
- License: original text is made available under CC BY-SA and GFDL; the
  processed dataset card identifies CC BY-SA 3.0/GFDL for its release.[^13]
- Size: 6.4M cleaned English articles in the 2023-11-01 English configuration.
  The corpus is multi-billion-token scale; a precise Nexa-BPE count is
  UNKNOWN until Nexa processes the selected version.
- Composition/provenance: one cleaned article per record, built from official
  Wikimedia dumps, with title and URL fields. This is much more traceable
  than broad web crawl data.
- Quality: encyclopedic prose, strong factual/general-knowledge coverage,
  consistent document structure, and a transparent source. Limitations are
  editor bias, uneven article quality, citations/markup cleanup, and limited
  stylistic diversity.
- Training-use clarity: CC BY-SA expressly permits sharing and adaptation,
  including commercial use, subject to attribution/share-alike
  conditions.[^14] Whether model weights or model outputs are legally
  "Adapted Material" is not resolved by this report; preserve attribution and
  seek jurisdiction-specific legal advice before any public distribution.
- Synthetic-data status: not certified at document level. It is not packaged
  as an AI-generated corpus.
- Training fit: **recommended primary source** because it best satisfies
  provenance, reproducibility, English focus, and documented licensing while
  remaining large enough to sample without manual book collection.

### 3.11 PG-19

- Official source: [Google DeepMind PG-19 repository](https://github.com/google-deepmind/pg19)
  and [Project Gutenberg license policy](https://www.gutenberg.org/policy/license).
- License: PG-19's repository metadata says Apache-2.0, but that describes
  the release and cannot substitute for rights in the extracted books. The
  underlying texts are Project Gutenberg works published before 1919; Project
  Gutenberg says most are unrestricted under U.S. copyright law but requires
  country-specific review and inspection of individual restrictions.[^15]
- Size: 28,602 train books and 1,973,136,207 provider word-level tokens,
  plus validation/test books.[^15]
- Composition/provenance: English Project Gutenberg books published before
  1919, with book IDs, titles, and publication dates. Boilerplate license text
  was removed and selected offensive terms were mapped to placeholders.
- Quality: long-form coherent prose, very low repetition versus web crawl,
  and no contemporary generative-web contamination. Its own authors warn
  against using it alone for a general-purpose dialogue model because of dated
  language and historical bias.[^15]
- Training fit: **recommended limited secondary source**, capped at 20% of
  the initial mix. It is an existing dataset, not the prior manual
  200-book collection plan.

### 3.12 PMC Open Access subset, commercial-reuse-filtered

- Official sources: [PMC copyright guidance](https://pmc.ncbi.nlm.nih.gov/about/copyright/),
  [PMC Cloud Service guidance](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/), and
  [PMC OAI-PMH API](https://pmc.ncbi.nlm.nih.gov/tools/oai/).
- License: per article/version. PMC supports filtering for CC0, CC BY,
  CC BY-SA, and CC BY-ND commercial-reuse categories, but says users remain
  responsible for compliance and must inspect the individual license
  metadata.[^16]
- Size: millions of articles are available in the open-access subset; an
  exact all-license-filtered token total is UNKNOWN.
- Composition/provenance: biomedical and life-science articles/preprints,
  with article-level metadata and approved bulk retrieval channels.
- Quality: high technical prose and strong provenance, but narrow domain,
  scientific formatting, and possible third-party figures/illustrations.
- Training fit: **optional later secondary only**. It should not be part of
  the first 10M-token run because it adds operational license filtering and
  domain imbalance.

## 4. License Analysis

### What ODC-By does and does not settle

ODC-By is a database attribution license. It permits use, modification, and
sharing of the licensed database subject to attribution, but it cannot grant
copyright permissions the distributor does not own.[^17] FineWeb, Dolma,
C4, RefinedWeb, and many related releases use ODC-By, yet their sources
include third-party webpages. This is why a permissive dataset card is not a
per-document safe harbor.

### Common Crawl consequence

FineWeb, RedPajama, C4, RefinedWeb, and OSCAR all rely materially on Common
Crawl. Common Crawl says content can be subject to separate owner terms and
that users must evaluate lawfulness and respect third-party rights. Its current
terms also include downstream indemnification relating to AI systems.[^1]
That makes such datasets potentially useful research corpora, but unsuitable
as an unqualified recommendation under Nexa's explicit license-clarity goal.

### Clearer but still conditional paths

Wikimedia provides an identifiable contributor/license regime. PG-19 has
stable book-level provenance and historical works, but Project Gutenberg's
country-specific and individual-work caveats still matter. PMC has
machine-readable article-level licenses, but a future collection must select
only explicitly permitted records and must omit third-party assets where
needed. None of these statements is legal advice.

## 5. Provenance Analysis

| Candidate group | Provenance strength | Main gap |
| --- | --- | --- |
| Wikimedia | Official dumps, title/URL, article history, known licenses | Contributor-level attribution handling and article-quality variation |
| PG-19 | Book ID/title/publication-date metadata; fixed benchmark | Verify source-book jurisdiction/restrictions before public release |
| PMC OA | Article/version metadata and per-record license fields | Must filter every selected article/version; figures can have separate rights |
| FineWeb / C4 / RefinedWeb / RedPajama / OSCAR | Crawl/date and often URL metadata | Source pages are third-party web content with nonuniform rights |
| Dolma / The Pile / SlimPajama / ROOTS | Well documented mixtures | Compositional provenance does not collapse to one permission answer |

## 6. Data Quality Analysis

The web-scale candidates have scale and diversity, but filtering cannot
guarantee removal of spam, boilerplate, copied content, harmful material,
personal information, or AI-authored webpages. FineWeb and RefinedWeb are the
best documented quality-focused web candidates in this comparison; that is a
quality finding, not a licensing clearance.

Wikimedia is the best first-source tradeoff for a small model: it has clean
article boundaries, coherent exposition, and a reproducible snapshot. It is
not enough by itself to teach broad conversational, narrative, or technical
styles. PG-19 supplements longer narrative prose but must be held to a small
share due to dated language. PMC is high quality but should be deferred rather
than distort a general-English first run.

No candidate can be certified free of every duplicate. A future selected
10M-token corpus still needs local exact and near-duplicate checks across its
own selected documents. That does not mean downloading or preprocessing has
started in this phase.

### Per-candidate quality and filtering review

| Candidate | Duplicates, spam, and boilerplate | Synthetic/AI-text status | Code and language mix | Document length | Additional filtering needed for Nexa? |
| --- | --- | --- | --- | --- | --- |
| FineWeb | Per-crawl deduplication and main-text filtering; web spam/boilerplate and cross-crawl repetition can remain | UNKNOWN | English; code reported as not prevalent | Variable web pages | Yes: selected-subset duplicate check, source/URL review, and sensitive-content policy |
| Dolma v1.7 | Source-specific filtering/deduplication; heterogeneous quality remains | UNKNOWN; FLAN instruction component requires a separate compliance decision | Mixed English sources, including code | Variable by source | Yes: explicit source allow-list; omit code/instruction/uncertain sources |
| RedPajama-v2 | Quality signals and deduplication metadata are supplied, not a complete safety guarantee | UNKNOWN | Five languages; code share UNKNOWN | Variable web pages | Yes: English-only source selection, quality threshold, duplicate check |
| The Pile | Component-dependent; some sources contain web noise or reused material | UNKNOWN | Predominantly English with material code | Highly variable by component | Yes, but rights failures make V1 filtering uneconomical |
| C4 English | Heuristic cleanup/deduplication; known blocklist/bias issues | UNKNOWN | English; code share UNKNOWN | Variable web pages | Yes: audit blocklist effects, sampling, duplicate/sensitive-content checks |
| SlimPajama | Cleaned/deduplicated; inherits source-specific noise and rights | UNKNOWN | English with a code component | Variable by component | Yes: component allow-list and a fresh cross-source duplicate check |
| RefinedWeb | Strong exact/fuzzy/substring deduplication; toxic content and web noise can remain | UNKNOWN | English; code share UNKNOWN | Variable web pages | Yes: selected-subset audit and source-rights review |
| OSCAR 23.01 | Web-derived; quality varies by language and sensitive data may remain | UNKNOWN | More than 150 languages | Variable web pages | Yes, but access/rights blockers come first |
| ROOTS | Source-specific cleaning plus cross-pipeline deduplication | UNKNOWN | 59 languages, including programming languages | Variable by 498 sources | Yes: English/no-code source recipe and governance review |
| Wikimedia English | Article boundaries are clean; no crawl spam, but editorial/revision duplication and markup/citation artifacts need review | Not packaged as AI-generated; document-level zero-AI certification UNKNOWN | English, low code | From short stubs to long articles | Yes: namespace/template/list/citation cleanup and local deduplication |
| PG-19 | Long coherent books; no web spam; corpus-wide duplicate status UNKNOWN | Historical books predate generative AI | English, no expected code | Long; provider says average documents are about 20 times WikiText long | Yes: individual source-rights check and a small-share cap for historical-style bias |
| PMC OA | Structured peer-reviewed text; article versions, tables, references, and third-party figures require care | UNKNOWN at article level | Mostly English, low code | Long structured articles | Yes: CC0/CC-BY license filter, text-only extraction, article-version deduplication |

## 7. Size Analysis

The full corpus sizes rule out local full downloads:

| Dataset | Provider-reported scale | Full corpus locally realistic? |
| --- | ---: | --- |
| FineWeb | More than 18.5T tokens | No |
| Dolma v1.7 | About 2.3T tokens | No |
| RedPajama-v2 | 30.4T deduplicated head-middle tokens | No |
| The Pile | About 825 GiB | No; exceeds safe working-space margin |
| C4 English | 305 GB | No; leaves too little processing margin |
| SlimPajama | 627B tokens / about 883 GB compressed | No |
| RefinedWeb public extract | 600B tokens | No |
| OSCAR 23.01 | 3.57 TB | No |
| ROOTS | 1.6 TB | No |
| Wikimedia English | Multi-billion-token supply; sampleable | Yes, as a small selected subset only |
| PG-19 | 1.97B provider word tokens | Yes, as a small selected subset only |
| PMC OA | Millions of documents; token count UNKNOWN | Yes, only after record-level license filtering |

## 8. Candidate Scoring

Scores below reflect the actual Nexa constraints and current hardware. They
are not a claim that one database license resolves the rights of all contained
documents.

| Candidate | License clarity | Training-use clarity | Text quality | Diversity | Provenance | Reproducibility | Size suitability | Current-hardware suitability | Total / 40 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FineWeb | 3 | 2 | 4 | 4 | 4 | 5 | 5 | 4 | 31 | Conditional defer |
| Dolma v1.7 | 3 | 2 | 4 | 5 | 4 | 5 | 5 | 4 | 32 | Defer |
| RedPajama-v2 | 2 | 2 | 3 | 4 | 4 | 5 | 4 | 4 | 28 | Defer |
| The Pile | 1 | 1 | 3 | 5 | 2 | 3 | 3 | 1 | 19 | Reject |
| C4 English | 3 | 2 | 3 | 3 | 4 | 5 | 5 | 4 | 29 | Defer |
| SlimPajama-627B | 2 | 2 | 4 | 4 | 3 | 4 | 4 | 3 | 26 | Defer |
| RefinedWeb public extract | 3 | 2 | 4 | 3 | 4 | 4 | 4 | 3 | 27 | Defer |
| OSCAR 23.01 | 1 | 1 | 2 | 4 | 3 | 2 | 3 | 3 | 19 | Reject |
| ROOTS | 3 | 2 | 4 | 5 | 4 | 3 | 3 | 2 | 26 | Defer |
| Wikimedia Wikipedia English | 5 | 4 | 4 | 3 | 5 | 5 | 5 | 5 | 36 | Primary |
| PG-19 | 4 | 3 | 4 | 3 | 4 | 5 | 5 | 5 | 33 | Limited secondary |
| PMC OA commercial-reuse-filtered | 4 | 4 | 5 | 2 | 5 | 4 | 4 | 4 | 32 | Later secondary |

Score rationale:

- Web corpora lose training-use points because their publisher does not own
  every crawled document, not because their engineering is poor.
- The Pile loses most points because its mix includes components whose rights
  and distribution status are unsuitable for a conservative V1.
- Wikimedia wins on source and license traceability, not sheer scale.
- PG-19 loses diversity and some training-use clarity due historical style
  and Project Gutenberg's jurisdiction/individual-work conditions.
- PMC scores well technically but poorly as a general-text base because
  biomedical prose is too narrow for the first mix.

## 9. Recommended Primary Dataset

**Primary: Wikimedia Wikipedia English, pinned to the processed 2023-11-01
English snapshot or an equivalently pinned official dump version.**

This recommendation is intentionally not FineWeb. FineWeb would be the leading
quality-oriented web option if Nexa explicitly accepted Common Crawl terms,
per-document rights uncertainty, and potential modern web synthetic-content
risk. Under the current strict independence and license-clarity requirements,
Wikimedia offers the strongest identifiable provenance and a permissive,
traceable license framework while retaining vastly more text than Nexa's first
experiment needs.

The recommended primary is a subset, not the full corpus. Its selection must
later preserve source version, article ID, title, URL, license notices, and
attribution data in a manifest.

## 10. Recommended Secondary Dataset

**Optional secondary: PG-19, limited to at most 20% of the initial token
budget.**

PG-19 is a preassembled, versioned large dataset; it does not revive the
abandoned plan to manually gather hundreds of books. It adds long-form
narrative prose and has a pre-generative-AI historical source date. Before any
future download, the selection should confirm the chosen books' Project
Gutenberg status for the project's relevant jurisdiction and strip
Project-Gutenberg trademark/license boilerplate only if the applicable terms
allow that processing.

**Deferred optional secondary: PMC OA records under CC0 or CC BY only.**

This can later add small amounts of scientific exposition after a per-article
license manifest exists. Do not include CC BY-NC content, custom-license
content, or third-party figures/illustrations in the default Nexa corpus.

## 11. Recommended Initial Token Budget

**Recommend 10M tokens for the first Nexa V1 run.**

Suggested mix, subject to the later exact licensing/manifest pass:

| Source | Target share | Provisional target |
| --- | ---: | ---: |
| Wikimedia Wikipedia English | 80% | 8M tokens |
| PG-19 | 20% maximum | 2M tokens |
| PMC OA | 0% in the first run | Deferred |

Why 10M rather than 50M:

- The system has no CUDA-capable training GPU and PyTorch is CPU-only.
- The first unknown is training throughput and memory behavior at the planned
  context/model configuration, not disk capacity.
- A 10M-token corpus is sufficient to validate selection, cleaning,
  deduplication, Nexa-BPE training, split behavior, and an end-to-end
  from-scratch run without committing to an unmeasured multi-day CPU job.
- 50M is the next serious data experiment only after the 10M run records
  actual tokens/second, peak RAM, loss behavior, and checkpoint size.

Do not interpret 10M as an optimum data-to-parameter ratio for the declared
41.6M configuration. It is a bounded V1 data and systems experiment. A model
at that scale would ultimately need substantially more data and appropriate
compute; this report recommends proving the pipeline first.

## 12. Estimated Storage Requirements

For the 10M recommendation, reserve at least 1 GiB on F: even though the
estimated data workspace is only 0.11-0.17 GiB. The additional margin covers
metadata, checksums, logs, temporary shards, validation output, and a
recoverable failure. Keep all source, cleaned, deduplicated, split, tokenized,
and cache directories on F:.

For a future 50M run, reserve at least 5 GiB. The direct data footprint remains
small, but a larger reservation protects against uncompressed JSON, retry
artifacts, and a staged deduplication implementation. This is not an
authorization to download any data now.

## 13. Risks and Limitations

1. This is a technical and licensing-risk assessment, not legal advice.
   Obtain jurisdiction-specific legal review before publishing a model trained
   on any non-project-owned corpus.
2. Dataset availability and terms can change. Recheck terms immediately before
   a later download; Common Crawl's terms expressly permit updates.
3. No broad web corpus establishes that every source document is licensed for
   training or free of AI-authored text. Do not replace that uncertainty with
   a dataset-card label.
4. CC BY-SA/GFDL use creates attribution and share-alike obligations. Keep
   provenance from the start and resolve treatment of model weights/outputs
   before distribution.
5. Historical books diversify prose but encode dated language, social bias,
   and some redaction/placeholder transformations in PG-19.
6. Wikipedia has editorial bias, factual errors, vandalism risk, and a narrow
   encyclopedic register.
7. The 10M target is an estimate until Nexa's own BPE counts the final cleaned
   documents. No pretrained tokenizer may be used to declare the final Nexa
   count.
8. The system disk is nearly full. Any later data tooling that defaults to C:
   can fail even when F: has capacity.

## 14. Rejected and Deferred Candidates

| Candidate | Status | Reason |
| --- | --- | --- |
| The Pile | Rejected | Mixed rights, unsuitable components, and local-scale mismatch. |
| OSCAR 23.01 | Rejected | Content-rights disclaimer, gated/suspended access, multilingual scope, and size. |
| FineWeb | Conditional defer | Strong quality/reproducibility but Common Crawl rights/terms and synthetic-web uncertainty. |
| Dolma v1.7 | Defer | Mixed provenance and FLAN instruction component require an explicit compliant source recipe. |
| RedPajama-v2 | Defer | Common Crawl terms, 30T-token scale, and no preselected license-safe slice. |
| C4 English | Defer | Common Crawl rights/terms and known filtering/bias concerns. |
| SlimPajama-627B | Defer | Component-specific licensing and local-scale mismatch. |
| RefinedWeb public extract | Defer | Common Crawl terms despite high-quality documentation. |
| ROOTS | Defer | Complex 498-source governance, gated release, multilingual/code scope, and size. |
| FineWeb-Edu and other model-scored web derivatives | Rejected under current rule | Avoid sources whose selection depends on pretrained-model labeling or whose synthetic-data status cannot be established. |

## Sources

[^1]: Common Crawl Foundation. [Terms of Use](https://commoncrawl.org/terms-of-use). Last updated March 7, 2024; accessed September 9, 2026.

[^2]: Hugging Face FineWeb team. [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb). Accessed September 9, 2026.

[^3]: Allen Institute for AI. [Dolma repository](https://github.com/allenai/dolma). Accessed September 9, 2026.

[^4]: Allen Institute for AI. [OLMo 1.7-7B: A 24 point improvement on MMLU](https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d). Accessed September 9, 2026.

[^5]: Allen Institute for AI. [DataDecide repository](https://github.com/allenai/DataDecide). Accessed September 9, 2026.

[^6]: Together Computer. [RedPajama-Data repository](https://github.com/togethercomputer/RedPajama-Data). Accessed September 9, 2026.

[^7]: EleutherAI. [The Pile repository](https://github.com/EleutherAI/the-pile). Accessed September 9, 2026.

[^8]: Allen Institute for AI. [C4 dataset card](https://huggingface.co/datasets/allenai/c4); Colin Raffel et al. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683). Accessed September 9, 2026.

[^9]: Cerebras. [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B); MBZUAI-LLM. [SlimPajama-627B-DC dataset card](https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC). Accessed September 9, 2026.

[^10]: Guilherme Penedo et al. [The RefinedWeb Dataset for Falcon LLM: supplemental datasheet](https://proceedings.neurips.cc/paper_files/paper/2023/file/fa3ed726cc5073b9c31e3e49a807789c-Supplemental-Datasets_and_Benchmarks.pdf). NeurIPS 2023; accessed September 9, 2026.

[^11]: OSCAR Project. [OSCAR 23.01 dataset card](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301); [OSCAR quickstart](https://oscar-project.github.io/documentation/quickstart/). Accessed September 9, 2026.

[^12]: Hugo Laurençon et al. [The BigScience ROOTS Corpus](https://papers.nips.cc/paper_files/paper/2022/file/ce9e92e3de2372a4b93353eb7f3dc0bd-Paper-Datasets_and_Benchmarks.pdf). NeurIPS 2022; accessed September 9, 2026.

[^13]: Wikimedia. [Wikipedia processed dataset card](https://huggingface.co/datasets/wikimedia/wikipedia); [Wikimedia dump legal information](https://dumps.wikimedia.org/legal.html). Accessed September 9, 2026.

[^14]: Creative Commons. [CC BY-SA 4.0 deed](https://creativecommons.org/licenses/by-sa/4.0/); Wikimedia Foundation. [Wikimedia Developer App Guidelines](https://foundation.wikimedia.org/wiki/Legal:Wikimedia_Developer_App_Guidelines). Accessed September 9, 2026.

[^15]: Google DeepMind. [PG-19 repository](https://github.com/google-deepmind/pg19); Project Gutenberg. [License policy](https://www.gutenberg.org/policy/license). Accessed September 9, 2026.

[^16]: National Library of Medicine. [PMC copyright notice](https://pmc.ncbi.nlm.nih.gov/about/copyright/); [PMC Cloud Service guidance](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/); [PMC OAI-PMH API](https://pmc.ncbi.nlm.nih.gov/tools/oai/). Accessed September 9, 2026.

[^17]: Open Data Commons. [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/). Accessed September 9, 2026.
