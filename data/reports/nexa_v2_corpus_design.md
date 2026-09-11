# Nexa Phase 12 — V2 Corpus Expansion and Dataset Design

## Executive Summary
This report formalizes the Phase 12 corpus expansion using the established Phase 11 `NexaByteTokenizer(8192)`. The V1 dataset and tokenizer remain 100% frozen. The V2 corpus successfully incorporated multiple sources, passed robust deduplication, and yielded 314,828,862 total available tokens with a confirmed `0.00%` UNK rate.

## V1 Baseline
- V1 Training Tokens: ~18.9M
- V1 Validation Tokens: ~2.0M
- Source: Wikimedia Only

## Candidate Data Sources & Provenance
| Source | Status | Provenance/Notes |
|--------|--------|------------------|
| Wikimedia English | Processed | Open encyclopedic |
| Simple English Wikipedia | Processed | Open encyclopedic (grammar focus) |
| PG-19 | Processed | Project Gutenberg literature |
| Standard Ebooks | Unavailable | Missing python epub parsing libraries in current env |
| Project Gutenberg | Unavailable | Raw dir empty |

## Cleaning & Quality Filtering Results
- **PG19**: 25 documents kept.
- **English Wiki**: 19,945 documents kept.
- **Simple Wiki**: 122,126 documents kept.

## Deduplication Statistics
- Total Candidates: 142,096
- Exact Duplicates Removed: 15
- Normalized Text Duplicates Removed: 0
- **Final Unique Documents**: 142,081

## V2 Tokenization Statistics
- **Tokenizer**: NexaByteTokenizer (Vocab: 8192)
- **Total Tokens**: 314,828,862
- **UNK Rate**: 0.00000%
- **Train Split (90%)**: 282,296,935 tokens (29 shards)
- **Val Split (10%)**: 32,531,927 tokens (4 shards)

*Mathematical Check*: Token conservation holds true across the deterministic 90/10 document split.

## Corpus Size Scenarios (Planning Only)
*Assumes ~500 tok/sec throughput on Ryzen 5 5600G (6 cores)*. **This is a planning estimate only and will be empirically measured in Phase 13.**

| Scenario | Approx Tokens | Storage | Estimated CPU Training Time | Benefit | Risk |
|---|---|---|---|---|---|
| Conservative | 10,000,000 | 20.0 MB | ~5.6 hours | Rapid iteration (1 day training) | Suboptimal model capacity usage |
| Recommended | 25,000,000 | 50.0 MB | ~13.9 hours | Balanced learning within overnight window | Limits exposure to literature data |
| Ambitious | 50,000,000 | 100.0 MB | ~27.8 hours | Maximized data diversity | Heavy thermal load (>24 hours) CPU training |

## Recommended V2 Corpus
**Recommended Target**: 25M Tokens (Recommended Scenario)
*Why*: Given the V2 model scale-up to ~15M parameters, a 25M token dataset balances diversity while keeping the compute boundary strictly within a single overnight CPU run (~13.8 hours estimated).

## Limitations and Reproducibility
- Storage Estimates correspond directly to `.bin` UINT16 size (2 bytes/token).
- UNK rate is perfectly 0% due to the true byte-level architecture.
