# Nexa Phase 9 — V1 Evaluation Summary

**Generated:** 2026-09-11T04:10:11.180842+00:00
**Checkpoint:** `checkpoints\nexa_v1_pilot\step_0009229.pt`
**Git Commit:** `ea7861756c0beecc9d300c2718566aabc69e08a6`

## Model Configuration
- **Architecture:** 2048 vocab, 256 d_model, 6 layers, 8 heads
- **Parameter Count:** 6,819,072
- **Tokenizer:** nexa_bpe_v1 (2048 vocab)

## Validation Baseline
- **Phase 8.9 Reference Loss:** 3.3422
- **Recomputed Validation Loss:** 3.3635
- **Recomputed Perplexity:** 28.89
- **Absolute Difference:** 0.021250
- **Tolerance Check (<= 0.001):** FAIL

## Reproducibility Check
- **Greedy Determinism:** PASS
- **Seeded Sampling Determinism:** PASS

## Selected Output Samples (Edge Cases)
*(Note: The model is trained on a tiny 18.9M token corpus. It is not expected to contain broad factual knowledge or deep coherence, but rather basic syntax and simple associations.)*

### Prompt Type: `normal`
> The capital of France is

**greedy** (40 tokens, stop: MAX):
    a few of the most important functions of the four four four four four four four four four four four four four

**nucleus** (40 tokens, stop: MAX):
    an expedition to the Battle of the Prince of Hirimni and the Khr of Prince of Ibakhah. The invas

**penalized** (40 tokens, stop: MAX):
    an expedition to the Battle of the Prince of Hirimni and the Khr Shah is a conquestant member of the M

### Prompt Type: `very_short`
> A

**greedy** (40 tokens, stop: MAX):
    few funds of the four four four four funds are further further than the four four four four four four f

**nucleus** (40 tokens, stop: MAX):
    text is a cellular computation of analysis an analysis a cells of a carbon is a body multiplic

**penalized** (40 tokens, stop: MAX):
    text is a cellular computation of analysis given by the pathogenic term (some stronger body must be invol

### Prompt Type: `bos_minimal`
> 

**greedy** (40 tokens, stop: MAX):
    The Bronx is a multiple of the Bronze of the Bronze of the Bronze of the Bronze of the Bronze of the Bronze

**nucleus** (40 tokens, stop: MAX):
    Autoese cyclotics (Manjamia), and multiple the Alphairit of Ibijmia Ahma

**penalized** (40 tokens, stop: MAX):
    Autoese cyclopedia Manjamiaic (AIP) was also known as Arian of Indonesia. The invas

### Prompt Type: `explicit_eos_case`
> This is a complete sentence.<eos> And then

**greedy** (40 tokens, stop: MAX):
    the source of the source of the source of the source of the source of the source of the source of the source of

**nucleus** (40 tokens, stop: MAX):
    texts as a cellular water of northern water pitch. It is a cells of the weather of a body multiplic

**penalized** (40 tokens, stop: MAX):
    texts as a cellular water of northern water. It is the mass of what is the man to make a confusion multiplic

### Prompt Type: `max_new_tokens`
> List the numbers from one to one hundred: one, two,

**greedy** (40 tokens, stop: MAX):
    the house of the house of the LORD is the house of the LORD of hosts. 10:001:00

**nucleus** (40 tokens, stop: MAX):
    graduate instruments with the wind waters of the shield are the race of the greatest holy month. The LO

**penalized** (40 tokens, stop: MAX):
    graduate instruments with a windowing vote of the multiple was related to be the strongest body multane

### Prompt Type: `near_context_limit`
> word word word word word word word word word word word word word word word word word word word word ...

**greedy** (39 tokens, stop: MAX):
    to the word of the words of the words of the words of the words of the words of the words of the words of the words of the wor

**nucleus** (39 tokens, stop: MAX):
    to Bearkind he shall becapostle in the pray'st been sin; Alcattbelistrength unto his multitu

**penalized** (39 tokens, stop: MAX):
    to Bearkinner and walso, Derist prayed unto the words of they come in his word bow: And now that

### Prompt Type: `over_context`
> word word word word word word word word word word word word word word word word word word word word ...

**greedy** (39 tokens, stop: MAX):
    to the word of the words of the words of the words of the words of the words of the words of the words of the words of the wor

**nucleus** (39 tokens, stop: MAX):
    to Bearkind he shall becapostle in the pray'st been sin; Alcattbelistrength unto his multitu

**penalized** (39 tokens, stop: MAX):
    to Bearkinner and walso, Derist prayed unto the words of they come in his word bow: And now that

### Prompt Type: `repeated_token_stress`
> A A A A A A A A A A A A A A A

**greedy** (40 tokens, stop: MAX):
    conver, or a function of the properson of a function of the function of the function of the function of the function of the

**nucleus** (40 tokens, stop: MAX):
    gem, as a card des, a bipolulian approx, and the multiple is related to be the stronger band the multiplic

**penalized** (40 tokens, stop: MAX):
    gem, as a card desw; (by another approximity and multiple the relation of an A-dimensional may be flexible

### Prompt Type: `unusual_character`
> Here are some emojis and symbols: 🍎, 🚀, €, £, ¶. Now

**greedy** (40 tokens, stop: MAX):
    the mysterious mysterium is the mysterious mysterium of the mysterium mythology. The mythology

**nucleus** (40 tokens, stop: MAX):
    the example of the Missile is one of the three prayer's pathogens of the Planna of Israel. The invas

**penalized** (40 tokens, stop: MAX):
    the example of this process was a bipolar point of an original mass in what is "come to the aspectable may be far on the
