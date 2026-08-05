# Your reasoning benchmark may be measuring your padding

## Abstract

We spent sixteen days concluding that a 40M-parameter model could not learn
modular-arithmetic word problems. It can: it scores 93.8% where we had measured
4.7%. The gap was batched greedy decoding, which left-padded prompts with the
end-of-text token and attended over the pads. Because end-of-text is the
document separator, a padded prompt reads as a new document and the model
answers a question it invented. We report this and three other defects from the
same study, each of which produced a plausible null that survived the checks we
had. All four are cheap to test for and we found none of them discussed in the
papers we were building on.

## 1. Introduction

Three of 64 problems solved, then 60 of 64, from the same weights on the same
items. The only difference was whether we decoded them in a batch.

We found this while asking whether masking factual targets during training
frees model capacity for reasoning, which is the argument usually given for
filtering fact-heavy pages out of pretraining corpora. Four rounds of that
experiment returned nulls. Each null came from a broken precondition rather
than from the hypothesis, and each produced output that looked like evidence: a
floored benchmark, a corpus that stored nothing, a treatment effect sitting on
zero.

We report the four defects, the magnitude of each, and a cheap test that
catches it. None is specific to our hypothesis. We do not report a treatment
effect, and nothing here shows the hypothesis is false.

## 2. Setup

A 40.6M-parameter decoder (21.2M non-embedding), trained from scratch on a
synthetic corpus that interleaves four streams: templated biographies carrying
arbitrary facts, modular-arithmetic word problems with chain-of-thought traces,
Horn-clause deduction, and a bed of natural text. The benchmark is closed-book
exact match, scored against the majority-class rate rather than uniform chance.

Two corpora appear below and it matters which is which. Sections 3 and 6 use a
16.4B-token run of 31,280 steps on one L40S, carrying 1,531,800 entities at 100
exposures each, which puts fact demand at twice the achievable capacity for
that exposure count. Section 4 measures the control on a second corpus of 21.3B
tokens built at 996,408 entities and 200 exposures, where demand equals
capacity. No model has yet been trained on the second one.

## 3. Padding with the separator token

To decode a batch in one prefill, our generator padded every prompt to the
batch maximum. The padding went on the left, used the end-of-text token, and
was not masked out of attention. The code justified this on the grounds that
the pads were "ordinary EOTs the model has seen as separators."

They are separators, which is the problem. A prompt behind enough of them looks
like the start of a fresh document, so the model writes a fresh document. The
output stays fluent and on-format and it parses cleanly, which is why the
failure survived our checks: it answers a question the model invented.

On one checkpoint, 64 held-out problems, same items throughout:

| decoding | exact match |
|---|---:|
| batched, padded to the batch maximum | 3/64 |
| one prompt at a time | 60/64 |
| batched, after the fix | 60/64 |

An independent replication at a different seed and decode length found 0/64,
63/64 and 63/64, with every fixed-batch generation string-identical to its
unbatched twin. Corruption begins somewhere between 16 and 24 pad tokens.

Two anomalies we had already blamed on the model dissolve here. Accuracy had
been *rising* with the number of reasoning steps, which we read as a model
emitting a near-constant answer. Longer problems have longer prompts (100, 136,
206 and 245 mean tokens for one through four operations), and padding is the
gap to the longest prompt in the batch, so the easiest problems were damaged
most. Separately, half of all generations had fallen outside the answer space
and we had built a metric to separate formatting failures from arithmetic ones.
After the fix that rate is 1.000.

We now decode in groups of equal prompt length, which removes padding entirely.
This is not the general fix, and it is slow: a nominal batch of 32 fragments
into roughly 29 groups. An attention mask keeps both correctness and speed.

The check costs one line. Decode a prompt alone, decode it inside a
mixed-length batch, require the same string.

What the bug was hiding is a working benchmark. Scoring the finished run at
1,500 in-band items and 750 out-of-band:

| operations | trained on | exact match |
|---|---|---:|
| 1 | yes | 99.8% |
| 2 | yes | 99.2% |
| 3 | yes | 95.7% |
| 4 | yes | 88.3% |
| 5 | no | 41.5% |
| 6 | no | 7.2% |
| 7 | no | 5.3% |
| 8 | no | 4.9% |

Overall in-band accuracy is 96.0% against a 7.5% majority-class rate. Deduction,
whose constant-answer baseline is exactly 0.500, finishes at 0.950 and is a
second usable benchmark that had also read as dead. In-band the model is at
ceiling, which fails our responsiveness criterion from the opposite side to the
one we spent four rounds worrying about; the useful signal has moved to the
five-operation cell and to the length-generalisation cliff between five and six.

The metric we had built to separate formatting failures from arithmetic ones
now answers its own question. Across twenty checkpoints, the share of
generations landing in the answer space moves from 0.999 to 1.000, while
accuracy conditional on a valid answer moves from 0.196 to 0.960. The gain is
arithmetic. The model clears its baseline at step 1,564, the first checkpoint we
scored, so the sixteen days were never a question of budget.

## 4. A matched control with four criteria, two of which conflict

Our treatment masks fact values. The control masks an equal mass of non-value
targets, matched on count, span length, position in the document, and mean
per-token difficulty. The last one is what makes the comparison about *which*
tokens were masked rather than how much signal was removed.

Measured over 21.3B tokens, the treatment's masked spans average 1.9598 nats
and the control's 1.2712, a relative gap of 35.1% against a 20% tolerance we
had fixed in advance. Our first reading was that the corpus contains no
non-value tokens hard enough. That is wrong, and the way it is wrong is more
useful. Pricing each criterion separately over 800 documents:

| control | mean NLL | gap |
|---|---:|---:|
| shipped: length- and position-matched | 1.2712 | 35.1% |
| best possible, length-matched | 1.5667 | 20.0% |
| scattered tokens, mean-matched | 1.7677 | 9.8% |
| scattered tokens, the hardest available | 1.8644 | 4.9% |

Tokens that hard do exist in the documents, but not in contiguous runs.
Span-length matching is what blocks the difficulty match, and an oracle that
picks the hardest legal span every time lands exactly on the tolerance
boundary.

So the criteria are not jointly satisfiable here, and the reason generalises:
fact values are contiguous high-surprisal runs, while comparable surprisal in
ordinary text is scattered. A control can copy the shape of what the treatment
removes or the amount of learnable signal, not both. Anyone using a matched
control should say which they chose. We had not noticed we were choosing.

## 5. A data stream that quietly became another one

Our corpus mixes streams by target share. The fact stream is finite: it can
emit `entities x exposures` documents and no more. When its share asked for
more than that, the builder gave the remainder to the natural-text bed so the
token total stayed exact, which the design requires because every arm must
share a step count.

The reallocation is correct. It was also silent:

| stream | requested | realised |
|---|---:|---:|
| facts | 50% | 3.75% |
| bed | 10% | 56.25% |

The pilot that produced our first stop decision trained on a corpus that was
3.75% facts, in an experiment about fact load. Token count was exact, both mask
sidecars were byte-aligned, mask mass matched, and no control span overlapped a
value. Every integrity check passed. The manifest recorded the realised counts
in a field nobody read.

Record requested proportions next to realised ones and fail on the difference.
The requested values have to be stored separately, because the reallocation
overwrites the budget they came from.

## 6. Fitting fact text is not storing facts

Two measurements on the same run disagree. In the training phrasing,
cross-entropy on fact-value tokens is 1.9732 nats, against 25.03 on an earlier
under-exposed corpus. The model fits the text. Under a query template absent
from training, recoverable bits at the final checkpoint come out at −112.7 per
entity against a 52.96-bit ceiling: worse than the pool prior.

The probe draws a fixed entity set, so we can watch it over training rather than
trusting one reading. Across the twenty checkpoints we scored, spanning step
1,564 to step 31,280, the figure moves between −103.7 and −135.1 with no trend.
It is noisy and it never approaches zero.

The innocent reading is that the query format is unfamiliar rather than the
knowledge absent, and our own earlier work on real entities found exactly that.
The control is to run the same probe on entities the model never saw.

| cohort | recoverable bits/entity |
|---|---:|
| trained, 100 exposures each | −112.90 |
| never seen | −112.29 |
| difference | −0.61 ± 0.68 |

A model shown 1.5M facts a hundred times each behaves the same as one shown
none of them. A broken probe would depress both cohorts equally, so the
comparison is what carries the claim rather than the absolute value.

A metric that only ever returns negative numbers proves nothing by returning
one, so the probe needs a positive control. Given a stub model that places its
mass on the true value token, the same metric on the same entities returns
strongly positive bits, which shows the instrument can report presence. The
control that matters more is still outstanding: querying these same entities in
a phrasing the model did train on. If the metric goes positive there and stays
flat under held-out phrasing, the null is about accessibility. If it stays flat
in both, this section does not stand and we will withdraw it.

This run sits at twice the achievable capacity for its exposure count, which is
where our capacity analysis predicts the model stops trying to store and models
the marginal distribution of values instead. So we cannot claim this holds at
or below capacity. What we can say is that at twice capacity, the fact stream
consumed 70% of the token budget and left nothing retrievable behind.

## 7. What this does to the hypothesis

The argument for filtering fact-heavy pretraining data assumes those facts
occupy capacity that reasoning would otherwise use. In the one case we
measured, they occupied none, because the model declined to learn them rather
than compressing them. There was no store to free.

We have not measured a treatment effect and do not claim the hypothesis is
false. We do claim its first premise is measurable, that measuring it is cheap,
and that it failed here. The run that would settle whether it also fails at
capacity is one GPU-day and the corpus is built.

## 8. Limitations

One model size, one corpus, one seed. The difficulty table in Section 4 was frozen from an early checkpoint of the run
later analysed, so the gap between arms is unbiased but the absolute values are
not independent of the model. Section 6 compares two different prompts by
design, one in training phrasing and one held out, which is the comparison and
also its weakness.

Every number above is catalogued with its artifact in `RETAINED-RESULTS.md`,
sections 7 to 11. Citations are not yet in place; claims about what prior work
does and does not check have been removed pending a literature pass, and the
padding defect is a known failure mode in maskless custom decoders that we
should position against rather than claim as new.

## Appendix: checks we now run

None needs a GPU.

1. Decode a prompt alone and inside a mixed-length batch; require the same
   output.
2. Save generations, not only parsed answers. Ours kept the parse and discarded
   the text, so four rounds of invented documents left no trace in the logs.
3. Report the difficulty of what a control masks, not only the quantity, and
   price each matching criterion separately.
4. Record requested data proportions beside realised ones, stored separately
   from the budget the pipeline mutates.
5. Check a benchmark for ceiling as well as floor.
6. Query facts in a phrasing absent from training, and include entities the
   model never saw.
