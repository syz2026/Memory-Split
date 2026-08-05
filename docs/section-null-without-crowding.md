# The hypothesis is null in the regime we tested, and that regime never crowded

Memory Split predicts that arbitrary facts consume parameters a small model would
otherwise spend on reasoning. Taking the facts out of the weights should return
those parameters. We tested this at 162M parameters across a sixteen-fold range
of fact load. The mechanism works and the hypothesis is null.

The prediction rests on three premises.

1. Arbitrary facts occupy parameters in a small model.
2. Masking those facts frees the parameters they occupied.
3. The freed parameters are taken up by reasoning.

Premise 1 is false at every load we ran. Premise 2 fails independently in a
second cohort. Please note that neither failure depends on comparing the two
arms against each other. That matters, because the arm contrast carries a
training confound we set out in Section 4.

## 1. Total stored content falls as the fact load rises

We measure storage on the dense arm alone. Closed-book generative recall is
converted to bits against each attribute's pool entropy, so accuracy at or below
the guess rate contributes nothing. The ceiling is 52.96 bits per entity and is
fixed by the corpus definition.

| Entities | Exposures per fact | Bits per entity | Total bits stored | Share of what was shown |
|---|---:|---:|---:|---:|
| 50,000 | 196 | 33.14 | 1,657,145 | 62.6 percent |
| 200,000 | 49 | 0.246 | 49,273 | 0.47 percent |
| 800,000 | 12 | 0.202 | 161,252 | 0.38 percent |

The third column is the usual way to read this table. The fourth column settles
the question. A crowded model holds as much as it can, and it does not hold less
when shown more. Going from 50k to 200k entities quadruples the fact load and
cuts total stored content by a factor of thirty-four. At 800k the model holds ten
times fewer bits than at 50k while being shown sixteen times as many facts.
Compression under pressure cannot produce that shape. Abandonment can.

What predicts storage is the exposure count. Retention collapses by a factor of
133 between 196 and 49 exposures per fact. The entity count over the same step
changes by a factor of four. This is a threshold and not a gradient.

## 2. The model is never close to full

Allen-Zhu and Li put achievable capacity at 2 bits per parameter. At 162,220,800
parameters that is 324M bits.

| Entities | Demand as share of capacity | Measured storage as share of capacity |
|---|---:|---:|
| 50,000 | 0.82 percent | 0.51 percent |
| 200,000 | 3.26 percent | 0.02 percent |
| 800,000 | 13.06 percent | 0.05 percent |

The fullest the model ever gets is half a percent of its capacity. Even at our
heaviest load we asked for 13 percent of the ceiling and the model kept
0.05 percent.

Two bits per parameter is the 1000-exposure result, and achievable capacity at
100 exposures is closer to half that. Our operating points sit at 12 to 196
exposures, so the true ceiling is below 324M bits. Even against a ceiling four
times lower the peak occupancy is 2 percent. No reading of the capacity curve
puts this sweep near saturation.

## 3. The facts are absent rather than inaccessible

A four-way multiple-choice recognition probe scores the true value against three
pool-matched decoys by choice log-likelihood. Chance is 0.25. The probe runs on
800 items over 2,000 trained entities.

| Run | Generative recall | Recognition |
|---|---:|---:|
| dense 50k | 0.636 | 0.903 |
| dense 200k | 0.007 | 0.261 |
| dense 800k | 0.010 | 0.223 |
| split, all loads | 0.000 | 0.235 to 0.241 |

At 50k the probe reads 0.90 where generative recall reads 0.64, so it recovers
content the model cannot produce on demand. The same probe sits at chance at both
higher loads. A model that had stored the facts and lost the ability to verbalise
them would show the 50k pattern at 200k and 800k. It does not. The facts were
never written down.

## 4. Reasoning does not degrade under load

This is the direct test of crowding and it needs only the dense arm. Held-out
deduction accuracy on the dense model rises with fact load.

| Entities | Dense deduction accuracy |
|---|---:|
| 50,000 | 65.0 |
| 200,000 | 66.7 |
| 800,000 | 68.2 |

Multiplying the fact load by sixteen improves dense reasoning by 3.2 points.
Crowding predicts the opposite. A second cohort agrees from a different
direction, where dense loss on the reasoning extension is close to flat from 8M
to 160M parameters. Across a twenty-fold range of capacity, capacity is not the
binding constraint.

### The arm contrast, and why we do not rest the result on it

| Entities | Dense | Split | Split minus dense | 95 percent CI | McNemar p |
|---|---:|---:|---:|---|---:|
| 50,000 | 65.0 | 63.0 | negative 2.0 | [negative 4.8, positive 0.8] | 0.175 |
| 200,000 | 66.7 | 69.8 | positive 3.1 | [positive 0.4, positive 5.7] | 0.027 |
| 800,000 | 68.2 | 62.7 | negative 5.5 | [negative 8.4, negative 2.5] | 0.0003 |

There is no consistent advantage and the sign reverses twice across the dose. The
hypothesis predicts a difference that grows with load. The difference in
differences between the lightest and heaviest load is negative 3.5 points.

We report this as consistent with the null and not as a treatment effect. Three
confounds stand in the way of the stronger reading. First, the training objective
used mean reduction over non-ignored targets, so masking 24.89 percent of a
document's targets multiplied the weight on every surviving target by 1.331. The
two arms did not optimise the same objective. Second, the arms in this sweep read
different renderings of the corpus rather than one byte-identical stream. Third,
one seed was evaluated per cell, and the intervals above are bootstrap over
evaluation items rather than over training runs. Seed variation alone would
produce movement of the size we observe.

Each of these pushes on the contrast. None of them touches Sections 1 to 3, which
are single-arm measurements on the dense model.

## 5. Masking does not free what it removes

Premise 2 fails on its own terms. In a matched cohort at 40M parameters the split
arm recovers 94.2 percent of dense performance on the offloaded positions without
ever training on them. Supervision on those tokens is worth 0.33 nats out of
5.73. The offloaded content is recoverable from surrounding context, so masking
withdraws supervision rather than content.

Fact exposure in that corpus is 1.04 to 1.55 times per fact. A model that reads a
fact once does not memorise it, and a mask over a fact that was never memorised
frees nothing.

## 6. The conditions under which crowding could exist

Our experiments locate three necessary conditions. They fail for different
reasons and only one of them is bought with compute.

**Exposure above the storage threshold.** A fact must be seen often enough to be
written into the weights. Our sweep brackets that threshold between 49 and 196
exposures, and the transition across it is sharp. Below it the model declines to
learn, and declining to learn costs it nothing.

**Demand near achievable capacity.** Crowding requires the fact load to bind. Our
heaviest load reached 13 percent of the nominal ceiling in demand and one
twentieth of one percent in measured storage. A design that wants crowding must
put demand near one bit per parameter at the exposure count it actually runs, not
at the 1000-exposure figure.

**Facts not inferable from context.** If the masked values can be reconstructed
from their surroundings, the dense arm carries no memorisation burden and the
split arm loses only supervision. This condition is invisible in the corpus
statistics and has to be measured directly.

These conditions interact through a constraint that our design did not escape.
At a fixed token budget the entity count and the exposure count move against each
other, because exposures are the document count divided by the entity count. Our
biography budget was fixed at 736M tokens, so every increase in entities bought a
proportional decrease in exposures. The first two conditions therefore pulled in
opposite directions throughout the sweep. Raising the load toward capacity drove
exposure below the storage threshold before capacity was ever approached.

Separating them requires the token budget to scale with the entity count instead
of against it. Holding exposures at 200 and letting the corpus grow is the only
configuration in which the dense arm carries a real memorisation burden. A corpus
of 271,739 entities at 200 exposures over a 4B-token budget places demand at 1.7
bits per parameter for an 8M model and 0.09 for a 162M model. The small end of
that ladder is the only point where freed capacity would refer to something that
exists.

## 7. Scope

We are not claiming the null holds everywhere. We are claiming that this sweep
never entered the regime the hypothesis is about, and that its failure to enter
it was forced by the corpus rather than chosen. The model was shown a great many
facts and learned almost none of them. We assumed a model shown too many facts
would compress them until something gave way. It declines to learn them instead,
at no cost to anything else it does.

---

# Appendix A. The Gemma 3 1B reference

We scored a `gemma-3-1b-it-qat-4bit` reference against the 1B pair on the
held-out Reasoning-Gym set. The comparison is not like-for-like and we do not
read it as evidence about reasoning capability. This appendix records what was
run and why the headline reading fails.

## A.1 What was run

Fourteen task families from Reasoning-Gym v0.1.19, scored by teacher-forced
greedy agreement. Exact match counts an item correct when every output token is
correct. Token accuracy counts correct output tokens, so a partial answer still
contributes.

| Run | Families | Items per family | Total items |
|---|---:|---:|---:|
| dense, step 15582 | 14 | 512 | 7,168 |
| split90, step 15582 | 14 | 512 | 7,168 |
| Gemma 3 1B, zero-shot | 14 | 128 | 1,792 |
| Gemma 3 1B, two-shot | 10 | 64 | 603 |

The two-shot condition dropped `binary_matrix`, `largest_island`, `path_star`
and `spiral_matrix`, and skipped 293 items as overlength.

## A.2 Scores on the ten families that all four runs share

| Run | Exact | Token |
|---|---:|---:|
| dense | 27.7 | 88.7 |
| split90 | 25.1 | 89.0 |
| Gemma 3 1B, zero-shot | 10.5 | 50.3 |
| Gemma 3 1B, two-shot | 12.1 | 54.2 |

Across all fourteen families dense reaches 26.9 exact and split90 reaches 32.7.

## A.3 The split advantage does not survive the change of subset

Split90 leads dense by 5.8 points over fourteen families. It trails dense by 2.6
points over the ten families the two-shot condition ran. The entire aggregate
lead sits in the four dropped families. Spiral_matrix moves from 0.0 to 84.8 and
binary_matrix from 74.4 to 93.8, while course_schedule falls from 86.7 to 51.2
and futoshiki from 37.5 to 18.9. Split90 leads on six of fourteen families.

An ordering that reverses with the choice of subset is not a capability
difference. Prequential loss on the same tokens puts the two arms level at
0.0022 nats, which agrees with that reading. A few families crossing an
acquisition threshold explains the aggregate better than a difference in
reasoning.

## A.4 Why the gap over the reference is not a capability claim

Both models beat the reference on every subset and the margin is large. Please
note that the evaluation is in distribution for our models and out of
distribution for the reference.

The held-out items come from the same Reasoning-Gym generators as the training
corpus, drawn at held-out indices above 1,000,000,000. They carry the fixed
prompt template `Reasoning task={task}\nQuestion: {question}\nAnswer:`, which our
models read throughout training. Gemma meets both the generators and the
template for the first time. Teacher-forced greedy agreement rewards producing
the expected continuation in the expected form, so format familiarity alone
moves the score. The reference is also the 4-bit quantisation-aware-trained
instruct variant rather than a full-precision base model, and it was scored on a
quarter or an eighth of the items.

The held-out set is labelled `unsealed_exploratory_holdout` and was built after
the checkpoints existed.

## A.5 What the reference does support

The item generator and the scorer work. An independent oracle re-derived the
answer for every item, with zero rejections across all fourteen families. Our 1B
models produce well-formed answers in the target format after 8B training
tokens. Neither of those is a claim about reasoning relative to Gemma 3 1B, and
we do not make that claim.
