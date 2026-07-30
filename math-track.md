# Math Track — Phase 2

The math I need for this roadmap, organised **just-in-time**: learn each
topic right before Ng's course (or Karpathy's) actually uses it. Prerequisite
math evaporates; math you apply the same week sticks.

Budget: ~2 hrs/week, running in parallel with the main track. The output is
**intuition**, not a certificate. I'm done with a topic when I can explain it
with a picture.

---

## The three areas, in priority order

1. **Linear algebra** — most important for ML. Vectors, matrices, dot
   products, matrix multiply, eigenvectors.
2. **Calculus** — derivatives, chain rule, gradients. This is what backprop
   is built on.
3. **Probability & statistics** — distributions, Bayes, expectation, maximum
   likelihood.

---

## Primary resources (one per area — no menu)

| Area | Resource | Role |
|---|---|---|
| Linear algebra | [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | Visual intuition. Watch first. Best math explainer that exists. |
| Linear algebra (deeper) | [Imperial College — Maths for ML: Linear Algebra (Coursera)](https://www.coursera.org/learn/linear-algebra-machine-learning) | Go deeper, free to audit. Only if 3B1B leaves gaps. |
| Calculus | [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus) | Same visual approach, for derivatives and the chain rule. |
| Prob & stats | [StatQuest — Josh Starmer](https://www.youtube.com/c/joshstarmer) | On-demand "explain this" button. Watch the one video for whatever concept confuses me. |
| Reference | [Mathematics for Machine Learning](https://mml-book.github.io/) (Deisenroth et al., free PDF) | Look things up. Never read cover to cover. |

---

## Area 1 — Linear Algebra

Source: 3Blue1Brown, *Essence of Linear Algebra* (chapters map to the playlist).

| Ch | Topic | Why it matters | When I need it |
|---|---|---|---|
| 1 | Vectors — what they are | Every feature row is a vector | ✅ done — used it in Week 3 |
| 2 | Linear combinations, span, basis | The space a model can represent | ✅ done |
| 3 | Linear transformations & matrices | A matrix *is* a function that moves space | ✅ done |
| 4 | Matrix multiplication as composition | Why `(m,n) @ (n,k)` works; stacking transforms | ✅ done |
| 5 | Determinant | How much a transform stretches/squishes area | ✅ done |
| 6 | Inverse, column space, null space, rank | When a system is solvable; dimensionality | Phase 4 (understanding models that fail) |
| 7 | Dot products & duality | The core operation in every prediction | ✅ core idea done, deepen anytime |
| 9 | Change of basis | Reading the same vector in different coordinates | Phase 3+ |
| 10 | Eigenvectors & eigenvalues | Directions a transform doesn't rotate; PCA, later | Phase 4 (dimensionality reduction) |

**Status:** ch. 1–5 done. Gap closed. ch. 6–10 stay on the just-in-time
schedule (inverse/rank for Phase 4, eigenvectors for dimensionality reduction).

**Done when:** I can look at `X @ w` and *see* a transformation acting on a
space, not just index-shuffling.

---

## Area 2 — Calculus

Source: 3Blue1Brown, *Essence of Calculus*.

| Ch | Topic | Why it matters | When I need it |
|---|---|---|---|
| 1 | Derivative as a picture | Slope = how fast cost changes | ✅ done |
| 2 | Derivative formulas geometrically | Reading derivatives off shapes | ✅ done |
| 3 | Chain rule intuition (intro) | Foundation for the real chain rule | ✅ done |
| 4 | Chain rule & product rule | THE rule backprop is built on | ✅ done |
| 5 | What's special about e | Sigmoid = 1/(1 + e^(-z)); needed for logistic regression | ✅ done |
| 6 | Implicit differentiation | Minor for ML | Skip unless it comes up |
| 7 | Limits, epsilon-delta | Formal underpinning | Low priority |
| 8 | Integrals, fundamental theorem | Probability densities later | Phase 5-ish |
| 9 | Taylor series | Approximating functions | Optional |

**Status:** ch. 1–5 done, including ch. 4 (chain rule) — the one backprop is
built on. Phase 3 calculus prerequisite is already cleared. ch. 6–9 are
optional/on-demand.

**Done when:** I can explain why the gradient points the direction it does,
and later, why backprop multiplies derivatives layer by layer.

---

## Area 3 — Probability & Statistics

Source: StatQuest — watch on demand, not linearly. Pull up the exact video the
moment a concept blocks me.

| Topic | Why it matters | When I need it |
|---|---|---|
| Mean, variance, standard deviation | Feature scaling (z-score) uses σ directly | ✅ touched in Week 3 |
| Distributions (normal especially) | Assumptions behind many models | Phase 1 tail / Phase 4 |
| Probability basics, conditional prob | Classification outputs *are* probabilities | Week 4 (logistic regression) |
| Bayes' theorem | Naive Bayes, reasoning under uncertainty | Phase 4 |
| Expectation | Expected loss, why we average | Phase 3+ |
| **Maximum likelihood estimation** | **Why log loss is the "right" cost for logistic regression** | Week 4–5 |
| Precision, recall, confusion matrix | Evaluating classifiers properly (imbalanced data) | Phase 4 — done properly there |

**Status:** basic descriptive stats touched. Everything else on demand.

**Done when:** I don't reach for StatQuest mid-drill because the stats concept
is already clear.

---

## Immediate next actions

1. **StatQuest — probability + MLE** → watch when Week 4 hits the log-loss cost function (Day 2). This explains *why* log loss is the right cost.

Linear algebra ch. 1–5 done. Calculus ch. 1–5 done (including chain rule). All
math prerequisites through Phase 3 are cleared — the rest is optional and stays
just-in-time. StatQuest probability is the only near-term pull, and only when
Day 2's cost function calls for it.

---

## The rule for this whole track

> When Ng or Karpathy references a math idea I don't have, I pause, watch the
> one video that explains it, build the picture, then come back. I never learn
> math "to be safe." I learn it because something I'm building needs it.
