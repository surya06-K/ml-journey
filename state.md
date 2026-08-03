# Surya — ML Learning State File

Last updated: 2026-07-28

## Background

B.Tech CS at BITS Pilani Hyderabad. Solo founder of JustAutomateX (AI automation agency for Indian SMBs). Two live production clients: SREE (field service reports) and VD VSP (896-SKU WhatsApp order intake → PDF quotation). Stack: self-hosted n8n on Hetzner, Groq (llama-3.3-70b), Gemini 2.5 Flash, WhatsApp Cloud API, Supabase pgvector, Next.js.

Ships production AI daily but treated models as black boxes until starting this roadmap. Python/JS fluent. No DSA interview track — not pursuing that.

## Where he is on the 24-week roadmap

**Week 4 Day 1 done (logistic regression — classification, sigmoid, decision boundary). Phase 1 (Andrew Ng ML Spec Course 1).**

Week 3 fully complete: all videos, all Coursera labs (c1w2_lab1–lab6), all quizzes (c1w2_quiz2 + C1W1 graded quiz), multi-feature regression from scratch (6/6). Week 4 Day 1 covered but NOT yet in session-log.yaml — C1 Week 3 syllabus isn't ingested, so no item id exists (Rule 1). Notes for Week 4 Day 1 not written yet (user chose to log-and-move-on).

Math track (see math-track.md): linear algebra 3B1B ch. 1–5 done, calculus 3B1B ch. 1–5 done (including ch. 4 chain rule and ch. 5 Euler's). All math prerequisites through Phase 3 cleared. Only near-term pull is StatQuest probability/MLE when Week 4 Day 2 hits log loss.

Next up: ingest C1 Week 3 syllabus into ml-syllabus.yaml, then continue Week 4 (cost function / log loss, gradient descent, from-scratch implementation).

## Topics genuinely internalized

These he can explain cold, has been quizzed on, and answered correctly (or corrected and re-locked):

- **NumPy fundamentals:** vectorization (why it's faster — SIMD, C under the hood, shorter code), broadcasting rule, boolean masks, axis behaviour ("axis I pass is the axis that disappears"). Passed a 25-question self-test.
- **Pandas:** read_csv, filtering, groupby, missing values, .loc/.iloc. Can connect groupby to his own client data (orders per customer, reports per machine).
- **Train/test split:** why you hold back data, what happens if you don't (overfitting looks like perfection).
- **Underfitting/overfitting:** U-shaped complexity-vs-test-error curve. Knows the sweet spot is on the validation curve, not training.
- **Linear regression model:** `f(x) = wx + b`. Knows what every symbol means.
- **Cost function:** `J(w,b) = (1/2m) · Σ(ŷ - y)²`. Can explain why errors are squared (cancellation + big-error penalty) and why the 2 is there (cancels the derivative's factor of 2).
- **Gradient descent:** the update rule, minus sign logic, simultaneous update requirement, learning rate failure modes. Implemented from scratch and saw α=0.3 diverge in real time.
- **Multiple linear regression:** vector form `f(x) = w·x + b`, shapes table (X is (m,n), w is (n,), b scalar, ŷ is (m,)), inner dimensions must match for `X @ w`.
- **Vectorization (multi-feature):** three reasons np.dot beats a loop.
- **Feature scaling:** three methods (divide by max, mean normalisation, z-score). Knows *why* — unscaled features make contours elongated, gradient descent zigzags.
- **Convergence checking:** learning curve plot (recommended) and automatic ε test.
- **Learning rate selection:** log-scale search, pick largest α where J decreases consistently.

## Topics skimmed or explained to him (not yet drilled)

- **Feature engineering:** understands the frontage × depth = area example but hasn't done a hands-on drill. Couldn't answer the quiz question about it — I explained it.
- **Polynomial regression:** understands the concept (add x², x³ to fit curves, still linear in parameters, must scale). Couldn't answer the quiz question — I explained it. No code drill.
- **Decision trees / random forests:** knows "many trees average out mistakes" at a high level from Kaggle week. Not deeply tested.
- **3Blue1B calculus (ch. 1-3):** watched, wrote one sentence about derivatives. Not drilled.
- **3Blue1B linear algebra (ch. 3-5):** listed in the plan but slipping. No evidence of completion.

## Drills completed with code

1. **`cost_function_from_scratch.py`** — [`week-02-linear-regression/cost_function_from_scratch.py`](week-02-linear-regression/cost_function_from_scratch.py)
   - `predict()`, `compute_cost()`, `compute_gradient()`, `gradient_descent()` in raw NumPy
   - 5/5 tests passing
   - Also ran the break-it experiment: α=0.3 diverging, α=0.0001 crawling

2. **Week 1 self-test** — 25 questions (8 NumPy, 9 Pandas, 8 intro ML), passed on first attempt

3. **`multi_feature_regression.py`** — [`week-03-multiple-features/multi_feature_regression.py`](week-03-multiple-features/multi_feature_regression.py)
   - `predict()`, `compute_cost()`, `compute_gradient()`, `gradient_descent()` extended to matrix X and vector w
   - 6/6 tests passing
   - Key insight locked in: `X.T @ errors / m` gives one gradient per feature in one shot

## Mistakes he's been corrected on (likely to repeat)

1. **f(x) vs J(w,b) confusion.** Wrote J(w,b) when he meant f(x) in a quiz. f(x) is the model (predicts), J(w,b) is the cost function (scores wrongness). Has been corrected but mixed them up at least twice.

2. **"Cost function finds the best values."** Said J(w,b) finds the best w and b. Corrected: J *measures* wrongness, gradient descent *does* the finding. J is a scorecard, not a search algorithm.

3. **Cost going up = scaling problem.** When asked "cost increases during training, what's wrong?", answered feature scaling. Corrected: it's learning rate too large. Scaling affects convergence *speed*, not direction.

4. **Missing np.sum in gradient computation.** Returned arrays instead of scalars. Lesson he wrote down: "if I expected one number and got an array, I forgot a reduction." Could recur when writing the multi-feature version.

5. **Jupyter kernel state.** Ran cells out of order, got nonsense. Now knows Restart + Run All is step one. But it'll happen again.

6. **Forgetting to activate venv.** "No module named numpy" — trained him, but it's a habit that needs reinforcement.

7. **Sigmoid outputs 0 or 1.** (Week 4 Day 1) Thought the sigmoid itself produces the class. Corrected: it outputs a *probability* in (0,1); the 0/1 comes from a *separate* thresholding step at 0.5. Squash vs decide are two steps.

8. **Decision boundary shifts with new data.** (Week 4 Day 1) Thought the boundary "emerges/redefines" when a new point is added. Corrected: it's fixed by the trained w and b (the line w·x + b = 0); new points are classified *against* it, they don't move it.

9. **Conflates two different graphs.** (Week 4 regularization) Tied "big weights" to "local minima on the cost surface." Corrected: big weights make the *model function* f(x)-vs-x wiggly (overfitting); local minima live on the *cost surface* J-vs-(w,b) and were a separate problem already solved by log loss. Regularization smooths the model function and keeps cost convex — nothing to do with local minima. Watch for this graph-confusion recurring.

## His mental model of gradient descent (in his own words, from quizzes)

> "I'm standing on a hilly cost surface in fog. I can't see the whole landscape but I can feel the slope under my feet. Feel which way is downhill, take a small step, repeat."

> "The derivative points uphill, so subtracting it moves me downhill."

> "Steps shrink automatically — as I approach the minimum the curve flattens, so the derivative gets smaller."

> "Linear regression's cost is convex — one single bowl, no local minima — so gradient descent always finds the global minimum here."

He understands simultaneous update (compute both gradients from current w,b before assigning) and can articulate why violating it is a silent bug.

## His mental model of the cost function (in his own words)

> "For every training example, take prediction minus truth, square it, add them all up, average."

> "The cost function MEASURES wrongness. It doesn't find anything. Gradient descent does the finding."

> "If J = 0 exactly, the line passes through every training point — that's usually overfitting."

## Session planning system

He built a strict 8-rule system after I invented a 2.5-hour duration for a 10-minute segment. Rules live in `CLAUDE.md`. Key files:

- `ml-syllabus.yaml` — single source of truth for course item durations. Only `verified: true` items can be planned.
- `session-log.yaml` — actual session logs with planned vs actual minutes and difficulty ratings.
- `corrections.log` — append-only record of factual errors.

Current multipliers (default, no recalibration yet — need 5 sessions): video 1.0, reading 1.5, quiz 1.5, lab 2.0.

One session logged so far: 2026-07-25, 4 videos, planned 19m, actual 30m, difficulty "right".

## Open loops

1. **3B1B linear algebra ch. 3-5:** slipping, flagged but not addressed.
2. **3B1B calculus ch. 5 (Euler's number):** critical before Week 4 — needed for sigmoid function in logistic regression.
3. **Prospect data collection:** he needs to build a Google Sheet with 9 qualification-framework columns + converted (0/1) for the post-Week-4 mini-project (prospect scoring with logistic regression).
4. **Course 1 Week 3 syllabus:** must be ingested into ml-syllabus.yaml before any Week 4 planning (Rule 2).

All Coursera labs (c1w2_lab1–lab6), quizzes (c1w2_quiz2, C1W1 graded quiz) completed as of 2026-07-28.

## How to work with him

- Intuition first, then math. Never open with equations.
- Build over watch. Drills + code, not lectures.
- One resource at a time. Pick one, defend it.
- Anchor examples to his real work: order data → regression, lead scoring → logistic regression, document digitization → classification.
- Raw NumPy before sklearn for any new concept.
- Concise. Three sentences over ten. No fluff.
- He finishes what he starts — don't suggest switching courses mid-stream.
- No DSA/leetcode/interview advice. No business tangents unless he asks.
