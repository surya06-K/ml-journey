# Surya — ML Learning State File

Last updated: 2026-08-08

## Background

B.Tech CS at BITS Pilani Hyderabad. Solo founder of JustAutomateX (AI automation agency for Indian SMBs). Two live production clients: SREE (field service reports) and VD VSP — a dairy cooperative whose BCC staff across Andhra Pradesh file internal spare-parts requisition/usage reports via a searchable 896-item HTML form (`storemanagement.niat.tech`) → n8n webhook → PDF (PDFShift/pd.co) → Gmail → Google Sheets log. (Note: VD VSP is NOT WhatsApp or Supabase-based — that's a separate hackathon demo and a different RAG project respectively.) Broader stack across projects: self-hosted n8n on Hetzner, Groq (llama-3.3-70b), Gemini 2.5 Flash, WhatsApp Cloud API, Supabase pgvector, Next.js.

Ships production AI daily but treated models as black boxes until starting this roadmap. Python/JS fluent. No DSA interview track — not pursuing that.

## Where he is on the 24-week roadmap

**Week 4 complete. Phase 1 (Andrew Ng ML Spec Course 1) — Course 1 fully watched through regularization.**

Week 3 fully complete: all videos, all Coursera labs (c1w2_lab1–lab6), all quizzes, multi-feature regression from scratch (6/6). Week 4: full logistic regression — sigmoid, log loss, gradient descent, overfitting, regularization all covered and noted; logistic_regression_from_scratch.py implemented (6/6 passing, 100% train accuracy). C1 Week 3 syllabus still NOT ingested into ml-syllabus.yaml, so Week 4 sessions aren't formally logged (Rule 1).

Math track (see math-track.md): linear algebra 3B1B ch. 1–5 done, calculus 3B1B ch. 1–5 done (including ch. 4 chain rule and ch. 5 Euler's). All math prerequisites through Phase 3 cleared.

Phase 1 mini-project COMPLETE — lead scoring with logistic regression on UCI Bank Marketing data (41k rows). From-scratch NumPy logistic regression, full EDA, threshold tuning. Best F1=0.471 at t=0.2 (precision 0.409, recall 0.555). Regularization tested (λ=1,10), no improvement — bottleneck is class imbalance, not overfitting. Next up: Karpathy (Phase 3) or remaining Course 2/3 depending on roadmap intent.

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
- **Classification vs regression:** why linear regression fails on 0/1 (unbounded output, outlier-sensitive threshold).
- **Sigmoid:** σ(z)=1/(1+e^(-z)) squashes any z into (0,1); outputs a *probability*, not a class. Corrected the misconception that sigmoid outputs 0/1 — the 0/1 comes from a separate threshold at 0.5.
- **Logistic regression model + decision boundary:** f(x)=σ(w·x+b); boundary is w·x+b=0, fixed by trained w,b (corrected the idea it shifts with new data).
- **Log loss (cross-entropy):** why squared error breaks (non-convex → local minima), why log loss punishes confident-wrong with near-infinite cost, combined formula, convex.
- **Gradient descent for logistic regression:** gradient equations identical in FORM to linear regression; only f(x) changes (sigmoid). Drilled from scratch.
- **Overfitting:** low train cost / high new-data cost = failure to generalize; three fixes (more data, fewer features, regularization).
- **Regularization:** λ/2m·Σw² penalty on weights (not b); λ too small→overfit, too big→underfit. Regularized gradient adds (λ/m)·w shrink term. Corrected the two-graphs confusion (model-function wiggle vs cost-surface minima).

## Topics skimmed or explained to him (not yet drilled)

- **Feature engineering:** understands the frontage × depth = area example but hasn't done a standalone hands-on drill (multi-feature code touched the mechanics). Originally couldn't answer the quiz — I explained it.
- **Polynomial regression:** understands the concept (add x², x³ to fit curves, still linear in parameters, must scale). No dedicated code drill. Originally couldn't answer the quiz — I explained it.
- **Decision trees / random forests:** knows "many trees average out mistakes" at a high level from Kaggle week. Not deeply tested.
- **StatQuest probability / MLE:** not yet watched. The "why log loss is the right cost" (maximum likelihood) is the one stats gap; pull it when the mini-project or a later week needs it.

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

4. **`logistic_regression_from_scratch.py`** — [`week-04-logistic-regression/logistic_regression_from_scratch.py`](week-04-logistic-regression/logistic_regression_from_scratch.py)
   - `sigmoid()`, `predict()`, `compute_cost()` (log loss with np.clip), `compute_gradient()`, `gradient_descent()` in raw NumPy
   - 6/6 tests passing, 100% train accuracy on separable data
   - Proved the gradient is identical in form to linear regression — predict/gradient/GD were near copy-paste from Week 3; only sigmoid + log-loss line are new
   - Caught reverting to Python built-in `sum` instead of `np.sum` (habit fix)

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

1. **Course 1 Week 3 syllabus:** not yet ingested into ml-syllabus.yaml. Needed before any Week 4 session can be formally logged (Rule 1/2). Week 4 content is done but its sessions aren't in session-log.yaml for this reason.
2. **StatQuest probability/MLE:** watch when the mini-project or a later week needs it. Only remaining near-term math pull.

Math track (linear algebra + calculus 3B1B ch. 1–5) fully done. All Coursera labs (c1w2_lab1–lab6) and quizzes completed as of 2026-07-28.

## Phase 1 mini-project — DECISION (2026-08-04, revised)

**Theme: spare-parts demand forecasting for VD VSP.** (Revised after pulling the real VD VSP project details from the client-project chat — the first framing was built on wrong assumptions; corrected below.)

**CORRECTED understanding of VD VSP (the earlier notes were wrong):**
- **Not WhatsApp, not Supabase.** Production VD VSP is a searchable 896-item HTML form (`storemanagement.niat.tech`) → n8n webhook → HTML report → PDF (PDFShift primary, pd.co fallback) → Gmail → **Google Sheets** log (append-only rows). No relational DB, no FK schema. The WhatsApp Cloud API build was a separate hackathon demo (NIAT TakeOver'26, 40-item pricelist, Meta test number) — NOT shipped to this client. Supabase pgvector belongs to a different project (the RAG chatbot).
- **Not retail orders.** VD VSP is a **dairy cooperative**. The system is for **BCC (Branch Control Center) staff across Andhra Pradesh** to file internal **spare-parts requisition / usage reports** — not external customers placing orders. So customer / delivery / payment / discount / tax fields mostly don't exist. Closest analogs: BCC location ≈ "customer," a requisition ≈ "order," each catalog line = one line item with quantity + description/remarks.
- Known bug fixed previously: duplicate "NA" item codes caused ID collisions; resolved by array-index-based item identification.

**Why the theme changed from order-value regression:** line total = qty × catalog unit price = deterministic lookup. Predicting it is trivial, nothing to learn. Dead target. The interesting regression/forecasting target is **quantity** — e.g. monthly usage per SKU, or per BCC location. Business value: inventory planning for the cooperative instead of reacting.

- **Candidate targets:** monthly quantity per SKU, or quantity per requisition line.
- **Candidate features:** SKU / part category, BCC location, month (seasonality), lagged past usage.
- **HONEST VIABILITY CAVEAT:** depends entirely on the actual data — total volume and how much history exists. An internal requisition log may be too low-volume / low-signal to model well. Cannot judge until the sheet is seen. **Fallback dataset: SREE field service reports** if VD VSP proves too thin. (KaagazAI extraction runs are a third option.)
- **Prospect scoring:** deferred, not dropped — possible later classification case study.
- **Privacy:** client data stays local. `.gitignore` blocks `*.csv`; only the notebook + findings get committed.

**Blocking data needed before scoping (from the Google Sheet, not a DB):**
1. Column headers of the log sheet(s).
2. Structure: one row per report (with a JSON/blob of items) vs one row per line item (flattened) vs separate header + line-item sheets.
3. Total row count and the date range (min/max timestamp).
4. A ~20-row sample to inventory types and spot data-quality issues.

**First step for him:** open the VD VSP Google Sheet(s); share them or paste headers + ~20 sample rows + total row count + timestamp min/max. Then decide if VD VSP is viable or switch to SREE.

**Planned pipeline once data lands:** clean → EDA → feature engineering (encode categoricals, lag features for usage) → from-scratch model (raw NumPy per his rule) → sklearn check → business-framed writeup.

## How to work with him

- Intuition first, then math. Never open with equations.
- Build over watch. Drills + code, not lectures.
- One resource at a time. Pick one, defend it.
- Anchor examples to his real work: order data → regression, lead scoring → logistic regression, document digitization → classification.
- Raw NumPy before sklearn for any new concept.
- Concise. Three sentences over ten. No fluff.
- He finishes what he starts — don't suggest switching courses mid-stream.
- No DSA/leetcode/interview advice. No business tangents unless he asks.
