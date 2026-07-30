# Week 4 — Logistic regression and classification

**Course:** Andrew Ng — Supervised Machine Learning: Regression and Classification (Course 1, Week 3)
**Status:** 🔄 In progress — Day 1 done

---

## The through-line for this week

Weeks 2–3 predicted a *number* (house price, order value). This week predicts
a *category* — yes/no, converts/doesn't, spam/not-spam. The whole machinery
(model → cost → gradient descent → from-scratch code) repeats, but every piece
changes to handle 0/1 outputs instead of continuous ones.

Anchored to my work: **lead scoring.** A prospect either converts (1) or
doesn't (0). That's binary classification.

---

## Day 1 — Classification, sigmoid, decision boundary  ✅

### Why linear regression fails for classification

Regression finds a continuous value from the data. Classification doesn't want
a new value — it wants to pick a category (0 or 1) and stick the input in the
right bucket.

Two concrete reasons a straight line breaks here:

1. **Unbounded output.** `wx + b` can produce 7.3 or −2.1. Those aren't valid
   as a yes/no answer, and they can't be read as a probability (probabilities
   live in [0, 1]).
2. **Outliers tilt the line.** One extreme point can rotate the whole fit and
   drag the 0.5 threshold to the wrong place, misclassifying points that were
   fine before.

So we need something that (a) always outputs a value between 0 and 1, and
(b) doesn't get yanked around by extreme inputs. That's the sigmoid.

### The sigmoid function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

It's a smooth **S-curve** that squashes *any* real number into the open
interval (0, 1):

- `z` very negative → σ(z) → 0
- `z` very positive → σ(z) → 1
- `z = 0` → σ(z) = exactly 0.5

**Key point I got wrong the first time:** the sigmoid does **not** output 0 or
1. It outputs a **probability** between them. Its value is read as
*P(y = 1 | x)* — "given this input, the probability the answer is yes."

### The logistic regression model

Take the linear piece from before, `z = w·x + b`, and pass it through the
sigmoid:

$$f(x) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

So the model outputs a probability. Same `w·x + b` core as linear regression —
just wrapped in a squashing function.

### Turning a probability into a decision (two separate steps)

1. **Sigmoid squashes** → gives a probability, e.g. 0.73.
2. **Threshold decides** → if probability ≥ 0.5, predict 1; else predict 0.

These are distinct. Confusing "the sigmoid" with "the 0/1 answer" was my Day 1
mistake — the sigmoid never hands you a 0 or 1, the threshold does.

### The decision boundary

The boundary is the set of inputs where the model is exactly 50/50 — where
σ(z) = 0.5. The sigmoid equals 0.5 exactly when its input is zero, so:

$$w \cdot x + b = 0$$

That equation **is** the decision boundary. It splits the feature space:

- one side: `w·x + b > 0` → probability > 0.5 → predict **1**
- other side: `w·x + b < 0` → probability < 0.5 → predict **0**

**Two things I had wrong:**

- The boundary is **fixed by the learned `w` and `b`.** It does *not* appear or
  shift when a new point is added for classification. New points get *placed
  relative to* an already-fixed boundary; they don't move it.
- With raw features the boundary is a **straight line**. Add polynomial
  features (x², x₁x₂, …) and it can **curve** — same idea as polynomial
  regression from Week 3, now separating two classes instead of tracing a
  trend.

---

## What tripped me up on Day 1

- Thought the sigmoid itself outputs 0 or 1. It outputs a **probability**; the
  0/1 comes from a separate thresholding step.
- Thought the decision boundary emerges/redefines when new data is added. It's
  set by the trained parameters `w` and `b` and stays put; new points are just
  classified against it.

---

## Day 2 — The cost function (log loss)  ✅

### Why squared error from Week 2 doesn't work here

In Week 2 the cost `J = (1/2m)·Σ(ŷ − y)²` gave a clean **convex** bowl — one
minimum, gradient descent always finds it. Now `ŷ = σ(w·x + b)`. Put that
sigmoid inside the squared-error formula and the cost surface becomes
**non-convex** — a wavy landscape with multiple dips. Gradient descent settles
into a **local minimum** that isn't the true global one. So squared error is
out: not "wrong," but it makes the optimization unsolvable.

### What replaces it: log loss (cross-entropy)

Loss for a single example, split by the true label:

- if y = 1:  loss = −log(f(x))
- if y = 0:  loss = −log(1 − f(x))

**The behaviour that matters — punish confident-wrong hard.** Say truth = 1:

| Model predicts | Meaning | −log(pred) penalty |
|---|---|---|
| 0.99 | confident, right | ≈ 0.01 (almost none) |
| 0.50 | unsure | ≈ 0.69 (moderate) |
| 0.02 | confident, WRONG | ≈ 3.9 (huge) |

As the prediction heads toward the wrong extreme, the −log curve rockets toward
infinity, so the cost value blows up. That's exactly what we want: a
confidently-wrong prediction should be scored as very bad, because it drives
worse decisions (in lead scoring: trashing a lead that actually converts).

**Key distinction (same as f vs J):** log loss doesn't *give an answer* — it
*scores* how bad a prediction was. Confident-wrong → near-infinite **cost**,
which then drives gradient descent to correct hard.

### The combined formula

Because y is always 0 or 1, both cases fold into one line (one term always
zeroes out):

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\left[ y^{(i)}\log(f(x^{(i)})) + (1-y^{(i)})\log(1-f(x^{(i)})) \right]$$

- when y = 1 → the `(1−y)` term dies, leaving −log(f(x))
- when y = 0 → the `y` term dies, leaving −log(1−f(x))

And crucially: this cost **is convex** for logistic regression, so gradient
descent works again — one global minimum, no bad dips.

---

## Self-check

- [x] Explain why linear regression fails for 0/1 problems (unbounded output + outlier sensitivity)
- [x] Write the sigmoid and describe its S-shape and limits
- [x] State that the sigmoid outputs a probability, not a class
- [x] Write the logistic regression model f(x) = σ(w·x + b)
- [x] Define the decision boundary as w·x + b = 0 and know it's fixed by w, b
- [x] Explain why squared error breaks (non-convex → local minima)
- [x] Log loss punishes confident-wrong predictions with near-infinite cost
- [x] Write the combined log-loss formula and know why it's convex
- [ ] Day 3 — gradient descent for logistic regression
- [ ] Day 4 — from-scratch implementation
