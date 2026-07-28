# Week 3 — Multiple features and vectorization

**Course:** Andrew Ng — Supervised Machine Learning: Regression and Classification (Course 1, Week 2)
**Status:** ✅ Complete — all 4 days done

---

## The through-line for this week

Week 2 had one feature (`x`) at a time. Real problems have many — a house has size, bedrooms, age, location, etc. This week generalises everything from Week 2 to `n` features at once, and shows why NumPy's dot product makes it fast.

---

## Day 1 — Multiple features + vectorization  ✅

### The model

**Long form:**

```
f(x) = w₁x₁ + w₂x₂ + … + wₙxₙ + b
```

**Vector form:**

```
f(x) = w · x + b
```

`·` is the dot product. Both forms mean the exact same thing — each feature gets its own weight, multiply, sum them all up, add `b`.

**What changes going from 1 feature to n:** `w` and `x` stop being single numbers and become vectors of length `n`. `b` stays a single number.

**⚠️ Don't confuse `f(x)` with `J(w,b)`.** `f(x)` is the *model* — the thing that predicts. `J(w,b)` is the *cost function* — the thing that scores how wrong the model is. Mixed these up on the Day 1 quiz.

### Shapes — the thing that will bite me later if I don't lock it in now

For `m` training examples and `n` features:

| Variable | Shape | Meaning |
|---|---|---|
| `X` | `(m, n)` | full dataset — `m` rows (examples), `n` columns (features) |
| `w` | `(n,)` | one weight per feature |
| `b` | scalar | one number, added to every prediction |
| predictions `ŷ` | `(m,)` | one prediction per example |

**The dot product that has to work:**

```
X @ w  →  (m, n) @ (n,)  =  (m,)
```

Rule: the *inner* dimensions have to match (the two `n`s). When they match, they collapse and the outer dimensions form the result.

Then broadcast `b`:

```python
predictions = X @ w + b       # (m,) + scalar → (m,)
```

**Debugging tell:** when I see `ValueError: shapes (...) and (...) not aligned`, first thing to do is print `.shape` on both operands.

### Why vectorize

Three reasons NumPy's `np.dot(w, x)` beats a Python `for` loop over features:

1. **Shorter code.** One expression instead of a loop with an accumulator.
2. **Parallel execution.** NumPy uses SIMD — CPU instructions that process multiple values in a single clock cycle.
3. **Runs in C, not Python.** The actual loop still happens, but in a compiled language, so no Python interpreter overhead per element.

Together this is often 10–100× faster than the equivalent loop.

### What tripped me up on Day 1

- Wrote `J(w,b)` when I meant `f(x)` in the quiz — mixed up cost function with the model. Locking in: `J` scores, `f` predicts.
- Skipped the shapes question the first time. Realised I couldn't just wave through it — every ML error from now on is a shape mismatch.

---

## Day 2 — Convergence, learning rate, feature scaling  ✅

### Checking gradient descent for convergence

Two ways to tell if gradient descent has converged:

1. **Learning curve plot** (recommended): plot cost J vs. iteration number. When the curve flattens out, you've converged. Ng recommends this over the automatic test because it's visual and hard to misread.
2. **Automatic convergence test**: pick a small ε (e.g. 10⁻³). If J decreases by less than ε in one iteration, declare convergence. Problem: choosing the right ε is tricky, so the plot is more reliable in practice.

**Key insight:** if J *ever increases* during training, something is wrong — most likely α is too large.

### Choosing the learning rate

If cost goes up (or bounces around) during gradient descent → **α is too large**. The updates overshoot the minimum.

Fix: reduce α. Ng's method — try values on a log scale: `0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0` and pick the largest α where J decreases consistently every iteration.

- Too small α → works but crawls (thousands of extra iterations).
- Too large α → diverges or oscillates.

### Feature scaling

When one feature ranges 0–1 and another ranges 0–10,000, gradient descent zigzags — the large-range feature dominates the cost landscape. Scaling all features to similar ranges makes the contours more circular, so gradient descent takes a more direct path.

Three methods:

- **Divide by max:** x / max(x) → range [0, 1]
- **Mean normalisation:** (x - μ) / (max - min)
- **Z-score normalisation:** (x - μ) / σ → roughly [-3, 3]

### What I got wrong on Day 2 quiz

- Said cost going up is a scaling problem. It's not — it's **learning rate too large**. Feature scaling helps convergence *speed*, but doesn't cause cost to increase.

---

## Day 3 — Feature engineering + polynomial regression  ✅

### Feature engineering

Creating new features from existing ones that better capture the real relationship.

**Example:** you have `frontage` (lot width) and `depth` (lot length). Instead of giving the model two features, create one: `area = frontage × depth`. Price depends on total area, not width and depth independently — so one engineered feature works better than two raw ones.

This is a judgment call, not an algorithm. You decide what to combine based on what you know about the problem.

### Polynomial regression

A straight line (`wx + b`) can only go up or down. If the data curves — prices plateau after a certain size, growth tapers off — a line will always underfit.

Fix: add polynomial terms.

```
f(x) = w₁x + w₂x² + b          # can fit parabolas
f(x) = w₁x + w₂x² + w₃x³ + b  # can fit S-curves
```

The model is still *linear regression* — it's linear in the *parameters* (w₁, w₂, w₃). The features just happen to be powers of x.

**⚠️ Feature scaling is critical here.** If x = 1000, then x² = 1,000,000 and x³ = 1,000,000,000. Without scaling, the polynomial features will completely dominate and gradient descent will struggle.

---

## Day 4 — Extend from-scratch code to n features  ✅

See [`multi_feature_regression.py`](multi_feature_regression.py) — 6/6 tests passing.

### What changed from Week 2's scalar version

| Function | Week 2 (scalar) | Week 3 (matrix) |
|---|---|---|
| `predict` | `w * x + b` | `X @ w + b` |
| `compute_cost` | identical | identical (predict handles the shape) |
| `compute_gradient` dj_dw | `np.sum(errors * x) / m` → scalar | `X.T @ errors / m` → vector (n,) |
| `gradient_descent` | `w = w - alpha * dj_dw` (scalar) | same line, but NumPy does element-wise on vectors automatically |

### The key insight

`X.T @ errors` is the multi-feature gradient in one shot. Transpose flips X from (m,n) to (n,m), then `(n,m) @ (m,) = (n,)` — one gradient per feature. No loop over features needed.

Also used `w_init.copy()` instead of `w = w_init` — because w is now an array, and assignment would make both names point to the same object. Mutating w would silently corrupt w_init.

---

## Where this connects to JustAutomateX

At JustAutomateX, everything I care about has *multiple* features per record — a prospect has 9 qualification-framework dimensions, an order has customer + product + quantity + timing. Week 2's single-feature model was a warm-up; this week is where it starts looking like real problems.

---

## Self-check

- [x] Can write the model in long form and vector form
- [x] Can state the shapes of X, w, b, and predictions
- [x] Can explain why vectorisation is faster (3 reasons)
- [x] Convergence check — plot J vs iteration, or ε test
- [x] Learning rate — too big = cost goes up, try log scale
- [x] Feature scaling — 3 methods (max, mean norm, z-score)
- [x] Feature engineering — combine raw features (frontage × depth = area)
- [x] Polynomial regression — add x², x³ to fit curves, must scale
- [x] Extend the from-scratch code to multiple features — 6/6 passing
