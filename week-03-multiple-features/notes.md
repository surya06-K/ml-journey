# Week 3 — Multiple features and vectorization

**Course:** Andrew Ng — Supervised Machine Learning: Regression and Classification (Course 1, Week 2)
**Status:** 🔄 In progress — Day 1 done (multiple features + vectorization)

---

## The through-line for this week

Week 2 had one feature (`x`) at a time. Real problems have many — a house has size, bedrooms, age, location, etc. This week generalises everything from Week 2 to `n` features at once, and shows why NumPy's dot product makes it fast.

---

## 1. Multiple features — the model

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

---

## 2. Shapes — the thing that will bite me later if I don't lock it in now

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

---

## 3. Why vectorize

Three reasons NumPy's `np.dot(w, x)` beats a Python `for` loop over features:

1. **Shorter code.** One expression instead of a loop with an accumulator.
2. **Parallel execution.** NumPy uses SIMD — CPU instructions that process multiple values in a single clock cycle.
3. **Runs in C, not Python.** The actual loop still happens, but in a compiled language, so no Python interpreter overhead per element.

Together this is often 10–100× faster than the equivalent loop.

---

## 4. What tripped me up

- Wrote `J(w,b)` when I meant `f(x)` in the Day 1 quiz — mixed up cost function with the model. Locking in: `J` scores, `f` predicts.
- Skipped the shapes question the first time. Realised I couldn't just wave through it — every ML error from now on is a shape mismatch.

---

## 5. Where this connects

At JustAutomateX, everything I care about has *multiple* features per record — a prospect has 9 qualification-framework dimensions, an order has customer + product + quantity + timing. Week 2's single-feature model was a warm-up; this week is where it starts looking like real problems.

---

## Self-check

- [x] Can write the model in long form and vector form
- [x] Can state the shapes of X, w, b, and predictions
- [x] Can explain why vectorisation is faster (3 reasons)
- [ ] Feature scaling — Day 2
- [ ] Feature engineering + polynomial regression — Day 3
- [ ] Extend the from-scratch code to multiple features — Day 4
