# Week 2 — Linear regression and the cost function

**Course:** Andrew Ng — Supervised Machine Learning: Regression and Classification (Course 1)
**Status:** 🔄 In progress — linear regression and cost function done, gradient descent next

---

## The through-line for this week

In Week 1 I *used* models — `model.fit()` was a black box. This week opens that box.

A model is just numbers → the cost function measures how wrong those numbers are → gradient descent adjusts them until the cost is minimised.

That chain is what `fit()` was doing the whole time.

---

## 1. Supervised learning — the two splits

**Supervised vs unsupervised:** do I have the correct answers in my data?

**Regression vs classification:** am I predicting a continuous number, or a category?

> **TODO:** Give one example of each from your own work — a regression problem and a classification problem you could actually solve with JustAutomateX client data.

---

## 2. The linear regression model

```
f(x) = wx + b
```

| Symbol | Meaning |
|---|---|
| `x` | input feature |
| `f(x)` or `ŷ` | the model's prediction |
| `y` | the actual, true value |
| `w` | weight (slope) |
| `b` | bias (intercept) |
| `m` | number of training examples |

`w` and `b` are the **parameters** — the only things the model learns.

> **TODO — intuition first:** Explain what "training a linear regression model" means, without using any symbols. Pretend you're explaining it to a client.

---

## 3. The cost function

```
J(w,b) = (1 / 2m) · Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²
```

Read plainly: for every training example, take prediction minus truth, square it, add them all up, average.

> **TODO:** Why square the errors instead of just summing them? Give both reasons.

> **TODO:** What's the `2` in `2m` doing? (It isn't about the maths being more correct.)

### The bowl

Plotting `J` against `w` and `b` gives a bowl shape. Every point on the floor is one possible model; the height is how wrong that model is.

> **TODO — the key sentence of this week:** Finish this in your own words: "Training a model means ___________."

> **TODO:** If `J(w,b) = 0` exactly, what does that mean geometrically — and why might it actually be bad news?

---

## 4. Gradient descent

> **TODO:** Fill this section in after Day 3. Cover:
> - the hill-in-fog intuition
> - the update rule and why there's a minus sign
> - what the learning rate α controls, and what breaks if it's too large or too small
> - why `w` and `b` must be updated *simultaneously*

---

## 5. Implementation

See [`cost_function_from_scratch.py`](cost_function_from_scratch.py) — cost function and gradient descent written in raw NumPy, no scikit-learn.

> **TODO:** Once it runs, note anything that surprised you about the numbers.

---

## 6. Connecting to the maths

Parallel track: 3Blue1Brown — Essence of Calculus (ch. 1–3), Essence of Linear Algebra (ch. 1–3).

> **TODO:** One sentence on what a derivative actually is, and why gradient descent needs one.

---

## 7. Where this connects

> **TODO:** Linear regression on real data — what could you predict from JustAutomateX order history? Write it as a business question, not a technical one.

---

## Self-check

- [ ] Can explain supervised vs unsupervised, regression vs classification
- [ ] Can explain what a cost function measures and why errors are squared
- [ ] Can explain gradient descent in plain English
- [ ] Know what the learning rate does in both failure directions
- [ ] Implemented cost + gradient descent from scratch, no libraries
- [ ] Completed the course labs and quizzes
