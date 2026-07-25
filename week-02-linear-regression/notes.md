# Week 2 — Linear regression and the cost function

**Course:** Andrew Ng — Supervised Machine Learning: Regression and Classification (Course 1, Week 1)
**Status:** ✅ Days 1–3 complete — linear regression, cost function, and gradient descent implemented from scratch (5/5 passing). Day 4 (labs + graded quiz) remaining.

---

## The through-line for this week

In Week 1 I *used* models — `model.fit()` was a black box. This week opened that box.

A model is just numbers → the cost function measures how wrong those numbers are → gradient descent adjusts them until the cost is minimised.

That chain is what `fit()` was doing the whole time.

---

## Day 1 — Intro to ML

### The two splits

**Supervised vs unsupervised:** do I have the correct answers in my data?

**Regression vs classification:** am I predicting a continuous number, or a category?

### Examples from my own work

**Regression** (predict a number): from VD VSP order history, predict the total order value of the next order for a given customer.

**Classification** (predict a category): from prospect data using my qualification framework, predict whether a lead is likely to convert (yes / no).

---

## Day 2 — Linear regression and the cost function

### The model

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

**In plain language.** Training a linear regression model means finding the straight line that best fits my data. The model learns two things — a slope and an intercept — that together define that line. Once the line is set, giving it a new input produces a predicted output. The whole "training" is figuring out where the line should sit so predictions on past data are as close as possible to the real answers.

### The cost function

```
J(w,b) = (1 / 2m) · Σ (ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²
```

Read plainly: for every training example, take prediction minus truth, square it, add them all up, average.

**Why square the errors.** Two reasons: (1) without squaring, a +5 and a −5 error would cancel out and falsely look like zero error, so squaring keeps every error positive. (2) squaring punishes big errors more than small ones — being off by 10 counts 4× as much as being off by 5, not 2× — which is usually what I want.

**Why the `2` in `2m`.** Pure mathematical convenience. When you take the derivative of the squared term (for gradient descent), a factor of 2 comes down that cancels the `1/2`, making the gradient formula cleaner. It has no effect on where the minimum is.

### The bowl

Plotting `J` against `w` and `b` gives a bowl shape. Every point on the floor is one possible model; the height is how wrong that model is.

**The key sentence.** *Training a model means finding the parameters (`w`, `b`) that sit at the bottom of the cost bowl — the ones with the lowest possible cost on the training data.*

The two horizontal axes are the parameters (`w` and `b`), the height is `J`. The bottom is the best model.

**If `J(w,b) = 0` exactly.** Geometrically the line passes through every training point — zero error on every example. That's usually bad news because it's the classic overfitting signal. The model has memorised the training data (including its noise) and almost certainly won't generalise to any new input.

---

## Day 3 — Gradient descent

**The intuition.** I'm standing on a hilly cost surface in fog. I can't see the whole landscape but I can feel the slope under my feet. Feel which way is downhill, take a small step, repeat. That's the algorithm.

**The update rule.**

```
w = w - α · (∂J/∂w)
b = b - α · (∂J/∂b)
```

**The minus sign.** The derivative points *uphill*, so subtracting it moves me downhill. Beautifully, this makes the direction adapt automatically — starting left of the minimum the slope is negative and `w` increases; starting right the slope is positive and `w` decreases. Same rule, both sides, always toward the minimum.

**Learning rate α.** Too small → converges but crawls. Too large → overshoots the minimum, lands higher on the far side, cost grows every step and blows up to `nan`. Saw this with my own eyes when I ran `α=0.3` — cost went 185 → 1203 → 7818 → … → 3.8 billion in 10 steps.

**Steps shrink automatically.** As I approach the minimum the curve flattens, so the derivative gets smaller, so `α · derivative` gets smaller. I don't need to decay `α` manually.

**Simultaneous update.** Compute both gradients from the *current* `w` and `b`, then assign both. Updating `w` first and then using that new `w` to compute `b`'s gradient is a silent bug — it's a different, worse algorithm and it won't crash, it'll just give worse answers.

**One reassurance.** Linear regression's cost is convex — one single bowl, no local minima — so gradient descent always finds the global minimum here. (Stops being true for neural nets in Phase 3.)

### Implementation

See [`cost_function_from_scratch.py`](cost_function_from_scratch.py) — cost function and gradient descent written in raw NumPy, no scikit-learn. 5/5 passing.

**Notes from getting it working.** First surprise was a self-inflicted bug — forgot `np.sum` in `compute_gradient`, so it returned arrays instead of scalars. Everything downstream broke with a TypeError. Lesson: if I expected one number and got an array, I forgot a reduction.

Once fixed, gradient descent recovered `w=2.0000, b=1.0000` from data generated by `y = 2x + 1` — starting from `w=0, b=0`, cost dropped from 22.16 to essentially 0 over 10,000 iterations. Watching a working optimizer for the first time on my own code felt more real than any lecture.

---

## Day 4 — Coursera labs + graded quiz  *(pending)*

Optional labs already covered by the from-scratch implementation. Quiz to close out Course 1 Week 1.

---

## Math parallel — Essence of Calculus (3B1B, ch. 1–3)

**A derivative is** the slope of a function at a specific point — it tells you which direction the function is heading and how steeply. Gradient descent needs one because the slope is literally the "which way is downhill" signal it reads off the ground to decide its next step.

---

## Where this connects to JustAutomateX

*"For a given customer, based on their past orders, what's the expected value of their next order?"* That's a linear regression problem — continuous number out — and directly useful for VD VSP planning and for prioritising outreach on high-value accounts.

---

## Self-check

- [x] Can explain supervised vs unsupervised, regression vs classification
- [x] Can explain what a cost function measures and why errors are squared
- [x] Can explain gradient descent in plain English
- [x] Know what the learning rate does in both failure directions
- [x] Implemented cost + gradient descent from scratch, no libraries
- [ ] Completed the course labs and quizzes  ← Day 4 remaining
