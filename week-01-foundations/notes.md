# Week 1 — Foundations

**Covered:** environment setup, NumPy, Pandas, Kaggle Intro to Machine Learning
**Status:** ✅ Complete — 25-question self-test passed

---

## Day 1 — Environment setup

Local: `uv` virtual environment + JupyterLab, with NumPy, Pandas, Matplotlib, scikit-learn.
Cloud: Google Colab with T4 GPU verified (needed for Phase 3).

**Why isolated virtual environments matter.** Keeps each project's packages separate so they don't clash. Same idea as `node_modules` per project in JS — one project can need numpy 1.x, another can need 2.x, and they don't step on each other. Also keeps my system Python clean.

---

## Day 2 — NumPy: the vectorization mindset

The core shift: stop writing `for` loops over data, start operating on whole arrays at once.

### Key operations

| Operation | Example |
|---|---|
| Create | `np.arange(10)`, `np.zeros(5)`, `np.linspace(0, 1, 5)` |
| Inspect | `.shape`, `.dtype`, `.ndim` |
| Elementwise math | `a ** 2`, `a + b`, `a * 3` |
| Dot product | `np.dot(a, b)` |
| Reduce | `.sum()`, `.mean()`, `.max()` |
| Reshape | `.reshape(4, 3)` |

### Vectorization

```python
# Loop version
result = []
for n in nums:
    result.append(n * n + 1)

# Vectorized
result = nums ** 2 + 1
```

**Why the vectorized version is faster.** The `for` loop runs in Python, which is slow per element. NumPy runs the same loop under the hood in C on the whole array at once, so there's no Python overhead per element. The loop still happens — just in a way faster language.

### Boolean masks

```python
n = np.arange(1, 31)
n[n % 3 == 0]     # keep only multiples of 3
```

**What `n % 3 == 0` returns before it's used as an index.** A boolean array the same shape as `n` — True where the element is divisible by 3, False otherwise. NumPy then uses that mask to keep only the positions marked True.

### Broadcasting

```python
m = np.arange(1, 13).reshape(4, 3)
m + np.array([100, 200, 300])   # vector added to every row
```

**The broadcasting rule.** Line up the shapes from the right. Two dimensions match if they're equal, or if one of them is 1 (that one gets stretched to fit). If one array has fewer dimensions, the missing ones are treated as 1.

### Axis behaviour — the one that trips everyone up

```python
m.mean(axis=0)   # collapses rows   -> one value per COLUMN
m.sum(axis=1)    # collapses columns -> one value per ROW
```

**My mnemonic.** `axis=0` collapses rows → one number per column. `axis=1` collapses columns → one number per row. Rule: **whatever axis I pass in is the axis that disappears from the shape.** So a `(4, 3)` matrix with `axis=0` becomes shape `(3,)` — the 4 is gone.

---

## Day 3 — Pandas: working with real tables

### Key operations

| Task | Code |
|---|---|
| Load | `pd.read_csv(url_or_path)` |
| Inspect | `.head()`, `.info()`, `.describe()`, `.shape` |
| Select columns | `df['col']`, `df[['a', 'b']]` |
| Filter rows | `df[df['Age'] > 30]` |
| Multiple conditions | `df[(df['Sex'] == 'male') & (df['Pclass'] == 3)]` |
| Missing values | `.isna().sum()`, `.fillna(...)`, `.dropna()` |
| Group and aggregate | `df.groupby('Pclass')['Survived'].mean()` |
| New column | `df['FamilySize'] = df['SibSp'] + df['Parch'] + 1` |
| Sort | `.sort_values('Fare', ascending=False)` |

### groupby — the workhorse

Split the data into groups, compute something per group, combine the results.

```python
df.groupby('Pclass')['Survived'].mean()
```

**A real question I'd answer with `groupby` on JustAutomateX data.** Group VD VSP orders by customer to get total spend and number of orders per customer — instant top-customer list. Same shape of query for SREE service reports: group by machine to see which ones generate the most complaints. Basically anything that starts with "per customer, what's the…" or "per machine, how many…".

### Missing data

**What filling missing ages with the median assumes, and when it breaks.** It assumes missing values are essentially random — that a missing age isn't systematically different from a recorded age. That breaks when missingness correlates with something else. Example: if older passengers were less likely to have their age recorded, filling with the median pulls their real ages down, and any model trained on this data will underestimate ages for that group.

### .loc vs .iloc

- `.loc` → select by **label or condition** — `df.loc[df['Age'] > 30, 'Name']`
- `.iloc` → select by **integer position** — `df.iloc[0:5, 0:3]`

---

## Day 4 — Kaggle: Intro to Machine Learning

### The workflow

```python
X = df[features]          # inputs
y = df['Survived']        # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy_score(y_test, predictions)
```

### Train / test split

**Why we hold back a test set.** Held-back data is the only honest way to see if the model actually learned patterns or just memorised. If I only test on training data, an overfit model looks perfect — and I won't know it's broken until real users hit it.

### Underfitting and overfitting

- **Underfitting** — model too simple, misses real patterns in the data
- **Overfitting** — model memorises the training data (including noise) and fails on anything new

**Shape of complexity vs test error.** It's a U-shape. Too simple and both training and test error are high — the model can't even capture the real pattern (underfitting). As I add complexity, both drop. Past the sweet spot, training error keeps going down but test error starts climbing again — the model is memorising noise instead of learning (overfitting).

The sweet spot is at the bottom of the U **on the test/validation set** — not the training set. Training error keeps dropping past the sweet spot, so if I judged by training error alone I'd pick a worse model.

How I'd find it in practice: hold out a validation set, try a few different complexities (like tree depth, number of features, or regularisation strength), plot validation error against complexity, pick the one where the curve bottoms out. Cross-validation is the more robust version — split the data multiple ways and average the errors, so a lucky or unlucky split doesn't fool me.

### Decision trees and random forests

A decision tree splits the data on feature thresholds. A random forest builds many trees and averages them.

**Why averaging many trees beats a single tree.** One tree makes its own mistakes based on which random splits it happened to pick. Different trees make different mistakes. Average them and the mistakes cancel out while the real signal survives — like a group of biased opinions averaging into something more balanced.

---

## What I got wrong / had to look up

- Jupyter kernel state — ran cells out of order and got nonsense numbers. Restart Kernel + Run All is now my first debugging step.
- Missing `np.sum` in the gradient — returned arrays instead of scalars, caused a TypeError downstream. Rule: if I expected one number and got an array, I forgot a reduction.
- `axis=0` vs `axis=1` — needed the "axis I pass is the axis that disappears" mnemonic before it stuck.
- Kept forgetting to `source` the venv before running scripts from Terminal. "No module named numpy" trained me.

---

## Where this connects to JustAutomateX

Most of it maps directly onto my agency work. Pandas `groupby` is what I'd use to summarise VD VSP orders per customer or SREE reports per machine. NumPy vectorisation is what my n8n workflows should have been doing instead of iterating row by row. Train/test split is a discipline I've never applied to my LLM pipelines — I've been evaluating them on the same examples I built them with, which is the same trap as judging a model on training data.

---

## Self-test result

Completed a 25-task self-test covering NumPy (8), Pandas (9), and intro ML concepts (8). Passed on first attempt.
