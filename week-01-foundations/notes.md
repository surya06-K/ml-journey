# Week 1 — Foundations

**Covered:** environment setup, NumPy, Pandas, Kaggle Intro to Machine Learning
**Status:** ✅ Complete — 25-question self-test passed

---

## 1. Environment

Local: `uv` virtual environment + JupyterLab, with NumPy, Pandas, Matplotlib, scikit-learn.
Cloud: Google Colab with T4 GPU verified (needed for Phase 3).

> **TODO:** One line on why an isolated virtual environment matters, in your own words.

---

## 2. NumPy — the vectorization mindset

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

> **TODO:** Why is the vectorized version faster? (Hint: what language is the loop actually running in?)

### Boolean masks

```python
n = np.arange(1, 31)
n[n % 3 == 0]     # keep only multiples of 3
```

> **TODO:** Explain in one sentence what `n % 3 == 0` evaluates to *before* it's used as an index.

### Broadcasting

```python
m = np.arange(1, 13).reshape(4, 3)
m + np.array([100, 200, 300])   # vector added to every row
```

> **TODO:** Describe the rule NumPy uses to decide whether two shapes can broadcast together.

### Axis behaviour — the one that trips everyone up

```python
m.mean(axis=0)   # collapses rows   -> one value per COLUMN
m.sum(axis=1)    # collapses columns -> one value per ROW
```

> **TODO:** Write your own way of remembering which axis is which. (Whatever mnemonic actually sticks for you.)

---

## 3. Pandas — working with real tables

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

> **TODO:** Write down one question you'd answer with `groupby` on real JustAutomateX data (client orders, lead records, complaint logs). What would you group by, and what would you aggregate?

### Missing data

> **TODO:** Filling missing ages with the median is convenient — but what assumption does it make, and when would that assumption be wrong?

### .loc vs .iloc

- `.loc` → select by **label or condition** — `df.loc[df['Age'] > 30, 'Name']`
- `.iloc` → select by **integer position** — `df.iloc[0:5, 0:3]`

---

## 4. Kaggle: Intro to Machine Learning

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

> **TODO:** In your own words — why do we hold back a test set? What specifically goes wrong if we judge a model only on the data it trained on?

### Underfitting and overfitting

- **Underfitting** — model too simple, misses real patterns in the data
- **Overfitting** — model memorises the training data (including noise) and fails on anything new

> **TODO:** Describe the shape of the relationship between model complexity and test error. Where's the sweet spot, and how would you find it in practice?

### Decision trees and random forests

A decision tree splits the data on feature thresholds. A random forest builds many trees and averages them.

> **TODO:** Why does averaging many trees usually beat a single tree? Answer without using the word "better."

---

## 5. What I got wrong / had to look up

> **TODO:** Be honest here. Listing what tripped you up is more useful to future-you than a clean summary — and it shows genuine engagement to anyone reading this repo.

---

## 6. Where this connects

> **TODO:** One paragraph — which of these skills maps onto something you already do at JustAutomateX, and how?

---

## Self-test result

Completed a 25-task self-test covering NumPy (8), Pandas (9), and intro ML concepts (8).

> **TODO:** Record your score and which section was weakest.
