# Mini-project — Lead scoring with logistic regression

**Phase 1 capstone.** Predict which prospects convert, using logistic regression
implemented from scratch in NumPy.

**Status:** ✅ Complete

---

## The problem

I run an AI-automation agency. Every prospect conversation costs time I don't
have, and I currently qualify leads on gut feel. The question worth answering:
**given what I know about a lead before I invest time in them, how likely are
they to convert?**

That's binary classification — exactly what logistic regression is for.

My own funnel doesn't have enough labelled history to train on yet. So I'm
building the pipeline on a public dataset with the same shape, and I'll point
it at my own leads once they accumulate. The code is the deliverable; the data
is swappable.

---

## The data

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) —
direct marketing campaigns (phone calls) of a Portuguese bank, May 2008–Nov 2010.

- **41,188 rows**, 20 input features + target
- **11.27% conversion rate** (4,640 yes / 36,548 no) — realistically imbalanced
- No missing values, but `unknown` is used as a category in several columns
- Target `y`: did the client subscribe to a term deposit

Not committed to the repo (5.8 MB). Reproduce with:

```bash
mkdir -p data && cd data && \
curl -L -o bank.zip "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip" && \
unzip -o bank.zip && unzip -o bank-additional.zip
```

---

## Approach

| Stage | File | What happens |
|---|---|---|
| 1. Explore | `01-explore.ipynb` | Distributions, conversion rate by feature, find the traps |
| 2. Prepare | `02-prepare.py` | Encode categoricals, scale, train/test split |
| 3. Train | `03-train.py` | My from-scratch logistic regression on real data |
| 4. Evaluate | `04-evaluate.py` | Precision/recall, threshold choice |
| 5. Write up | this README | Results and honest limitations |

Model code is imported from
[`week-04-logistic-regression/logistic_regression_from_scratch.py`](../week-04-logistic-regression/logistic_regression_from_scratch.py) —
raw NumPy, no scikit-learn until the final validation check.

---

## Key decisions

- **Dropping `duration`:** removed before training. Call length is only known *after* the call ends — using it is data leakage. The model must predict with pre-call features only.
- **Handling `pdays = 999`:** 999 is a sentinel meaning "never previously contacted," not a real day count. Split into a binary `was_contacted_before` flag plus `pdays_actual` (real days for those who were contacted; 0 otherwise). Without this, the model treats "never contacted" as "contacted 999 days ago" — a fake signal that dominates the dot product.
- **Metric:** F1 score, not accuracy. At 11.27% conversion rate, a model that predicts "no" for every prospect scores 88.7% accuracy while catching zero converters. F1 balances precision (don't waste calls) against recall (don't miss converters).

---

## Results

Logistic regression from scratch (raw NumPy, no scikit-learn), trained on 32,950 rows, evaluated on 8,238 holdout rows.

**At default threshold (0.5):**

| Metric | Value |
|---|---|
| Precision | 0.679 |
| Recall | 0.176 |
| F1 | 0.280 |
| Train accuracy | 0.897 |

The model is precise when it says "yes" (68% correct), but misses 82% of actual converters. At 0.5, it's too cautious — it learned that "no" is almost always right.

**Threshold tuning:**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.1 | 0.290 | 0.670 | 0.405 |
| 0.2 | 0.409 | 0.555 | 0.471 |
| 0.3 | 0.494 | 0.422 | 0.455 |
| 0.4 | 0.556 | 0.302 | 0.391 |
| 0.5 | 0.679 | 0.176 | 0.280 |

**Best F1 at threshold 0.2** (0.471): catches 56% of converters, with 41% of flagged prospects actually converting. For lead scoring — where missing a real converter costs more than a wasted call — this is the operating point I'd ship.

---

## Limitations

- **Regularization tested, no improvement.** Added L2 regularization (λ=1, 10) to the from-scratch implementation — results were unchanged. The bottleneck is class imbalance, not overfitting: 54 features with 33k rows leaves enough data per parameter. Techniques targeting imbalance directly (oversampling, class weights, SMOTE) would likely help more, but are beyond Phase 1 scope.
- **Feature leakage risk in economic indicators.** `euribor3m`, `emp.var.rate`, `nr.employed`, and `cons.price.idx` reflect macroeconomic conditions at the time of the campaign. A model deployed *today* would need current values of these — they're available but shift over time, so the model's learned relationships may not hold in a different economic cycle.
- **No cross-validation.** Results are from a single 80/20 split. A different shuffle could give different numbers. K-fold cross-validation would give a more robust estimate.
- **Dataset is from 2008–2010.** Consumer behavior, communication preferences, and banking products have changed. The patterns learned here may not transfer to 2026 prospects.
- **"Unknown" categories kept as-is.** 21% of `default` values are "unknown." These were one-hot encoded as their own category rather than imputed or dropped. If "unknown" correlates with the target for non-random reasons (e.g., data entry patterns), this could be a subtle leak.
- **Not my data.** This project proves the pipeline works on a public dataset with the same shape as my lead-scoring problem. The real test is pointing it at JustAutomateX's own conversion data once that has enough volume.
