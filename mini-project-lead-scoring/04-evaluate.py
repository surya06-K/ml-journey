import numpy as np
import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join('..', 'week-04-logistic-regression'))
from logistic_regression_from_scratch import gradient_descent, predict

# Load and train (same as 03-train.py)
data = np.load('data/prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

w_init = np.zeros(X_train.shape[1])
b_init = 0.0
w, b, _ = gradient_descent(X_train, y_train, w_init, b_init,
                            alpha=0.1, num_iters=1000, lam=0)

# --- Evaluate on TEST set (not train) ---
test_probs = predict(X_test, w, b)
test_preds = (test_probs >= 0.5).astype(int)

# TODO: compute TP, FP, TN, FN
TP = np.sum((test_preds == 1) & (y_test == 1))
FP = np.sum((test_preds == 1) & (y_test == 0))
TN = np.sum((test_preds == 0) & (y_test == 0))
FN = np.sum((test_preds == 0) & (y_test == 1))

# TODO: compute precision, recall, F1
precision = (TP / (TP + FP))
Recall = (TP / (TP + FN))
f1 = 2 * (precision * Recall) / (precision + Recall)
# TODO: print them
print(f"Precision: {precision : .4f}")
print(f"Recall: {Recall : .4f}")
print(f"F1 Score: {f1 : .4f}")



print("\nThreshold tuning:")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    preds = (test_probs >= threshold).astype(int)
    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  t={threshold:.1f}  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")