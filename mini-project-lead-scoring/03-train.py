import numpy as np
import sys, os

# Import your from-scratch functions from Week 4
sys.path.append(os.path.join('..', 'week-04-logistic-regression'))
from logistic_regression_from_scratch import gradient_descent, predict

# 1. Load the prepared arrays
data = np.load('data/prepared.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# 2. Initialize parameters
w_init = np.zeros(X_train.shape[1])   # one weight per feature
b_init = 0.0

# 3. Train
w, b, cost_history = gradient_descent(X_train, y_train, w_init, b_init,
                                      alpha=0.1, num_iters=1000)

# 4. Check it learned — cost should drop
print(f"Initial cost: {cost_history[0]:.4f}")
print(f"Final cost:   {cost_history[-1]:.4f}")

# 5. Train accuracy
train_preds = (predict(X_train, w, b) >= 0.5).astype(int)
train_acc = np.mean(train_preds == y_train)
print(f"Train accuracy: {train_acc:.4f}")
print("Max value:", X_train.max())


