"""
Logistic regression from scratch — sigmoid, log loss, gradient descent.

The logistic twin of Week 3's multi_feature_regression.py. Almost everything
carries over; the differences are exactly two:

  1. predict() runs the linear part through a sigmoid → outputs a PROBABILITY
     in (0, 1), not an unbounded number.
  2. compute_cost() uses LOG LOSS (cross-entropy), not squared error.

The gradient comes out looking IDENTICAL in form to linear regression:
    dJ/dw = (1/m) * X^T @ (f(x) - y)
    dJ/db = (1/m) * sum(f(x) - y)
...the only thing that changed is what f(x) is (sigmoid now).

No scikit-learn. NumPy only. Implement the five functions, then run
`python logistic_regression_from_scratch.py` to check against the tests.

Shapes (m examples, n features):
    X : (m, n)      y : (m,)  with values in {0, 1}
    w : (n,)        b : scalar
    predictions f(x) : (m,)  — probabilities in (0, 1)
"""

import numpy as np


def sigmoid(z):
    """
    The sigmoid (logistic) function:  g(z) = 1 / (1 + e^(-z))

    Squashes any real number (or array) into the open interval (0, 1).

    Args:
        z: float or np.ndarray — the linear part, w·x + b

    Returns:
        same shape as z — values in (0, 1)

    Hint: one line. np.exp() is the exponential. Works element-wise on arrays.
    """
    z = np.clip(z, -500, 500)
    return (1/(1+(np.exp(-z))))


def predict(X, w, b):
    """
    Logistic model:  f(x) = sigmoid(X @ w + b)

    Returns a PROBABILITY per example, not a 0/1 class. (Turning it into a
    class is a separate thresholding step — not done here.)

    Args:
        X: np.ndarray, shape (m, n)
        w: np.ndarray, shape (n,)
        b: float

    Returns:
        np.ndarray, shape (m,) — probabilities in (0, 1)

    Hint: same X @ w + b as Week 3, then pass the whole thing through sigmoid().
    """
    return sigmoid(np.dot(X, w) + b)


def compute_cost(X, y, w, b, lam=0):
    """
    Log loss (binary cross-entropy):

        J(w,b) = -(1/m) * sum[ y*log(f) + (1-y)*log(1-f) ]

    where f = predict(X, w, b).

    Args:
        X: np.ndarray, shape (m, n)
        y: np.ndarray, shape (m,) — labels in {0, 1}
        w: np.ndarray, shape (n,)
        b: float

    Returns:
        float — the cost J

    Hints:
        - reuse predict()
        - NumPy does the y*log(f) + (1-y)*log(1-f) element-wise, then np.sum
        - to avoid log(0) blowing up, clip f into [eps, 1-eps] first:
              eps = 1e-15
              f = np.clip(f, eps, 1 - eps)
    """
    f = predict(X, w, b)
    m = len(y)
    eps = 1e-15
    f = np.clip(f, eps, 1 - eps)
    J = -(1/m) * np.sum(y * np.log(f) + (1 - y) * np.log(1 - f))
    J += (lam / (2 * m)) * np.sum(w ** 2)
    return J


def compute_gradient(X, y, w, b, lam=0):
    """
    Gradients of the log-loss cost. Same FORM as linear regression — the only
    difference is that f(x) is the sigmoid output.

        dJ/dw = (1/m) * X^T @ (f - y)     ← shape (n,)
        dJ/db = (1/m) * sum(f - y)         ← scalar

    Args:
        X: np.ndarray, shape (m, n)
        y: np.ndarray, shape (m,)
        w: np.ndarray, shape (n,)
        b: float

    Returns:
        (dj_dw, dj_db)
        dj_dw: np.ndarray, shape (n,)
        dj_db: float

    Hint: this is almost copy-paste from Week 3's compute_gradient. errors =
    predict(...) - y, then X.T @ errors / m. That's the whole trick.
    """
    f = predict(X, w, b)
    m = len(y)
    dw = (1/m) * X.T @ (f - y) + (lam / m) * w
    db = (1/m) * np.sum(f-y)
    return dw, db



def gradient_descent(X, y, w_init, b_init, alpha, num_iters, lam=0):
    """
    Run gradient descent. Identical loop to Week 3.

    Args:
        X: np.ndarray, shape (m, n)
        y: np.ndarray, shape (m,)
        w_init: np.ndarray, shape (n,)
        b_init: float
        alpha: float
        num_iters: int
        lam: float, L2 regularization parameter

    Returns:
        (w, b, cost_history)

    Hint: remember w_init.copy() — w is a mutable array; assigning w = w_init
    would make both names point to the same object (the bug from Week 3).
    """
    w = w_init.copy()
    b = b_init
    
    cost_his = []

    for i in range(num_iters):
        dw, db = compute_gradient(X, y, w, b, lam)

        
        w = w - alpha * dw
        b = b - alpha * db

        cost = compute_cost(X, y, w, b, lam)
        cost_his.append(cost)

    return w, b, cost_his

        

    


# ---------------------------------------------------------------------------
# Tests — don't edit these. If one fails, the bug is in your function.
# ---------------------------------------------------------------------------

def _run_tests():
    passed = 0
    total = 6

    # --- 1. sigmoid basic values ---
    try:
        assert abs(sigmoid(0.0) - 0.5) < 1e-12, "sigmoid(0) should be 0.5"
        assert sigmoid(20.0) > 0.999, "sigmoid(large +) should approach 1"
        assert sigmoid(-20.0) < 0.001, "sigmoid(large -) should approach 0"
        out = sigmoid(np.array([0.0, 0.0]))
        assert out.shape == (2,) and np.allclose(out, 0.5), "sigmoid must work element-wise"
        print("PASS  sigmoid() — 0.5 at 0, saturates at the ends, vectorized")
        passed += 1
    except NotImplementedError:
        print("TODO  sigmoid() not implemented")
    except AssertionError as e:
        print(f"FAIL  sigmoid(): {e}")

    # Build a linearly separable dataset for the rest.
    # True boundary: 1*x1 + 2*x2 - 0.5 = 0. Label = 1 when that's > 0.
    np.random.seed(0)
    m, n = 200, 2
    X = np.random.randn(m, n)
    w_true = np.array([1.0, 2.0])
    b_true = -0.5
    z = X @ w_true + b_true
    y = (z > 0).astype(float)

    # --- 2. predict returns probabilities in (0,1) with right shape ---
    try:
        p = predict(X, np.zeros(n), 0.0)
        assert p.shape == (m,), f"expected shape ({m},), got {p.shape}"
        assert np.all((p > 0) & (p < 1)), "predictions must be probabilities in (0,1)"
        assert np.allclose(p, 0.5), "with w=0,b=0 every probability should be 0.5"
        print("PASS  predict() — probabilities in (0,1), correct shape")
        passed += 1
    except NotImplementedError:
        print("TODO  predict() not implemented")
    except AssertionError as e:
        print(f"FAIL  predict(): {e}")

    # --- 3. cost at w=0,b=0 should be ln(2) ≈ 0.6931 ---
    try:
        c = compute_cost(X, y, np.zeros(n), 0.0)
        assert abs(c - np.log(2)) < 1e-6, f"expected ln(2)≈0.6931 at all-0.5 preds, got {c}"
        print("PASS  compute_cost() — ln(2) when every prediction is 0.5")
        passed += 1
    except NotImplementedError:
        print("TODO  compute_cost() not implemented")
    except AssertionError as e:
        print(f"FAIL  compute_cost(): {e}")

    # --- 4. gradient shape and direction at w=0,b=0 ---
    try:
        dw, db = compute_gradient(X, y, np.zeros(n), 0.0)
        assert dw.shape == (n,), f"dj_dw should be shape ({n},), got {dw.shape}"
        assert np.isscalar(db) or np.ndim(db) == 0, "dj_db should be a scalar"
        # finite-difference check on w[0]
        eps = 1e-6
        wp = np.zeros(n); wp[0] += eps
        wm = np.zeros(n); wm[0] -= eps
        num = (compute_cost(X, y, wp, 0.0) - compute_cost(X, y, wm, 0.0)) / (2 * eps)
        assert abs(num - dw[0]) < 1e-4, f"gradient w[0] {dw[0]} disagrees with numeric {num}"
        print("PASS  compute_gradient() — correct shape and matches finite differences")
        passed += 1
    except NotImplementedError:
        print("TODO  compute_gradient() not implemented")
    except AssertionError as e:
        print(f"FAIL  compute_gradient(): {e}")

    # --- 5. gradient descent drives cost down ---
    try:
        w_fit, b_fit, hist = gradient_descent(X, y, np.zeros(n), 0.0, alpha=0.5, num_iters=2000)
        assert len(hist) == 2000, f"expected 2000 cost entries, got {len(hist)}"
        assert hist[-1] < hist[0], "cost should decrease over training"
        assert hist[-1] < 0.2, f"should fit this separable data well, final cost {hist[-1]:.3f}"
        print(f"PASS  gradient_descent() — cost fell to {hist[-1]:.4f}")
        passed += 1
    except NotImplementedError:
        print("TODO  gradient_descent() not implemented")
    except AssertionError as e:
        print(f"FAIL  gradient_descent(): {e}")

    # --- 6. trained model classifies at ~correct accuracy ---
    try:
        w_fit, b_fit, _ = gradient_descent(X, y, np.zeros(n), 0.0, alpha=0.5, num_iters=2000)
        preds = (predict(X, w_fit, b_fit) >= 0.5).astype(float)
        acc = np.mean(preds == y)
        assert acc > 0.95, f"expected >95% training accuracy on separable data, got {acc:.2%}"
        print(f"PASS  end-to-end — {acc:.1%} training accuracy")
        passed += 1
    except NotImplementedError:
        print("TODO  gradient_descent()/predict() not implemented")
    except AssertionError as e:
        print(f"FAIL  end-to-end: {e}")

    print(f"\n{passed}/{total} passing")

    if passed == total:
        print("\nAll green. Now break it and observe:")
        print("  - print w_fit/b_fit vs w_true=[1,2], b_true=-0.5 — close in DIRECTION,")
        print("    but magnitudes drift bigger the longer you train (no regularization!)")
        print("  - add an L2 penalty (lambda/m)*w to the gradient and watch weights shrink")
        print("  - drop the np.clip in compute_cost, push alpha to 5.0, and find the nan")


if __name__ == "__main__":
    _run_tests()
