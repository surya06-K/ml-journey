"""
Multiple-feature linear regression from scratch.

Same chain as Week 2's cost_function_from_scratch.py, but now:
  - x is a matrix X with shape (m, n)
  - w is a vector with shape (n,)
  - b is still a scalar

No scikit-learn. NumPy only. The point is to make the Week 2 code
work with real-shaped data.

Hints are in the docstrings. Run `python multi_feature_regression.py`
to check your implementations.
"""

import numpy as np


def predict(X, w, b):
    """
    Multiple-feature model: f(x) = X @ w + b

    Args:
        X: np.ndarray, shape (m, n) — m examples, n features each
        w: np.ndarray, shape (n,)   — one weight per feature
        b: float                    — bias

    Returns:
        np.ndarray, shape (m,) — one prediction per example

    Hint: one line. Think about which operation replaces w * x
    from the single-feature version.
    
    """
    a = np.dot(X, w)
    c = a+b
    return c


def compute_cost(X, y, w, b):
    """
    Squared error cost:  J(w,b) = (1 / 2m) * sum((y_hat - y) ** 2)

    Exactly the same formula as Week 2 — the only difference is that
    predict() now takes a matrix instead of a vector.

    Args:
        X: np.ndarray, shape (m, n)
        y: np.ndarray, shape (m,)
        w: np.ndarray, shape (n,)
        b: float

    Returns:
        float — the cost J
    """
    m = len(X)
    predictions = predict(X, w, b)
    errors = predictions - y
    sums = np.sum(errors**2)/(2*m)
    return sums


def compute_gradient(X, y, w, b):
    """
    Gradients of J with respect to w and b.

        dJ/dw = (1/m) * X^T @ (y_hat - y)     ← shape (n,)
        dJ/db = (1/m) * sum(y_hat - y)         ← scalar

    Args:
        X: np.ndarray, shape (m, n)
        y: np.ndarray, shape (m,)
        w: np.ndarray, shape (n,)
        b: float

    Returns:
        (dj_dw, dj_db)
        dj_dw: np.ndarray, shape (n,) — one partial derivative per feature
        dj_db: float

    Hint: the big change from Week 2 — dj_dw is now a VECTOR, not a scalar.
    Think about what X^T @ errors does: it multiplies each feature column
    by the error vector and sums across examples. That gives one gradient
    per feature in one shot.
    """
    m = len(X)
    predictions = predict(X, w, b)
    errors = predictions - y
    jw = np.dot(X.T, errors)/ m
    jb = np.sum(errors)/m
    return (jw, jb)
    


def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    """
    Run gradient descent on multiple features.

    Same loop as Week 2. The only difference: w is a vector now,
    so the update w = w - alpha * dj_dw is element-wise on a vector
    instead of on a scalar. NumPy handles that automatically.

    Args:
        X: np.ndarray, shape (m, n)
        y: np.ndarray, shape (m,)
        w_init: np.ndarray, shape (n,)
        b_init: float
        alpha: float
        num_iters: int

    Returns:
        (w, b, cost_history)
    """
    w, b = w_init.copy(), b_init
    cost_his = []

    for i in range(num_iters):
        dw, db = compute_gradient(X, y, w, b)
        w = w- alpha * dw
        b = b - alpha * db
        cost_his.append(compute_cost(X, y, w, b))
    return w, b, cost_his


# ---------------------------------------------------------------------------
# Tests — don't edit these. If one fails, the bug is in your function.
# ---------------------------------------------------------------------------

def _run_tests():
    passed = 0
    total = 6

    # Dataset: y = 2*x1 + 3*x2 + 1  (known weights and bias)
    np.random.seed(42)
    m, n = 100, 2
    X = np.random.randn(m, n)
    w_true = np.array([2.0, 3.0])
    b_true = 1.0
    y = X @ w_true + b_true

    # 1. predict — shape check
    try:
        out = predict(X, w_true, b_true)
        assert out.shape == (m,), f"expected shape ({m},), got {out.shape}"
        print("PASS  predict() — correct shape")
        passed += 1
    except NotImplementedError:
        print("TODO  predict() not implemented")
    except AssertionError as e:
        print(f"FAIL  predict(): {e}")

    # 2. predict — value check
    try:
        out = predict(X, w_true, b_true)
        assert np.allclose(out, y), "predictions don't match expected values"
        print("PASS  predict() — correct values")
        passed += 1
    except NotImplementedError:
        pass
    except AssertionError as e:
        print(f"FAIL  predict(): {e}")

    # 3. cost is zero for perfect fit
    try:
        c = compute_cost(X, y, w_true, b_true)
        assert abs(c) < 1e-10, f"perfect fit should cost 0, got {c}"
        print("PASS  compute_cost() — perfect fit gives zero")
        passed += 1
    except NotImplementedError:
        print("TODO  compute_cost() not implemented")
    except AssertionError as e:
        print(f"FAIL  compute_cost(): {e}")

    # 4. gradient is zero at the optimum
    try:
        dw, db = compute_gradient(X, y, w_true, b_true)
        assert dw.shape == (n,), f"dj_dw should be shape ({n},), got {dw.shape}"
        assert np.allclose(dw, 0, atol=1e-10), f"dj_dw at optimum should be ~0, got {dw}"
        assert abs(db) < 1e-10, f"dj_db at optimum should be ~0, got {db}"
        print("PASS  compute_gradient() — zero gradient at the minimum")
        passed += 1
    except NotImplementedError:
        print("TODO  compute_gradient() not implemented")
    except AssertionError as e:
        print(f"FAIL  compute_gradient(): {e}")

    # 5. gradient descent recovers w=[2, 3], b=1
    try:
        w_init = np.zeros(n)
        w_fit, b_fit, hist = gradient_descent(X, y, w_init, 0.0, alpha=0.1, num_iters=1000)
        assert np.allclose(w_fit, w_true, atol=0.05), \
            f"expected w~{w_true}, got w={w_fit}"
        assert abs(b_fit - b_true) < 0.05, \
            f"expected b~{b_true}, got b={b_fit:.4f}"
        print(f"PASS  gradient_descent() — found w={w_fit.round(3)}, b={b_fit:.3f}")
        passed += 1
    except NotImplementedError:
        print("TODO  gradient_descent() not implemented")
    except AssertionError as e:
        print(f"FAIL  gradient_descent(): {e}")

    # 6. cost decreases over training
    try:
        w_init = np.zeros(n)
        _, _, hist = gradient_descent(X, y, w_init, 0.0, alpha=0.1, num_iters=1000)
        assert hist[-1] < hist[0], "cost should decrease over training"
        assert len(hist) == 1000, f"expected 1000 cost entries, got {len(hist)}"
        print("PASS  gradient_descent() — cost decreases monotonically")
        passed += 1
    except NotImplementedError:
        pass
    except AssertionError as e:
        print(f"FAIL  gradient_descent(): {e}")

    print(f"\n{passed}/{total} passing")

    if passed == total:
        print("\nAll green. Now break it:")
        print("  - try alpha=10.0 and watch cost explode")
        print("  - remove the /m from compute_gradient and see what happens")
        print("  - transpose X in predict (use X.T @ w) and read the error message")


if __name__ == "__main__":
    _run_tests()
