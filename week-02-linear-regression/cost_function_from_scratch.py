"""
Linear regression from scratch — cost function and gradient descent.

No scikit-learn. NumPy only. The point is to build the thing Andrew Ng
derives in Course 1, Week 1 with my own hands.

Rules I'm following:
  - implement in raw NumPy first, library version only afterwards
  - no for-loops over training examples where vectorization works

Run `python cost_function_from_scratch.py` to check my implementations
against the built-in tests at the bottom.
"""

import numpy as np


def predict(x, w, b):
    """
    The linear regression model: f(x) = wx + b

    Args:
        x: np.ndarray, shape (m,) — input features
        w: float — weight (slope)
        b: float — bias (intercept)

    Returns:
        np.ndarray, shape (m,) — predictions for every example

    Hint: one line, no loop.
    """
    return w * x + b


def compute_cost(x, y, w, b):
    """
    Squared error cost:  J(w,b) = (1 / 2m) * sum((y_hat - y) ** 2)

    Args:
        x: np.ndarray, shape (m,) — input features
        y: np.ndarray, shape (m,) — true target values
        w: float — weight
        b: float — bias

    Returns:
        float — the cost J

    Hints:
        - m is just len(x)
        - reuse predict()
        - no loop; NumPy handles the whole array at once
    """
    m = len(x)
    predictions = predict(x, w, b)
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * m)


def compute_gradient(x, y, w, b):
    """
    Partial derivatives of the cost with respect to w and b.

        dJ/dw = (1/m) * sum((y_hat - y) * x)
        dJ/db = (1/m) * sum(y_hat - y)

    Args:
        x: np.ndarray, shape (m,)
        y: np.ndarray, shape (m,)
        w: float
        b: float

    Returns:
        (dj_dw, dj_db) — both floats

    Note: no 1/2 here. Squaring in the cost gave a factor of 2 on the way
    down, and it cancelled the 1/2. That's what the 2 in 2m was for.
    """
    m = len(x)
    errors = predict(x, w, b) - y
    dj_dw = np.sum(errors * x) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db


def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    """
    Run gradient descent to fit w and b.

    Repeat num_iters times:
        w = w - alpha * dJ/dw
        b = b - alpha * dJ/db

    CRITICAL: update w and b SIMULTANEOUSLY. Compute both gradients from
    the current w and b, and only then assign the new values. Using an
    already-updated w to compute b's gradient is the classic bug.

    Args:
        x, y: np.ndarray, shape (m,)
        w_init, b_init: float — starting values
        alpha: float — learning rate
        num_iters: int — how many steps to take

    Returns:
        (w, b, cost_history)
        w, b: the fitted parameters
        cost_history: list of the cost at each iteration — used to check
                      that it's actually decreasing
    """
    w, b = w_init, b_init
    cost_history = []
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost_history.append(compute_cost(x, y, w, b))
    return w, b, cost_history


# ---------------------------------------------------------------------------
# Tests — these pass once the four functions above are implemented correctly.
# Don't edit these. If one fails, the bug is in the function, not the test.
# ---------------------------------------------------------------------------

def _run_tests():
    passed = 0
    total = 5

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])   # exactly y = 2x + 1

    # 1. predict
    try:
        out = predict(x, 2.0, 1.0)
        assert np.allclose(out, y), f"expected {y}, got {out}"
        print("PASS  predict()")
        passed += 1
    except NotImplementedError:
        print("TODO  predict() not implemented")
    except AssertionError as e:
        print(f"FAIL  predict(): {e}")

    # 2. cost is zero for a perfect fit
    try:
        c = compute_cost(x, y, 2.0, 1.0)
        assert abs(c) < 1e-10, f"perfect fit should cost 0, got {c}"
        print("PASS  compute_cost() — perfect fit gives zero cost")
        passed += 1
    except NotImplementedError:
        print("TODO  compute_cost() not implemented")
    except AssertionError as e:
        print(f"FAIL  compute_cost(): {e}")

    # 3. cost is positive and correct for a bad fit
    try:
        c = compute_cost(x, y, 0.0, 0.0)
        assert abs(c - 28.5) < 1e-9, f"expected 28.5, got {c}"
        print("PASS  compute_cost() — bad fit gives the right number")
        passed += 1
    except NotImplementedError:
        pass
    except AssertionError as e:
        print(f"FAIL  compute_cost(): {e}")

    # 4. gradient is zero at the optimum
    try:
        dw, db = compute_gradient(x, y, 2.0, 1.0)
        assert abs(dw) < 1e-10 and abs(db) < 1e-10, \
            f"gradient at the minimum should be ~0, got ({dw}, {db})"
        print("PASS  compute_gradient() — zero gradient at the minimum")
        passed += 1
    except NotImplementedError:
        print("TODO  compute_gradient() not implemented")
    except AssertionError as e:
        print(f"FAIL  compute_gradient(): {e}")

    # 5. gradient descent recovers w=2, b=1
    try:
        w, b, hist = gradient_descent(x, y, 0.0, 0.0, alpha=0.01, num_iters=10000)
        assert abs(w - 2.0) < 0.01 and abs(b - 1.0) < 0.05, \
            f"expected w~2, b~1, got w={w:.4f}, b={b:.4f}"
        assert hist[-1] < hist[0], "cost should decrease over training"
        print(f"PASS  gradient_descent() — found w={w:.4f}, b={b:.4f}")
        passed += 1
    except NotImplementedError:
        print("TODO  gradient_descent() not implemented")
    except AssertionError as e:
        print(f"FAIL  gradient_descent(): {e}")

    print(f"\n{passed}/{total} passing")

    if passed == total:
        print("\nAll green. Now go break it on purpose:")
        print("  - set alpha=0.3 and watch what happens to the cost")
        print("  - set alpha=0.0001 and see how far 10000 iterations gets you")
        print("  - print cost_history every 1000 steps and watch it flatten out")


if __name__ == "__main__":
    _run_tests()
