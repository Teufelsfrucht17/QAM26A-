"""Constrained mean-variance optimisation for QAM 2026 (Assignment 1, rule 5).

Only numpy is needed. Keep this file next to your notebook and write

    from QAM26_optimization import mean_variance_long_only
    theta = mean_variance_long_only(mu, Sigma, gamma)
"""

import numpy as np


def mean_variance_long_only(mu, Sigma, gamma, tol=1e-10):
    """Long-only, fully invested mean-variance portfolio.

    Solves
        maximise    theta' mu - (gamma / 2) theta' Sigma theta
        subject to  theta_i >= 0 for every asset,  sum of theta = 1.

    Parameters
    ----------
    mu    : N expected (excess) returns
    Sigma : N x N covariance matrix (must be positive definite)
    gamma : variance aversion, a positive number

    Returns
    -------
    theta : numpy array of N weights, all >= 0, summing to one.

    Method: a primal active-set algorithm for quadratic programmes. It starts from 1/N, solves the problem on
    the assets that are currently allowed a positive weight, and moves assets in and out of that set until the
    optimality conditions hold. The function stops with an error if they do not hold at the end.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    N = len(mu)
    if Sigma.shape != (N, N):
        raise ValueError("Sigma must be an N x N matrix, with N the length of mu")
    if not (np.isfinite(mu).all() and np.isfinite(Sigma).all()):
        raise ValueError("mu and Sigma must not contain missing values")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    H = gamma * (Sigma + Sigma.T) / 2          # the quadratic term, made exactly symmetric
    theta = np.ones(N) / N                     # a feasible starting point
    free = np.ones(N, dtype=bool)              # True: the asset may have a positive weight

    for _ in range(50 * N):
        # Best portfolio that invests only in the free assets (weights sum to one, signs not restricted).
        k = int(free.sum())
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = H[np.ix_(free, free)]
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        solution = np.linalg.solve(K, np.append(mu[free], 1.0))
        target = np.zeros(N)
        target[free] = solution[:k]
        step = target - theta

        if np.abs(step).max() < tol:
            # We are at the best point for this set of free assets. Would an excluded asset help?
            gradient = H @ theta - mu
            lam = -gradient[free].mean()                   # multiplier of the constraint sum = 1
            nu = gradient + lam                            # multipliers of theta_i >= 0 (excluded assets)
            nu[free] = 0.0
            worst = int(np.argmin(nu))
            if nu[worst] >= -1e-9 * max(1.0, np.abs(gradient).max()):
                break                                      # no: optimal
            free[worst] = True                             # yes: let it in
        else:
            # Move towards the target, but stop at the first weight that would turn negative.
            falling = free & (step < -tol)
            alpha, blocking = 1.0, -1
            if falling.any():
                ratios = np.full(N, np.inf)
                ratios[falling] = -theta[falling] / step[falling]
                blocking = int(np.argmin(ratios))
                if ratios[blocking] < 1.0:
                    alpha = max(ratios[blocking], 0.0)
                else:
                    blocking = -1
            theta = theta + alpha * step
            if blocking >= 0:
                theta[blocking] = 0.0
                free[blocking] = False
    else:
        raise RuntimeError("mean_variance_long_only did not converge")

    theta[~free] = 0.0
    theta = np.clip(theta, 0.0, None)
    theta = theta / theta.sum()

    # Final check of the optimality conditions.
    gradient = H @ theta - mu
    held = theta > 1e-12
    lam = -gradient[held].mean()
    scale = max(1.0, np.abs(gradient).max())
    if np.abs(gradient[held] + lam).max() > 1e-7 * scale or (gradient[~held] + lam).min(initial=0.0) < -1e-7 * scale:
        raise RuntimeError("mean_variance_long_only: the optimality conditions do not hold")
    return theta
