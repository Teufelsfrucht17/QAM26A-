"""Shrinkage estimators for QAM"""

import numpy as np
import pandas as pd

def shrinkage(x, LD):
    """
    Compute first and second moments using shrinkage estimation.
    If LD=1, the function computes the covariance matrix using Ledoit and Wolf shrinkage estimator for the covariance matrix.
    If LD=0, sample covariance is used instead.

    Parameters:
    x (numpy.ndarray): T*N matrix of return realizations. (it is possible to add an assert to make sure x is in the right format)
    LD (int): If 1, Ledoit and Wolf shrinkage estimator is used. If 0, sample covariance is used.

    Returns:
    meanEstimator (numpy.ndarray): Estimated mean.
    covEstimator (numpy.ndarray): Estimated covariance matrix.
    delta (float): Shrinkage constant.
    """
    if isinstance(x, pd.DataFrame):
        x = x.values

    N = x.shape[1]
    T = x.shape[0]
    mu0 = np.mean(np.mean(x))  # Shrinkage target for mean
    mdif = np.mean(x, axis=0) - mu0

    if LD == 1:
        covEstimator, shrinkage_par = covCor(x)
    else:
        covEstimator = np.cov(x, rowvar=False)

    deltarg = ((N - 2) / T) / (mdif.T @ np.linalg.inv(covEstimator) @ mdif)
    delta = min(1, deltarg)
    meanEstimator = delta * mu0 + (1 - delta) * np.mean(x, axis=0)

    return meanEstimator, covEstimator, delta


def covCor(x, shrink=-1):
    """
    Estimate the covariance matrix.

    Parameters:
    x (numpy.ndarray): T*N matrix of return realizations.
    shrink (float): Shrinkage parameter. If not specified, it's computed.

    Returns:
    sigma (numpy.ndarray): Estimated covariance matrix.
    shrinkage (float): Shrinkage parameter.
    """
    if isinstance(x, pd.DataFrame):
        x = x.values

    # de-mean returns
    t, n = x.shape
    meanx = np.mean(x, axis=0)
    x = x - meanx

    # compute sample covariance matrix
    sample = (1 / t) * (x.T @ x)

    var = np.diag(sample)
    sqrtvar = np.sqrt(var)
    rBar = (np.sum(np.sum(sample / (sqrtvar[:, np.newaxis] * sqrtvar[:, np.newaxis].T))) - n) / (n * (n - 1))
    prior = rBar * np.outer(sqrtvar, sqrtvar)
    np.fill_diagonal(prior, var)

    if shrink == -1:  # Compute shrinkage parameters and constant
        # pi-hat
        y = x ** 2
        phiMat = y.T @ y / t - 2 * (x.T @ x) * sample / t + sample ** 2
        phi = np.sum(phiMat)

        # rho-hat
        term1 = ((x ** 3).T @ x) / t
        help = x.T @ x / t
        helpDiag = np.diag(help)
        term2 = helpDiag[:, np.newaxis] * sample
        term3 = help * var
        term4 = var * sample
        thetaMat = term1 - term2 - term3 + term4
        np.fill_diagonal(thetaMat, 0)
        rho = np.sum(np.diag(phiMat)) + rBar * np.sum(np.sum(((1. / sqrtvar)[:, np.newaxis] * sqrtvar[np.newaxis, :]) * thetaMat))

        # gamma-hat
        gamma = np.linalg.norm(sample - prior, 'fro') ** 2

        # compute shrinkage constant
        kappa = (phi - rho) / gamma
        shrinkage = max(0, min(1, kappa / t))
    else:  # use specified constant
        shrinkage = shrink

    # compute the estimator
    sigma = shrinkage * prior + (1 - shrinkage) * sample

    return sigma, shrinkage
