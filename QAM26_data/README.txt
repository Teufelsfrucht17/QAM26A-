QAM 2026 - course data pack (frozen on 10 November 2026)

Source: Kenneth R. French Data Library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Files downloaded 10 November 2026; built from the 2026-09 CRSP database.
Copyright 2026 Eugene F. Fama and Kenneth R. French. Cite the Data Library when you use these data.

Monthly observations, 1926-07 to 2026-09 (1,203 months). The first column, "month", is YYYY-MM.
There are no missing values in these files, except the momentum factor (Mom) from 1926-07 to 1926-12.

industries30_returns.csv   value-weighted returns of the 30 industry portfolios, PERCENT per month
industries30_nfirms.csv    number of firms in each industry portfolio
industries30_avgsize.csv   average firm size (market equity, USD million) in each industry portfolio
industries10_*.csv         the same three tables for the 10 industry portfolios
factors.csv                Mkt-RF, SMB, HML (Fama-French three factors), RF (one-month T-bill),
                           Mom (momentum factor, available from 1927-01); PERCENT per month

Notes
- Returns are in percent, as published. Divide by 100 before any portfolio calculation.
- Industry market capitalisation at the end of a month = number of firms x average firm size.
- Nothing else was changed: the tables are the "Average Value Weighted Returns -- Monthly",
  "Number of Firms in Portfolios" and "Average Firm Size" tables of the original files.

QAM26_shrinkage.py (beside this folder in the data pack): the shrinkage estimators used in rule 4 of Assignment 1.
  from QAM26_shrinkage import shrinkage
  mu, Sigma, delta = shrinkage(x, 1)    # x: T x N array of excess returns (decimals) in the estimation window
  James-Stein mean (shrunk toward the grand mean), Ledoit-Wolf covariance matrix (constant-correlation target).
  Keep the file next to your notebook.

QAM26_optimization.py (beside this folder in the data pack): the constrained optimisation used in rule 5 of Assignment 1.
  from QAM26_optimization import mean_variance_long_only
  theta = mean_variance_long_only(mu, Sigma, gamma)   # mu: N expected excess returns, Sigma: N x N covariance matrix
  Maximises theta'mu - (gamma/2) theta'Sigma theta subject to theta >= 0 and sum(theta) = 1. Needs only numpy.
  Keep the file next to your notebook.

Assignment 1 uses industries30_returns.csv, factors.csv and the two .py files. The other files are for later in the course.
