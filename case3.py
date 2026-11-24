"""Black-Litterman portfolio allocator for UChicago Trading Competition 2022, Case 3.

Standalone version of case3.ipynb. Import `allocate_portfolio` for the
competition interface, or run this file to backtest against the bundled data:

    python case3.py                      # full backtest
    python case3.py --days 500           # first 500 days only
    python case3.py --data-dir path/to/case3data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_ASSETS = 9
TAU = 0.05            # Black-Litterman uncertainty scaling of the prior
DELTA = 2.4           # risk-aversion parameter (2.15 - 2.65)
SHARPE_LOOKBACK = 20  # days of realized returns used by the Sharpe filter
MAX_HISTORY = 9       # accepted weight vectors averaged into the held portfolio
P = np.eye(N_ASSETS)  # picking matrix: one absolute view per asset

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'case3data'


def load_shares(data_dir):
    shares = pd.read_csv(Path(data_dir) / 'Shares Outstanding.csv', index_col=0,
                         encoding='utf-8-sig')
    return shares.iloc[0].to_numpy(dtype=float)


def log_returns(prices):
    """(T, N) prices -> (T-1, N) log returns."""
    return np.diff(np.log(prices), axis=0)


def covariance(returns):
    """Sample covariance (N, N) of a (T, N) return array."""
    return np.cov(returns, rowvar=False)


def implied_market_returns(prices, sigma, delta, shares):
    """Equilibrium returns Pi = delta * Sigma * w_mkt (market-cap weights)."""
    mcap = prices * shares
    w_mkt = mcap / mcap.sum()
    return delta * sigma @ w_mkt


def black_litterman(sigma, pi, P, Q, tau):
    """Posterior expected returns with Omega = tau * P Sigma P^T."""
    tau_sigma_inv = np.linalg.pinv(tau * sigma)
    omega_inv = np.linalg.pinv(tau * P @ sigma @ P.T)
    A = tau_sigma_inv + P.T @ omega_inv @ P
    b = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    return np.linalg.pinv(A) @ b


def mean_variance_weights(sigma, mu, delta):
    """Unconstrained mean-variance weights w = (delta * Sigma)^-1 mu."""
    return np.linalg.pinv(delta * sigma) @ mu


def normalize_gross(w):
    """Scale so that sum(|w|) = 1; equal weights if w is all zeros."""
    gross = np.abs(w).sum()
    if not np.isfinite(gross) or gross == 0:
        return np.repeat(1 / len(w), len(w))
    return w / gross


def sharpe(returns):
    """Mean / std of a return series (not annualized); -inf if undefined."""
    sd = returns.std()
    return returns.mean() / sd if sd > 0 else -np.inf


class BLAllocator:
    def __init__(self, shares):
        self.shares = shares
        self.prices = []
        self.preds = [[], [], []]
        self.accepted = [np.repeat(1 / N_ASSETS, N_ASSETS)]

    def held(self):
        return normalize_gross(np.mean(self.accepted, axis=0))

    def allocate(self, asset_prices, *predictions):
        self.prices.append(np.asarray(asset_prices, dtype=float))
        for hist, p in zip(self.preds, predictions):
            hist.append(np.asarray(p, dtype=float))

        # need at least 2 returns to estimate a covariance
        if len(self.prices) < 3:
            return np.repeat(1 / N_ASSETS, N_ASSETS)

        prices = np.array(self.prices)
        ret_real = log_returns(prices)
        sigma = covariance(ret_real)
        pi = implied_market_returns(prices[-1], sigma, DELTA, self.shares)

        candidate = np.zeros(N_ASSETS)
        for hist in self.preds:
            Q = log_returns(np.array(hist[-2:]))[0]  # latest change in the analyst's forecast
            mu = black_litterman(sigma, pi, P, Q, TAU)
            candidate += normalize_gross(mean_variance_weights(sigma, mu, DELTA))
        candidate = normalize_gross(candidate)

        # Sharpe filter on a time series of recent realized returns
        window = ret_real[-SHARPE_LOOKBACK:]
        if sharpe(window @ candidate) > sharpe(window @ self.held()):
            if len(self.accepted) >= MAX_HISTORY:
                self.accepted = [np.repeat(1 / N_ASSETS, N_ASSETS)]
            self.accepted.append(candidate)
        return self.held()


_allocator = None


def allocate_portfolio(asset_prices, asset_price_predictions_1,
                       asset_price_predictions_2, asset_price_predictions_3):
    """Competition entry point: returns 9 weights with sum(|w|) = 1."""
    global _allocator
    if _allocator is None:
        _allocator = BLAllocator(load_shares(DEFAULT_DATA_DIR))
    return _allocator.allocate(asset_prices, asset_price_predictions_1,
                               asset_price_predictions_2, asset_price_predictions_3)


def backtest(alloc_fn, actual, pred, n_days=None):
    n = len(actual) if n_days is None else min(n_days, len(actual))
    simple_ret = actual[1:n] / actual[:n - 1] - 1
    weights = np.array([alloc_fn(actual[t], pred[0][t], pred[1][t], pred[2][t])
                        for t in range(n - 1)])
    port = (weights * simple_ret).sum(axis=1)  # weights at t earn the t -> t+1 return
    return {
        'annualized_sharpe': np.sqrt(252) * port.mean() / port.std(),
        'cumulative_return': np.prod(1 + port) - 1,
        'avg_gross_exposure': np.abs(weights).sum(axis=1).mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--days', type=int, default=None,
                        help='number of days to backtest (default: all)')
    args = parser.parse_args()

    def load(name):
        return pd.read_csv(args.data_dir / name, index_col=0).to_numpy(dtype=float)

    actual = load('Actual Testing Data.csv')
    pred = [load(f'Predicted Testing Data Analyst {i}.csv') for i in (1, 2, 3)]

    allocator = BLAllocator(load_shares(args.data_dir))
    results = pd.DataFrame({
        'Black-Litterman': backtest(allocator.allocate, actual, pred, args.days),
        '1/n': backtest(lambda prices, *_: np.repeat(1 / N_ASSETS, N_ASSETS),
                        actual, pred, args.days),
    })
    print(results.to_string(float_format='{:.4f}'.format))


if __name__ == '__main__':
    main()
