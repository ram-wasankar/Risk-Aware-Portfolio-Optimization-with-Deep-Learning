# Research design

## Data and information timing

The study generates 505 daily observations for six fictional assets: growth, value, international, treasury, credit, and gold. Returns share a market factor, have asset-specific noise, mild mean reversion, fixed drifts, and two predeclared volatility regimes. This is a controlled synthetic market, not historical asset-price data. The generator is fixed at seed 20261110; it is not selected to favor a model.

Polars computes trailing five-day mean returns and twenty-day sample volatility within each asset. Each model receives twelve trailing observations of those features plus asset identity. Features through day t are available before trading; portfolios are assumed executable at that day's close and earn returns starting t+1. This idealized execution assumption excludes latency and market impact.

Targets are the next five-day compounded return and next five-day realized daily volatility multiplied by sqrt(5). Targets are trained in percentage units and converted back to decimal returns. Volatility is a scale estimate, not a calibrated predictive interval. Predictions use fixed sanity bounds: means ±4%, volatility 0.3%–12% per five-day horizon.

## Purged walk-forward protocol

| Fold | First test feature day | Last test feature day | Training labels must end before |
| --- | ---: | ---: | ---: |
| 1 | 260 | 335 | 260 |
| 2 | 340 | 415 | 340 |
| 3 | 420 | 495 | 420 |

Each model is refitted once per fold using an expanding prefix. Training samples satisfy `feature_day + 5 < test_start`, purging overlapping labels at the boundary. Feature standardization is fitted on training samples only. Later folds may learn from outcomes of earlier folds once those outcomes are historical. There is no future training block, so a post-test embargo is unnecessary here.

Testing uses sixteen non-overlapping five-day holdings per fold: 48 rebalances and 240 daily returns. Daily training labels overlap; this is useful for fitting but does not create independent training observations. No shuffled cross-validation, early stopping, hyperparameter search, test-set model selection, or confidence intervals for strategy superiority are claimed.

## Forecasting

- **XGBoost:** separate return and volatility regressors, 80 trees, depth 3, learning rate 0.05, L2 regularization 5. Flattened sequence input.
- **LSTM:** one recurrent layer with 16 hidden units and a two-output linear head.
- **Temporal Transformer:** 16-dimensional projection, learned temporal position embeddings, one encoder layer, two attention heads, 32-dimensional feed-forward layer, and a two-output head. All tokens are historical; no future token is supplied.
- Neural models use Adam, learning rate 0.005, 18 full-batch epochs, mean-squared error, and CPU execution. Fixed settings prioritize a fast, inspectable comparison over exhaustive tuning.

## Constrained allocation

For each rebalance, take up to 200 already-observed daily returns and form non-overlapping five-day block scenarios. Standardize each asset column, then rescale it to the forecast volatility and recenter it at the forecast mean. This preserves historical scenario dependence approximately; it does not model new tail events or stochastic correlation changes.

With scenario losses `L_s = -r_s @ w`, CVaR uses the convex representation:

```text
CVaR_95(w) = eta + sum(max(L_s - eta, 0)) / (0.05 * scenario_count)
maximize forecast_mean @ w - 0.6 * CVaR_95(w) - cost_rate * norm1(w - previous)
```

CVXPY solves with CLARABEL under these constraints:

- Fully invested, long-only, at most 35% in one asset.
- Equity group (first three assets) ≤65%; bond group (treasury and credit) ≤65%.
- Five-day scenario CVaR ≤6%.
- Total dollars traded, `norm1(w - previous)`, ≤30% of wealth per rebalance.

These are model-scenario and target-weight constraints, not guarantees on realized losses or exposures after price drift. On infeasibility, existing holdings are retained and the trace records a fallback. That fallback is not presented as a feasible optimized portfolio.

## Backtest accounting

All strategies begin from equal weights without initial funding costs. Targets are held as fixed shares for five days; weights drift with each day's returns. The next trade is measured against those drifted weights. Costs are charged once on the first day of each block, at 5 bps per dollar of total absolute buys plus sells. The fee is deducted before that day's portfolio return. There are no taxes, borrowing, spread dynamics, financing fees, or capacity limits.

The baseline rebalances to equal weights on the same dates and pays the same cost convention. Net Sharpe is daily mean / sample standard deviation ×sqrt(252), with zero risk-free rate. Drawdown uses a wealth series beginning at 1, so an initial loss counts. Results cover 240 daily test observations; annualization does not make this a multi-year track record.

## Controls and tracking

XGBoost ablations remove both CVaR terms, remove the turnover cap while retaining cost penalty, or flatten forecast volatility across assets. Mean forecasts remain frozen. Cost sensitivity reruns the XGBoost optimizer and execution ledger at 0, 5, and 20 bps. Higher costs can change holdings, so sensitivity is not required to be monotonic.

Stress tests apply three explicitly hypothetical six-asset shock vectors to each strategy's last rebalance target weights. They are static exposure checks, not new historical backtests or stress-adjusted Sharpe estimates. Exact vectors are in `src/run.py`.

MLflow stores a parent study run and seven strategy runs, parameters, metrics, and result artifacts. The compact setup uses MLflow's explicitly enabled local file store with `mlflow-skinny`; tracking files remain local. Run IDs and metric exports are checked in. Model binaries are not stored. Reruns create new run IDs; seeds and input digest identify the deterministic data, while numeric outputs may vary slightly across dependency or solver versions.

## References

- [CVXPY portfolio optimization example](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html)
- [CVXPY CVaR API](https://www.cvxpy.org/api_reference/cvxpy.atoms.other_atoms.html#cvar)
- [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
