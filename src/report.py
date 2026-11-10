import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def main():
    m = json.loads(Path("results/metrics.json").read_text())
    daily = pl.read_csv("results/daily_returns.csv")
    names = ["equal_weight", "xgboost", "lstm", "transformer"]
    labels = ["Equal weight", "XGBoost", "LSTM", "Transformer"]
    colors = ["#959da4", "#516b86", "#a17b45", "#1d615b"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "risk-portfolio-study"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#fbfbf8")
    for ax in axes:
        ax.set_facecolor("#fbfbf8")
    for name, label, color in zip(names, labels, colors):
        r = daily.filter(pl.col("strategy")==name)["net_return"].to_numpy()
        axes[0].plot(np.r_[1, np.cumprod(1+r)], color=color, label=label, linewidth=1.7)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_xlabel("Out-of-sample trading day")
    axes[0].set_ylabel("Net wealth · initial = 1")
    axes[0].set_title("01 / Net performance", loc="left", pad=14)
    sharpe = [m["strategies"][n]["annualized_sharpe"] for n in names]
    axes[1].barh(labels, sharpe, color=colors, height=.55)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(sharpe)*1.25)
    for i, value in enumerate(sharpe):
        axes[1].text(value+.02, i, f"{value:.2f}", va="center")
    axes[1].set_xlabel("Annualized daily net Sharpe · RF = 0")
    axes[1].set_title("02 / Risk-adjusted comparison", loc="left", pad=14)
    costs = list(m["cost_sensitivity"])
    axes[2].plot([int(c) for c in costs], [m["cost_sensitivity"][c]["annualized_sharpe"] for c in costs],
                 color=colors[1], marker="o", linewidth=2)
    axes[2].set_xticks([0, 5, 20])
    axes[2].set_xlabel("Cost per dollar traded · bps")
    axes[2].set_ylabel("XGBoost net Sharpe")
    axes[2].set_title("03 / Transaction-cost sensitivity", loc="left", pad=14)
    fig.suptitle("RISK-AWARE PORTFOLIO / WALK-FORWARD REVIEW", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02, .87, "Six synthetic assets · Three purged folds · 240 test days · 5 bps base cost", color="#666")
    fig.tight_layout(rect=[0, 0, 1, .82], w_pad=2)
    fig.savefig("results/overview.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    rows = ["| Strategy | Net return | Net Sharpe | Max drawdown | Mean L1 turnover | Fallbacks |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in m["strategies"].items():
        rows.append(f"| {name} | {r['net_return']:.2%} | {r['annualized_sharpe']:.3f} | {r['max_drawdown']:.2%} | {r['mean_l1_turnover']:.2%} | {r['solver_fallbacks']} |")
    body = "\n".join(["# Recorded study results", "",
        "Actual model training and CVXPY solves on a seeded synthetic market. Figures describe this dataset and execution model, not a live investment track record.", "",
        "![Study overview](overview.svg)", "", *rows, "",
        "## Findings", "",
        "The Transformer has the highest net Sharpe in this single synthetic path. XGBoost and LSTM also exceed the equal-weight baseline here. This small comparison does not establish that one architecture is generally superior; no multi-seed significance claim is made.", "",
        "Removing CVaR increases the XGBoost portfolio's realized drawdown in this run. Removing the turnover cap improves its Sharpe slightly while increasing trading. Flattening volatility estimates worsens drawdown. The constraints do not improve every metric simultaneously.", "",
        "CVaR is a forecast-scenario loss constraint; realized maximum drawdown is a different path-dependent statistic. A low modeled CVaR is not a realized loss guarantee.", "",
        "## Cost sensitivity", "", "| Total-traded cost | XGBoost net Sharpe | Net return |", "| --- | ---: | ---: |"])
    for c, r in m["cost_sensitivity"].items():
        body += f"\n| {c} bps | {r['annualized_sharpe']:.3f} | {r['net_return']:.2%} |"
    body += "\n\nThese cases rerun both optimization and execution with frozen forecasts. They include the effect of changing cost penalties on target weights.\n\n## Hypothetical static stress losses\n\n| Strategy | Equity selloff | Inflation shock | Credit liquidity |\n| --- | ---: | ---: | ---: |"
    for name in names:
        r = m["stress"][name]
        body += f"\n| {name} | {r['equity_selloff']:.2%} | {r['inflation_shock']:.2%} | {r['credit_liquidity']:.2%} |"
    body += "\n\nShocks are applied to last rebalance target weights, without subsequent trading or recovery. This is a static exposure check; exact shock vectors are in src/run.py.\n\n## Forecast errors\n\n| Model | Mean 5-day return MAE | Mean 5-day volatility MAE |\n| --- | ---: | ---: |"
    for name in names[1:]:
        errors = [r for r in m["forecast_errors"] if r["model"]==name]
        body += f"\n| {name} | {np.mean([r['return_mae'] for r in errors]):.5f} | {np.mean([r['volatility_mae'] for r in errors]):.5f} |"
    body += "\n\nMAEs use decimal five-day return units and equal averaging over equally sized folds. Better return MAE need not imply better portfolio outcomes.\n\n## Reproduction and evidence\n\n```bash\npython -m src.run\npython -m src.report\npython -m unittest discover -s tests -v\n```\n\n"
    body += f"Seed: `{m['seed']}`. Dataset SHA-256: `{m['input_sha256']}`. Model/solver versions and exact train-label boundaries are in [metrics.json](metrics.json).\n\n"
    body += "Evidence: [daily returns](daily_returns.csv), [forecasts](forecasts.csv), [weight/solver traces](weights.json), [MLflow export](tracking_export.json), [verification](verification.json), and [methodology](../METHODS.md). MLflow's local file store is regenerated under mlruns/ and excluded from Git.\n"
    Path("results/RESULTS.md").write_text(body)


if __name__ == "__main__":
    main()
