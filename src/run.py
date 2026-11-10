import importlib.metadata
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from .data import ASSETS, FOLDS, HORIZON, SEED, digest, market, samples, training_mask
from .models import fit_predict
from .optimize import historical_blocks, hold, optimize, performance, scenarios


def export_tracking(report):
    client = mlflow.tracking.MlflowClient()
    records = []
    for name, run_id in report["run_ids"].items():
        run = client.get_run(run_id)
        records.append({"name": name, "run_id": run_id, "status": run.info.status,
                        "params": run.data.params, "metrics": run.data.metrics,
                        "artifacts": [item.path for item in client.list_artifacts(run_id)]})
    Path("results/tracking_export.json").write_text(json.dumps(records, indent=2)+"\n")


def forecast(frame, returns):
    x, y, days, assets = samples(frame, returns)
    predictions, folds, errors = {name: {} for name in ["xgboost", "lstm", "transformer"]}, [], []
    for fold, (start, end) in enumerate(FOLDS):
        train = training_mask(days, start)
        test = (days>=start)&(days<end)&((days-start)%HORIZON==0)
        folds.append({"fold": fold, "train_rows": int(train.sum()), "max_train_feature_day": int(days[train].max()),
                      "max_train_label_day": int(days[train].max()+HORIZON),
                      "first_test_feature_day": start, "last_test_feature_day": end-HORIZON})
        for name in predictions:
            pred = fit_predict(name, x[train], y[train], x[test], SEED+fold)
            errors.append({"model": name, "fold": fold,
                           "return_mae": float(np.abs(pred[:, 0]-y[test, 0]/100).mean()),
                           "volatility_mae": float(np.abs(pred[:, 1]-y[test, 1]/100).mean())})
            for day in np.unique(days[test]):
                select = days[test]==day
                assert np.array_equal(assets[test][select], np.arange(6))
                predictions[name][int(day)] = pred[select]
    return predictions, folds, errors


def backtest(returns, forecasts, name, cost_bps=5):
    previous = np.ones(6)/6
    daily, traces = [], []
    for day in sorted(forecasts):
        prediction = forecasts[day]
        history = historical_blocks(returns, day)
        mean, vol = prediction[:, 0], prediction[:, 1].copy()
        if name == "xgboost_flat_vol":
            vol[:] = vol.mean()
        simulated = scenarios(history, mean, vol)
        if name == "equal_weight":
            weights, info = np.ones(6)/6, {"status": "baseline", "fallback": False}
        else:
            weights, info = optimize(mean, simulated, previous, cost_bps,
                cvar_limit=None if name=="xgboost_no_cvar" else .06,
                risk_weight=0 if name=="xgboost_no_cvar" else .6,
                turnover_limit=None if name=="xgboost_no_turnover" else .30)
        turnover = float(np.abs(weights-previous).sum())
        fee = turnover*cost_bps/10000
        net, previous = hold(weights, returns[day+1:day+1+HORIZON], fee)
        daily.extend(net.tolist())
        traces.append({"day": day, **info, "total_traded": turnover,
                       "fee_fraction": fee, "weights": weights.tolist()})
    metrics = {**performance(daily), "rebalances": len(traces),
               "mean_l1_turnover": float(np.mean([r["total_traded"] for r in traces])),
               "cumulative_fee_fractions": float(sum(r["fee_fraction"] for r in traces)),
               "solver_fallbacks": sum(r["fallback"] for r in traces)}
    return metrics, daily, traces


def main():
    Path("results").mkdir(exist_ok=True)
    Path("mlruns").mkdir(exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("risk-aware-portfolio-synthetic")
    frame, returns = market()
    predictions, folds, errors = forecast(frame, returns)
    names = ["equal_weight", "xgboost", "lstm", "transformer", "xgboost_no_cvar", "xgboost_no_turnover", "xgboost_flat_vol"]
    report = {"data": "synthetic six-asset market; no downloaded prices", "seed": SEED,
              "input_sha256": digest(returns), "daily_test_observations": 240,
              "folds": folds, "forecast_errors": errors, "strategies": {}, "cost_sensitivity": {},
              "versions": {p: importlib.metadata.version(p) for p in ["numpy", "polars", "torch", "xgboost", "cvxpy", "mlflow-skinny"]},
              "run_ids": {}}
    daily_rows, trace_rows, forecast_rows = [], [], []
    with mlflow.start_run(run_name="walk-forward-study") as parent:
        report["run_ids"]["parent"] = parent.info.run_id
        mlflow.log_params({"seed": SEED, "dataset_sha256": digest(returns), "folds": 3,
                           "horizon": 5, "lookback": 12, "cost_bps": 5, "cvar_alpha": .95,
                           "cvar_limit": .06, "turnover_limit": .30, "risk_weight": .6,
                           "epochs": 18, "learning_rate": .005, "xgb_estimators": 80, "xgb_depth": 3})
        for name in names:
            source = name if name in predictions else "xgboost"
            metrics, daily, traces = backtest(returns, predictions[source], name)
            report["strategies"][name] = metrics
            trace_rows.extend({"strategy": name, **row} for row in traces)
            daily_rows.extend({"strategy": name, "test_day": i, "net_return": value} for i, value in enumerate(daily))
            with mlflow.start_run(run_name=name, nested=True) as run:
                report["run_ids"][name] = run.info.run_id
                mlflow.log_param("strategy", name)
                mlflow.log_metrics(metrics)
        for cost in [0, 5, 20]:
            metrics, _, _ = backtest(returns, predictions["xgboost"], "xgboost", cost)
            report["cost_sensitivity"][str(cost)] = metrics
        shocks = {"equity_selloff": [-.12, -.09, -.11, .02, -.04, .015],
                  "inflation_shock": [-.08, -.07, -.08, -.08, -.07, .03],
                  "credit_liquidity": [-.06, -.05, -.07, .01, -.12, -.02]}
        report["stress"] = {name: {shock: float(np.array(next(r for r in reversed(trace_rows) if r["strategy"]==name)["weights"]) @ vector)
                                   for shock, vector in shocks.items()} for name in names}
        report["stress_scope"] = "Hypothetical single-period shocks applied to last rebalance target weights; not a rerun of the backtest."
        for model, values in predictions.items():
            for day, pred in values.items():
                forecast_rows.extend({"model": model, "day": day, "asset": asset,
                                      "mean_5d": float(pred[i, 0]), "vol_5d": float(pred[i, 1])} for i, asset in enumerate(ASSETS))
        pl.DataFrame(forecast_rows).write_csv("results/forecasts.csv")
        pl.DataFrame(daily_rows).write_csv("results/daily_returns.csv")
        Path("results/weights.json").write_text(json.dumps(trace_rows, indent=2)+"\n")
        Path("results/metrics.json").write_text(json.dumps(report, indent=2)+"\n")
        for filename in ["metrics.json", "forecasts.csv", "daily_returns.csv", "weights.json"]:
            mlflow.log_artifact(f"results/{filename}")
    export_tracking(report)
    print(json.dumps({"strategies": report["strategies"], "cost_sensitivity": report["cost_sensitivity"]}, indent=2))


if __name__ == "__main__":
    main()
