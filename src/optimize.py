import cvxpy as cp
import numpy as np


def historical_blocks(returns, day, horizon=5):
    history = returns[max(0, day-199):day+1]
    history = history[-(len(history)//horizon)*horizon:]
    return np.prod(1+history.reshape(-1, horizon, 6), axis=1)-1


def scenarios(history, mean, volatility):
    scaled = (history-history.mean(axis=0))/np.maximum(history.std(axis=0, ddof=1), 1e-6)
    return scaled*volatility+mean


def empirical_cvar(losses, alpha=.95):
    losses = np.asarray(losses, dtype=float)
    if len(losses)==0 or not 0 < alpha < 1:
        raise ValueError("Nonempty losses and alpha in (0,1) required")
    # Discrete Rockafellar-Uryasev objective; exact for non-integral tail counts too.
    return min(float(z+np.maximum(losses-z, 0).mean()/(1-alpha)) for z in losses)


def optimize(mean, simulated, previous, cost_bps=5, cvar_limit=.06, turnover_limit=.30, risk_weight=.6):
    w, eta = cp.Variable(6), cp.Variable()
    loss = -simulated @ w
    cvar = eta+cp.sum(cp.pos(loss-eta))/(len(simulated)*.05)
    turnover = cp.norm1(w-previous)
    constraints = [cp.sum(w)==1, w>=0, w<=.35, cp.sum(w[:3])<=.65,
                   cp.sum(w[3:5])<=.65]
    if turnover_limit is not None:
        constraints.append(turnover<=turnover_limit)
    if cvar_limit is not None:
        constraints.append(cvar<=cvar_limit)
    objective = mean @ w-cost_bps/10000*turnover
    if risk_weight:
        objective -= risk_weight*cvar
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        # Retain existing holdings; report fallback, never claim a feasible solution was found.
        return previous.copy(), {"status": problem.status, "fallback": True}
    weights = np.asarray(w.value).reshape(-1)
    weights[np.abs(weights)<1e-9] = 0
    weights /= weights.sum()
    return weights, {"status": problem.status, "fallback": False,
                     "scenario_cvar": empirical_cvar(-simulated @ weights),
                     "turnover": float(np.abs(weights-previous).sum())}


def hold(weights, daily_returns, fee=0):
    current = weights.copy()
    net = []
    for index, asset_returns in enumerate(daily_returns):
        gross = float(current @ asset_returns)
        net.append((1+gross)*(1-fee if index==0 else 1)-1)
        current = current*(1+asset_returns)/(1+gross)
    return np.array(net), current


def performance(daily):
    daily = np.asarray(daily)
    wealth = np.r_[1., np.cumprod(1+daily)]
    drawdown = wealth/np.maximum.accumulate(wealth)-1
    return {"net_return": float(wealth[-1]-1), "annualized_sharpe": float(daily.mean()/daily.std(ddof=1)*np.sqrt(252)),
            "max_drawdown": float(drawdown.min()), "annualized_volatility": float(daily.std(ddof=1)*np.sqrt(252))}
