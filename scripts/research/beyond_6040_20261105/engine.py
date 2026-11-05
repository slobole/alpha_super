"""Fast daily simulator for inverse-volatility / vol-target portfolios. Research-only.

Timing: weights decided from information through close t-1 are applied to the return of day t
(lag=1). lag=2 reproduces the author's "signals from closes up to the day before the trade".

Model: positions drift between rebalances; turnover only when weights change. Costs are
cost_bps per unit of absolute weight change in risky assets. Idle cash earns the chosen cash
rate; borrowing (exposure > 1) pays cash + borrow_spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Params:
    assets: tuple[str, ...] = ("r_stock", "r_bond", "r_gold")
    vol_window: int = 63            # per-asset vol window for weights
    port_vol_window: int = 63       # realised portfolio vol window for the overlay
    port_vol_estimator: str = "rolling"  # rolling | ewma | max2163 | none
    target_vol: float = 0.08
    trigger_buffer: float = 0.005   # overlay acts only when vol > target + buffer
    rebalance: str = "M"            # D | W | M | Q
    band: float = 0.0               # no-trade band on daily exposure changes (exposure units)
    weighting: str = "ivol"         # ivol | equal | erc | minvar | ivol_ew | fixed
    fixed_weights: tuple[float, ...] = ()
    max_exposure: float = 1.0       # 1.0 = no leverage
    per_asset_cap: float = 10.0     # 0.7 in the capped arm
    total_cap: float = 10.0         # 1.5 in the capped arm
    tilt: str = "none"              # none | ensemble | sma10
    tilt_strength: float = 0.5      # multiplier = (1 - k) + k * s,  s in [-1, 1]
    tilt_freq: str = "D"            # D or same as rebalance
    cash: str = "ff"                # ff | zero | tbill
    cost_bps: float = 5.0
    borrow_spread: float = 0.015
    lag: int = 1
    continuous_overlay: bool = False  # exposure = min(cap, target/vol) every day, no dead band

    def label(self) -> str:
        return (f"{self.weighting}-{self.vol_window}-{self.port_vol_window}{self.port_vol_estimator[:3]}-"
                f"t{int(self.target_vol*1000)}-b{int(self.trigger_buffer*1000)}-{self.rebalance}-band{int(self.band*100)}-"
                f"x{self.max_exposure}-{self.tilt}{self.tilt_strength if self.tilt!='none' else ''}-{self.cash}-c{self.cost_bps}")


def _rebalance_mask(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    if freq == "D":
        return np.ones(len(idx), dtype=bool)
    if freq == "W":
        key = idx.to_period("W").astype(str)
    elif freq == "M":
        key = idx.to_period("M").astype(str)
    elif freq == "Q":
        key = idx.to_period("Q").astype(str)
    else:
        raise ValueError(freq)
    key = np.asarray(key)
    last = np.r_[key[1:] != key[:-1], True]   # last session of each period
    return last


def _ensemble_trend(R: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    sig_d = R.rolling(vol_window).std()
    out = 0.0
    for h in (21, 63, 126, 252):
        z = (R.rolling(h).sum()) / (sig_d * np.sqrt(h))
        out = out + z
    z = (out / 4.0).clip(-2.0, 2.0) / 2.0
    return z


def _sma10_trend(R: pd.DataFrame) -> pd.DataFrame:
    px = (1 + R.fillna(0)).cumprod()
    sma = px.rolling(210).mean()
    return np.sign(px - sma).where(sma.notna())


def _weights(R: pd.DataFrame, p: Params) -> pd.DataFrame:
    n = R.shape[1]
    sig = R.rolling(p.vol_window).std() * np.sqrt(TRADING_DAYS)
    if p.weighting == "equal":
        w = pd.DataFrame(1.0 / n, index=R.index, columns=R.columns).where(sig.notna())
    elif p.weighting == "fixed":
        w = pd.DataFrame([p.fixed_weights] * len(R), index=R.index, columns=R.columns).where(sig.notna())
    elif p.weighting in ("ivol", "ivol_ew"):
        inv = 1.0 / sig
        w = inv.div(inv.sum(axis=1), axis=0)
        if p.weighting == "ivol_ew":
            w = 0.5 * w + 0.5 / n
    elif p.weighting in ("erc", "minvar"):
        w = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
        vals = R.values
        reb = _rebalance_mask(R.index, p.rebalance)
        for t in np.where(reb)[0]:
            if t < p.vol_window:
                continue
            C = np.cov(vals[t - p.vol_window + 1:t + 1].T)
            if not np.all(np.isfinite(C)):
                continue
            if p.weighting == "minvar":
                x = _minvar(C)
            else:
                x = _erc(C)
            w.iloc[t] = x
        w = w.ffill()
    else:
        raise ValueError(p.weighting)
    return w


def _minvar(C: np.ndarray) -> np.ndarray:
    n = len(C)
    # long-only minimum variance via projected gradient (small n)
    x = np.full(n, 1.0 / n)
    for _ in range(2000):
        g = C @ x
        x = x - 0.5 * g / (np.trace(C) / n)
        x = np.clip(x, 0, None)
        s = x.sum()
        x = x / s if s > 0 else np.full(n, 1.0 / n)
    return x


def _erc(C: np.ndarray) -> np.ndarray:
    n = len(C)
    x = 1.0 / np.sqrt(np.diag(C))
    x = x / x.sum()
    for _ in range(500):
        rc = x * (C @ x)
        x = x * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.5
        x = x / x.sum()
    return x


def run(panel: pd.DataFrame, p: Params, start: str | None = None, end: str | None = None) -> dict:
    """Simulate. Returns dict with 'ret' (daily portfolio return, pd.Series), 'exposure', 'weights', 'turnover'."""
    R = panel[list(p.assets)].copy()
    if end is not None:
        R = R.loc[:end]
    cash_annual = {"ff": panel["dff"], "zero": panel["dff"] * 0.0, "tbill": panel["tbill"]}[p.cash].reindex(R.index).ffill()
    c_daily = cash_annual.values / TRADING_DAYS
    borrow_daily = (cash_annual.values + p.borrow_spread) / TRADING_DAYS

    W = _weights(R, p)                       # target weights among risky assets (sum 1), info through t
    if p.tilt != "none":
        if p.tilt == "ensemble":
            s = _ensemble_trend(R, p.vol_window)
        elif p.tilt == "sma10":
            s = _sma10_trend(R)
        else:
            raise ValueError(p.tilt)
        mult = (1.0 - p.tilt_strength) + p.tilt_strength * s
        if p.tilt_freq != "D":
            reb = _rebalance_mask(R.index, p.rebalance)
            mult = mult.where(pd.Series(reb, index=R.index), axis=0).ffill()
        W_t = W * mult                        # tilted weights, sum <= 1 (rest is cash)
    else:
        W_t = W

    reb = _rebalance_mask(R.index, p.rebalance)
    T, n = R.shape
    rv = R.fillna(0.0).values
    Wt = W_t.values
    Wb = W.values
    lag = p.lag

    # realised portfolio vol of the risky sleeve as held: use base weights (current target) applied to history
    # -> daily series of risky-sleeve returns at current target weights, computed causally: r_p(t) = w(t-1).r(t)
    hold = np.full(n, np.nan)
    pos = np.zeros(n)            # dollar weights of risky assets (fraction of NAV)
    cur_target = np.full(n, np.nan)
    last_tgt = np.full(n, -1.0)
    exposure = 1.0
    nav = 1.0
    out_ret = np.zeros(T)
    out_exp = np.full(T, np.nan)
    out_turn = np.zeros(T)
    out_pos = np.full((T, n), np.nan)
    sleeve_hist = np.full(T, np.nan)   # risky-sleeve (unit exposure) daily return as held
    ewma_var = np.nan
    lam = 0.94
    pvw = p.port_vol_window
    started = False

    for t in range(T):
        ti = t - lag                  # information index
        if ti < 0 or not np.all(np.isfinite(Wt[ti])):
            out_pos[t] = 0.0
            out_ret[t] = c_daily[t]
            nav *= 1 + out_ret[t]
            continue
        # --- decide target before today's return (info through ti)
        trade_today = reb[ti] or not started
        new_target = cur_target.copy()
        if trade_today:
            new_target = Wt[ti].copy()
            started = True
        elif p.tilt != "none" and p.tilt_freq == "D":
            # daily tilt update keeps base weights, changes multiplier
            new_target = Wt[ti].copy()
        # overlay: realised vol of sleeve as held
        if p.port_vol_estimator == "none":
            sig_p = np.nan
        elif p.port_vol_estimator == "ewma":
            sig_p = np.sqrt(ewma_var * TRADING_DAYS) if np.isfinite(ewma_var) else np.nan
        else:
            lo = max(0, ti - pvw + 1)
            h = sleeve_hist[lo:ti + 1]
            h = h[np.isfinite(h)]
            sig_p = h.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(h) >= min(pvw, 40) else np.nan
            if p.port_vol_estimator == "max2163":
                h2 = sleeve_hist[max(0, ti - 20):ti + 1]
                h2 = h2[np.isfinite(h2)]
                if len(h2) >= 15 and np.isfinite(sig_p):
                    sig_p = max(sig_p, h2.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if np.isfinite(sig_p) and p.continuous_overlay:
            new_exp = min(p.max_exposure, p.target_vol / sig_p)
        elif np.isfinite(sig_p):
            if sig_p > p.target_vol + p.trigger_buffer:
                new_exp = min(p.max_exposure, p.target_vol / sig_p)
            elif p.max_exposure > 1.0 and sig_p < p.target_vol - p.trigger_buffer:
                new_exp = min(p.max_exposure, p.target_vol / sig_p)
            else:
                new_exp = 1.0 if p.max_exposure <= 1.0 else exposure if np.isfinite(exposure) else 1.0
                if p.max_exposure > 1.0 and not np.isfinite(exposure):
                    new_exp = 1.0
        else:
            new_exp = 1.0
        if trade_today or abs(new_exp - exposure) >= p.band:
            exposure = new_exp
        # final risky weights
        tgt_total = new_target * exposure
        tgt_total = np.minimum(tgt_total, p.per_asset_cap)
        s = tgt_total.sum()
        if s > p.total_cap:
            tgt_total *= p.total_cap / s
        # trade only when the final target changed since the last trade (rebalance day, tilt update, or an
        # exposure change that passed the band). *** CRITICAL *** compare with the last TRADED target, not with
        # the drifted position, so that drift alone never triggers a trade.
        if trade_today or not np.allclose(tgt_total, last_tgt, atol=1e-12):
            turn = np.abs(tgt_total - pos).sum()
            pos = tgt_total.copy()
            cur_target = new_target
            last_tgt = tgt_total.copy()
        else:
            turn = 0.0
        # --- earn today's return
        risky = float(pos @ rv[t])
        gross = pos.sum()
        cash_w = 1.0 - gross
        fin = cash_w * c_daily[t] if cash_w >= 0 else cash_w * borrow_daily[t]
        r = risky + fin - turn * p.cost_bps / 1e4
        out_ret[t] = r
        out_exp[t] = gross
        out_turn[t] = turn
        out_pos[t] = pos
        # sleeve history at unit exposure for the overlay (base weights, not tilted, as the author's rule)
        if np.all(np.isfinite(Wb[ti])):
            sleeve_hist[t] = float(Wb[ti] @ rv[t])
            ewma_var = sleeve_hist[t] ** 2 if not np.isfinite(ewma_var) else lam * ewma_var + (1 - lam) * sleeve_hist[t] ** 2
        # drift positions
        pos = pos * (1 + rv[t]) / (1 + r) if (1 + r) != 0 else pos
        nav *= 1 + r

    idx = R.index
    res = {
        "ret": pd.Series(out_ret, index=idx, name="ret"),
        "exposure": pd.Series(out_exp, index=idx),
        "turnover": pd.Series(out_turn, index=idx),
        "weights": pd.DataFrame(out_pos, index=idx, columns=list(p.assets)),
        "cash_daily": pd.Series(c_daily, index=idx),
        "params": p,
    }
    if start is not None:
        for k in ("ret", "exposure", "turnover", "weights", "cash_daily"):
            res[k] = res[k].loc[start:]
    return res


# ---------------------------------------------------------------- benchmarks
def fixed_mix(panel: pd.DataFrame, weights: dict[str, float], rebalance: str = "M", cash: str = "ff",
              cost_bps: float = 5.0, start=None, end=None) -> dict:
    assets = tuple(weights.keys())
    p = Params(assets=assets, weighting="fixed", fixed_weights=tuple(weights.values()), port_vol_estimator="none",
               rebalance=rebalance, cash=cash, cost_bps=cost_bps, vol_window=2)
    return run(panel, p, start=start, end=end)


# ---------------------------------------------------------------- metrics
def metrics(ret: pd.Series, cash_daily: pd.Series | None = None, exposure: pd.Series | None = None,
            turnover: pd.Series | None = None) -> dict:
    r = ret.dropna()
    if len(r) < 60:
        return {}
    n = len(r)
    nav = (1 + r).cumprod()
    years = n / TRADING_DAYS
    cagr = nav.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    out = {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "years": round(years, 2),
           "cagr": cagr, "vol": vol, "sharpe_raw": r.mean() / r.std() * np.sqrt(TRADING_DAYS)}
    if cash_daily is not None:
        ex = r - cash_daily.reindex(r.index).fillna(0.0)
        out["excess_cagr"] = (1 + ex).prod() ** (1 / years) - 1
        out["sharpe"] = ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)
    else:
        out["sharpe"] = out["sharpe_raw"]
    dd = nav / nav.cummax() - 1
    out["max_dd"] = dd.min()
    out["calmar"] = cagr / abs(dd.min()) if dd.min() < 0 else np.nan
    # time under water (longest)
    uw = (dd < 0).astype(int)
    runs = (uw.groupby((uw != uw.shift()).cumsum()).cumsum() * uw)
    out["longest_underwater_years"] = runs.max() / TRADING_DAYS
    m = (1 + r).resample("ME").prod() - 1
    out["worst_12m"] = ((1 + m).rolling(12).apply(np.prod, raw=True) - 1).min()
    out["worst_36m"] = ((1 + m).rolling(36).apply(np.prod, raw=True) - 1).min()
    out["cvar5_daily"] = r[r <= r.quantile(0.05)].mean()
    out["cvar5_monthly"] = m[m <= m.quantile(0.05)].mean()
    out["skew_monthly"] = m.skew()
    out["pct_positive_years"] = float(((1 + r).resample("YE").prod() - 1 > 0).mean())
    if exposure is not None:
        e = exposure.reindex(r.index).dropna()
        out["avg_exposure"] = e.mean()
        out["pct_fully_invested"] = float((e >= 0.999).mean())
    if turnover is not None:
        out["turnover_per_year"] = turnover.reindex(r.index).sum() / years
    return out


def block_bootstrap_diff(a: pd.Series, b: pd.Series, cash: pd.Series, n_draws: int = 2000, block: int = 63,
                         seed: int = 7) -> dict:
    """Stationary block bootstrap of (excess Sharpe(a) - excess Sharpe(b)) on the common index."""
    j = pd.concat([a, b, cash], axis=1).dropna()
    xa = (j.iloc[:, 0] - j.iloc[:, 2]).values
    xb = (j.iloc[:, 1] - j.iloc[:, 2]).values
    T = len(xa)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_draws)
    sa = np.empty(n_draws)
    for k in range(n_draws):
        idx = np.empty(T, dtype=int)
        pos = 0
        while pos < T:
            s0 = rng.integers(0, T)
            L = rng.geometric(1.0 / block)
            L = min(L, T - pos)
            idx[pos:pos + L] = (s0 + np.arange(L)) % T
            pos += L
        ya, yb = xa[idx], xb[idx]
        sa[k] = ya.mean() / ya.std() * np.sqrt(TRADING_DAYS)
        diffs[k] = sa[k] - yb.mean() / yb.std() * np.sqrt(TRADING_DAYS)
    return {"diff_median": float(np.median(diffs)), "diff_p05": float(np.percentile(diffs, 5)),
            "diff_p95": float(np.percentile(diffs, 95)), "prob_a_better": float((diffs > 0).mean()),
            "sharpe_a_p05": float(np.percentile(sa, 5)), "sharpe_a_p95": float(np.percentile(sa, 95)), "n_days": T}


def block_metrics(ret: pd.Series, cash_daily: pd.Series, blocks: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for name, (s, e) in blocks.items():
        rows[name] = metrics(ret.loc[s:e], cash_daily.loc[s:e])
    return pd.DataFrame(rows).T


DECADES = {"1970s": ("1970-01-01", "1979-12-31"), "1980s": ("1980-01-01", "1989-12-31"),
           "1990s": ("1990-01-01", "1999-12-31"), "2000s": ("2000-01-01", "2009-12-31"),
           "2010s": ("2010-01-01", "2019-12-31"), "2020-22": ("2020-01-01", "2022-12-31"),
           "ETF era 2005-22": ("2005-01-01", "2022-12-31"), "Full 1970-2022": ("1970-01-01", "2022-12-31")}
VAULT_START = "2023-01-01"
DEV_END = "2022-12-31"


def to_jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, Params):
            out[k] = asdict(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
