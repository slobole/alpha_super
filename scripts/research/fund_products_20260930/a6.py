"""A6: streamlined menus rebuilt under one frozen rule set (SPEC_FROZEN.md, amendment A6).

Defensive: the owner's five slots (launch, next step, very defensive, more return, ideal target). Each slot has a
pre-declared default that a challenger replaces only by passing the challenge test. Growth: launch, more return,
MR target, and one 22% levered route per stage. Every candidate's bootstrap tail comes from one batched run with
the same seeds and arithmetic as ev.seeds_tail (checked against clean.json at start-up).

Usage: python a6.py   Writes <study>/report/a6.json.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import TBILL, Book, ga, lib
import evaluate as ev

CUT = pd.Timestamp("2017-06-30")
SPREAD = 0.015
TARGET_CAGR = 0.22
CASH_STEPS = [round(0.05 * k, 2) for k in range(13)]                      # 0% .. 60%
LIMITS = (-0.05, -0.07, -0.10, -0.15, -0.20, -0.25, -0.30)
RULES = {"DEF": (-0.07, "p10", 0.10), "CALM": (-0.05, "p7", 0.10)}
FLOOR = {"gfc": -0.01, "bear_2022": -0.01, "worst": -0.05}
GROWTH = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
GROWTH_PLUS = {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09}
GROWTH_MR = {"taa3x": 0.27, "taa3x_1n": 0.27, "ndx_vxn": 0.07, "dv2": 0.09, "hpi_vote": 0.09, "hpi_ibs_rsi": 0.09,
             "btal_qqq": 0.03, "core5": 0.03, TBILL: 0.06}
THREE_X = {"taa3x", "taa3x_1n"}
REG_T = {"three_x": 0.75, "other": 0.50}
GATED_PODS = {"dv2", "hpi_vote", "hpi_ibs_rsi"}   # A4 / A6-c: stock MR waits for measured live slippage <= 3-4 bps/side
FIN_TAG = "dtb3+spread/act360"                     # A6-c: margin financed like negative cash in the frames
CAP_TOP = 250e6                                    # top of the capacity grid: values here mean "at least"


def blend(*parts: tuple[float, dict]) -> dict:
    """Weighted sum of weight dicts (each normalised first); the result sums to 1."""
    out: dict = {}
    for share, w in parts:
        tot = sum(w.values())
        for k, v in w.items():
            out[k] = out.get(k, 0.0) + share * v / tot
    s = sum(out.values())
    return {k: v / s for k, v in out.items() if v > 1e-12}


def with_cash(w: dict, c: float) -> dict:
    return blend((1 - c, w), (c, {TBILL: 1.0})) if c > 0 else blend((1.0, w))


def pro_rata(w: dict, pod: str, share: float) -> dict:
    return blend((1 - share, w), (share, {pod: 1.0}))


C_L = {"core5": 0.6, "btal_qqq": 0.4}
BASES = {
    # Launch (wired at launch once CORE5 is wired).
    "C_L": ("launch", C_L), "C_5050": ("launch", {"core5": 0.5, "btal_qqq": 0.5}),
    "C_L+DV2_10": ("launch", pro_rata(C_L, "dv2", 0.10)), "C_L+HPI_10": ("launch", pro_rata(C_L, "hpi_vote", 0.10)),
    # Next step (one new wiring).
    "C_N": ("next", {"core5": 1 / 3, "btal_qqq": 1 / 3, "etf_dv2": 1 / 3}),
    "C_N6040": ("next", pro_rata(C_L, "etf_dv2", 1 / 3)),
    "C_L+IND_25": ("next", pro_rata(C_L, "etf_dv2", 0.25)), "C_L+IND_20": ("next", pro_rata(C_L, "etf_dv2", 0.20)),
    "C_DS3": ("next", {"core5": 1 / 3, "btal_qqq": 1 / 3, "downshock": 1 / 3}),
    "C_L+DS_20": ("next", pro_rata(C_L, "downshock", 0.20)), "C_L+DS_10": ("next", pro_rata(C_L, "downshock", 0.10)),
    # Ideal target.
    "C_T": ("target", {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}),
    "C_T+DS_05": ("target", pro_rata({"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}, "downshock", 0.05)),
    "C_T+DS_10": ("target", pro_rata({"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}, "downshock", 0.10)),
    "C_T6040": ("target", {"core5": 0.30, "btal_qqq": 0.20, "eom_flow": 0.25, "etf_dv2": 0.25}),
    "C_EOMDS": ("target", {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "downshock": 0.25}),
    "C_EOM3": ("target", {"core5": 1 / 3, "btal_qqq": 1 / 3, "eom_flow": 1 / 3}),
}
DEFAULT = {"launch": "C_L", "next": "C_N", "target": "C_T"}
SWEEP_CONTEXT = {"C_L": C_L, "C_N": BASES["C_N"][1], "C_T": BASES["C_T"][1]}
SWEEP_W = (0.05, 0.10, 0.15, 0.20, 0.25)


def gross_dd(R: np.ndarray, idx_t: np.ndarray) -> np.ndarray:
    """ga.bootstrap_paths' gross max-DD arithmetic, gross only (paths x books). idx_t is day-major (days x paths)."""
    n, reps = idx_t.shape
    b = R.shape[1]
    gross, peak, dd, tmp = np.ones((reps, b)), np.ones((reps, b)), np.zeros((reps, b)), np.empty((reps, b))
    for t in range(n):
        np.add(R[idx_t[t]], 1.0, out=tmp)
        gross *= tmp
        np.maximum(peak, gross, out=peak)
        np.minimum(dd, gross / peak - 1.0, out=dd)
    return dd


class Lab:
    def __init__(self) -> None:
        self.data = lib.load_inputs()
        self.frames = ga.frames(self.data)
        self.frame, self.start = self.frames["main"]
        self.rf = self.frame[TBILL]
        self.dtb3 = lib.dtb3_annual_rate(self.frame.index)
        self.days = pd.Series(self.frame.index, index=self.frame.index).diff().dt.days
        block = self.frame.loc[self.start:fp.END].fillna(0.0)
        h = hashlib.sha256(np.ascontiguousarray(block.to_numpy(dtype=float)).tobytes())
        h.update(("|".join(block.columns) + f"|{self.start}|{fp.END}|2000|63.0|{fp.SEED0}").encode())
        self.fp = h.hexdigest()[:16]
        self.cands: dict[str, dict] = {}
        self.acc: dict[str, list[dict]] = {}       # name -> per-seed {limit: P(DD < limit)}, in seed order
        self._cache: dict = {}
        self._idx: dict[int, np.ndarray] = {}
        self._disk_path = fp.STUDY / "report" / "a6_tail_cache.json"
        raw = json.loads(self._disk_path.read_text(encoding="utf-8")) if self._disk_path.exists() else {}
        # A6-c: pre-fingerprint entries were computed this session on the same frame; adopt the unlevered ones only
        # (levered ones used BIL + spread financing and are dropped).
        self._disk = {}
        for k, v in raw.items():
            key = json.loads(k)
            if len(key) in (5, 6):
                self._disk[k] = v
            elif len(key) == 3 and round(float(key[1]), 4) == 1.0:
                self._disk[json.dumps([self.fp, key[0], 1.0, 0.0, ""])] = v

    # ── series ──
    def ret(self, w: dict, L: float = 1.0, frame_key: str = "main", spread: float = SPREAD) -> pd.Series:
        key = (tuple(sorted((k, round(v, 12)) for k, v in w.items())), round(L, 4), frame_key, spread)
        if key not in self._cache:
            fr, s = self.frames[frame_key]
            r = lib.book_returns(fr, Book("x", tuple(w), "EQ", w), s)
            if L != 1.0:
                r = L * r - (L - 1) * self.fin(r.index, spread)
            self._cache[key] = r
        return self._cache[key]

    def fin(self, index: pd.DatetimeIndex, spread: float = SPREAD) -> pd.Series:
        """Daily margin cost per borrowed dollar: (DTB3, prior observation, + spread) x calendar days / 360."""
        out = ((self.dtb3 + spread) * self.days / 360.0).reindex(index)
        assert not out.isna().any()
        return out

    def quick(self, r: pd.Series) -> dict:
        x = r.to_numpy()
        nav = np.r_[1, np.cumprod(1 + x)]
        cr = {k: float(lib.common.window_return_float(r, lo, hi)) for k, (lo, hi) in lib.CRISIS_DICT.items()}
        cdd = {}
        for k, (lo, hi) in lib.CRISIS_DICT.items():
            win = r[(r.index > pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))]
            w = np.r_[1.0, np.cumprod(1 + win.to_numpy())]
            cdd[k] = float((w / np.maximum.accumulate(w) - 1).min())
        yrs = (1 + r).groupby(r.index.year).prod() - 1
        return {"cagr": float(nav[-1] ** (252 / len(x)) - 1), "dd": float((nav / np.maximum.accumulate(nav) - 1).min()),
                "xs": ev.xsharpe(x, self.rf.reindex(r.index).to_numpy()), "vol": float(x.std(ddof=1) * np.sqrt(252)),
                "crises": cr, "crises_dd": cdd, "worst_crisis": float(min(cr.values())), "worst_year": float(yrs.min()),
                "recent_cagr": float(np.prod(1 + r.loc["2023-08-21":].to_numpy()) ** (252 / len(r.loc["2023-08-21":])) - 1)}

    def add(self, name: str, w: dict, L: float = 1.0, frame: str = "main", **meta) -> str:
        if name not in self.cands:
            r = self.ret(w, L, frame)
            self.cands[name] = {"w": w, "L": L, "frame": frame, "q": self.quick(r), **meta}
        return name

    # ── tails: batched per seed, cached on disk ──
    def _key(self, n: str) -> str:
        c = self.cands[n]
        lev = round(c["L"], 4) != 1.0
        key = [self.fp, sorted((k, round(v, 12)) for k, v in c["w"].items()), round(c["L"], 4), SPREAD if lev else 0.0, FIN_TAG if lev else ""]
        if c.get("frame", "main") != "main":
            key.append(c["frame"])
        return json.dumps(key)

    def idx(self, s: int) -> np.ndarray:
        """Day-major (days x paths) copy of the seed's bootstrap index, so each day's row is contiguous."""
        if s not in self._idx:
            n = len(self.ret(GROWTH))
            mat = lib.evaluation.stationary_bootstrap_index_mat(n, 2000, 63.0, fp.SEED0 + s)
            if s == 0:
                self._paths0 = mat.astype(np.int32)                     # path-major, for share()
            self._idx[s] = np.ascontiguousarray(mat.T.astype(np.int32))
        return self._idx[s]

    def run_seeds(self, names: list[str], seeds) -> None:
        """Make sure every book has the given seeds (computed in seed order, one batched pass per seed)."""
        names = list(dict.fromkeys(names))
        for n in names:
            if n not in self.acc:
                self.acc[n] = [{float(k): v for k, v in d.items()} for d in self._disk.get(self._key(n), [])]
        dirty = False
        for s in seeds:
            todo = [n for n in names if len(self.acc[n]) == s]
            if not todo:
                continue
            R = np.column_stack([self.ret(self.cands[n]["w"], self.cands[n]["L"], self.cands[n].get("frame", "main")).to_numpy() for n in todo])
            assert not np.isnan(R).any()
            dd = gross_dd(R, self.idx(s))
            for j, n in enumerate(todo):
                self.acc[n].append({L: float((dd[:, j] < L).mean()) for L in LIMITS})
                self._disk[self._key(n)] = [{str(k): v for k, v in d.items()} for d in self.acc[n]]
            dirty = True
            print(f"  seed {s}: {len(todo)} books", flush=True)
        if dirty:
            self._disk_path.write_text(json.dumps(self._disk), encoding="utf-8")

    @property
    def tails(self) -> dict:
        out = {}
        for n, a in self.acc.items():
            if len(a) < 10:
                continue
            t = {}
            for L in LIMITS:
                key = f"p{int(round(-L * 100))}"
                vals = [d[L] for d in a[:10]]
                t[key], t[key + "_max"] = float(np.mean(vals)), float(np.max(vals))
            out[n] = t
        return out

    def run_tails(self, names: list[str]) -> None:
        self.run_seeds(names, range(10))

    def seed0(self, n: str, key: str) -> float:
        return self.acc[n][0][-int(key[1:]) / 100]

    # ── rules ──
    def floor_ok(self, n: str) -> bool:
        q = self.cands[n]["q"]
        return q["crises"]["gfc"] >= FLOOR["gfc"] and q["crises"]["bear_2022"] >= FLOOR["bear_2022"] and q["worst_crisis"] >= FLOOR["worst"]

    def rule_ok(self, n: str, rule: str) -> bool:
        build, key, cap = RULES[rule]
        t = self.tails[n]
        return self.cands[n]["q"]["dd"] >= build and t[key] <= cap and t[key + "_max"] <= cap

    def levels(self, base: str, rule: str) -> list[str]:
        build = RULES[rule][0]
        lv = (f"{base}|c{c:.2f}" for c in CASH_STEPS)
        return [n for n in lv if n in self.cands and self.cands[n]["q"]["dd"] >= build and self.floor_ok(n)]

    def min_cash(self, base: str, rule: str) -> str | None:
        """Smallest cash level passing rule + floor. Exact: levels are scanned upward, and a level whose seed 0
        already breaks the cap fails for certain (the rule also caps the worst seed); the rest get all ten seeds."""
        key, cap = RULES[rule][1], RULES[rule][2]
        for n in self.levels(base, rule):
            self.run_seeds([n], [0])
            if self.seed0(n, key) > cap:
                continue
            self.run_tails([n])
            if self.rule_ok(n, rule):
                return n
        return None

    def prefetch(self, bases: list[str], rule: str) -> None:
        """Batch what min_cash will need: walk every base one level at a time on seed 0 (one batch per step), then
        give the first surviving level of each base all ten seeds in one batch."""
        key, cap = RULES[rule][1], RULES[rule][2]
        lv = {b: self.levels(b, rule) for b in bases}
        pos = {b: 0 for b in bases}
        first: dict[str, str] = {}
        while True:
            step = [lv[b][pos[b]] for b in bases if b not in first and pos[b] < len(lv[b])]
            if not step:
                break
            self.run_seeds(step, [0])
            for b in bases:
                if b in first or pos[b] >= len(lv[b]):
                    continue
                if self.seed0(lv[b][pos[b]], key) <= cap:
                    first[b] = lv[b][pos[b]]
                else:
                    pos[b] += 1
        self.run_tails(list(first.values()))

    # ── comparisons ──
    def _share(self, ra: pd.Series, rb: pd.Series, rf: pd.Series) -> float:
        A = pd.concat([ra, rb, rf.reindex(ra.index)], axis=1).dropna().to_numpy()
        self.idx(0)
        idx = self._paths0
        assert idx.shape[1] == len(A)
        wins = 0
        for k in range(2000):
            s = A[idx[k]]
            wins += ev.xsharpe(s[:, 0], s[:, 2]) > ev.xsharpe(s[:, 1], s[:, 2])
        return wins / 2000

    def share(self, a: str, b: str, frame_key: str = "main") -> float:
        fr = self.frames[frame_key][0]
        ra = self.ret(self.cands[a]["w"], self.cands[a]["L"], frame_key)
        rb = self.ret(self.cands[b]["w"], self.cands[b]["L"], frame_key)
        return self._share(ra, rb, fr[TBILL])

    def plus10(self, n: str) -> pd.Series:
        c = self.cands[n]
        r0 = self.ret(c["w"], c["L"])
        r5 = self.ret(c["w"], c["L"], "s3_plus_5bps").reindex(r0.index)
        return r0 + 2 * (r5 - r0)

    def share_plus10(self, a: str, b: str) -> float:
        return self._share(self.plus10(a), self.plus10(b), self.rf)

    def frame_stats(self, n: str, frame_key: str) -> dict:
        c = self.cands[n]
        r = self.ret(c["w"], c["L"], frame_key)
        fr = self.frames[frame_key][0]
        nav = np.cumprod(1 + r.to_numpy())
        return {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "xs": ev.xsharpe(r.to_numpy(), fr[TBILL].reindex(r.index).to_numpy()),
                "dd": float((np.r_[1, nav] / np.maximum.accumulate(np.r_[1, nav]) - 1).min())}

    def halves(self, n: str) -> tuple[float, float]:
        c = self.cands[n]
        r = self.ret(c["w"], c["L"])
        rf = self.rf.reindex(r.index)
        a, b = r.index <= CUT, r.index > CUT
        return ev.xsharpe(r[a].to_numpy(), rf[a].to_numpy()), ev.xsharpe(r[b].to_numpy(), rf[b].to_numpy())

    def challenge(self, ch: str, de: str, breach: str, cost_metric: str) -> dict:
        sh = self.share(ch, de)
        ex_c, ex_d = self.frame_stats(ch, "s6_exact"), self.frame_stats(de, "s6_exact")
        f5_c, f5_d = self.frame_stats(ch, "s3_plus_5bps"), self.frame_stats(de, "s3_plus_5bps")
        h_c, h_d = self.halves(ch), self.halves(de)
        checks = {"share_ge_80": sh >= 0.80, "breach_no_worse": self.tails[ch][breach] <= self.tails[de][breach],
                  "exact_xs_higher": ex_c["xs"] > ex_d["xs"], "plus5_not_lower": f5_c[cost_metric] >= f5_d[cost_metric],
                  "h1_higher": h_c[0] > h_d[0], "h2_higher": h_c[1] > h_d[1]}
        return {"challenger": ch, "default": de, "share": sh, "breach": [self.tails[ch][breach], self.tails[de][breach]],
                "exact_xs": [ex_c["xs"], ex_d["xs"]], "plus5": [f5_c[cost_metric], f5_d[cost_metric]],
                "halves_xs": [list(h_c), list(h_d)], "checks": checks, "passed": bool(all(checks.values()))}

    def solve_L(self, w: dict, target: float = TARGET_CAGR, spread: float = SPREAD) -> float:
        r0 = self.ret(w)
        r = r0.to_numpy()
        f = self.fin(r0.index, spread).to_numpy()
        for L in np.arange(1.0, 4.001, 0.01):
            x = L * r - (L - 1) * f
            if np.prod(1 + x) ** (252 / len(x)) - 1 >= target:
                return round(float(L), 2)
        raise ValueError("22% not reachable below 4x")

    def reg_t(self, w: dict, L: float) -> float:
        return float(sum(v * L * (REG_T["three_x"] if k in THREE_X else REG_T["other"]) for k, v in w.items()))


def main() -> int:
    lab = Lab()
    # Start-up check: batched gross tails equal ev.seeds_tail (clean.json) for the growth launch book.
    C = json.loads((fp.STUDY / "report" / "clean.json").read_text(encoding="utf-8"))
    Rchk = lab.ret(GROWTH).to_numpy()[:, None]
    p20, p25 = [], []
    for s_ in range(10):
        ddc = gross_dd(Rchk, lab.idx(s_))[:, 0]
        p20.append(float((ddc < -0.20).mean()))
        p25.append(float((ddc < -0.25).mean()))
    assert abs(np.mean(p20) - C["fund_growth"]["tail"]["p20"]) < 1e-12, "tail arithmetic drifted"
    assert abs(np.mean(p25) - C["fund_growth"]["tail"]["p25"]) < 1e-12
    print("tail check OK (recomputed)", np.mean(p20), "fingerprint", lab.fp, flush=True)

    # ── Batch 1: every defensive base x cash, plus the downshock sweep ──
    names = []
    for base, (stage, w) in BASES.items():
        for c in CASH_STEPS:
            names.append(lab.add(f"{base}|c{c:.2f}", with_cash(w, c), base=base, stage=stage, cash=c))
    sweep_names = []
    for ctx, w in SWEEP_CONTEXT.items():
        for ds in SWEEP_W:
            base = f"SWEEP_{ctx}+DS_{int(ds * 100):02d}"
            for c in CASH_STEPS:
                sweep_names.append(lab.add(f"{base}|c{c:.2f}", with_cash(pro_rata(w, "downshock", ds), c), base=base, stage="sweep", cash=c))
    # Raw books (always reported) get all ten seeds; each base's DEF boundary is located on seed 0 first.
    raws = [n for n in names + sweep_names if n.endswith("|c0.00")]
    print("batch 1:", len(raws), "raw books;", len(names) + len(sweep_names), "candidates", flush=True)
    lab.run_tails(raws)
    lab.prefetch(list(BASES) + sorted({lab.cands[n]["base"] for n in sweep_names}), "DEF")

    out: dict = {"spec": "A6", "defensive": {}, "challenges": {}, "sweep": [], "growth": {}, "notes": {}}

    # ── Defensive slots ──
    def pick(stage: str, rule: str = "DEF") -> tuple[str, list, str | None]:
        de = lab.min_cash(DEFAULT[stage], rule)
        if de is None:
            raise ValueError(f"default {DEFAULT[stage]} has no passing cash level")
        log, best, best_share, gated, gated_share = [], de, -1.0, None, -1.0
        for base, (st, w) in BASES.items():
            if st != stage or base == DEFAULT[stage]:
                continue
            ch = lab.min_cash(base, rule)
            if ch is None:
                log.append({"challenger": base, "default": de, "passed": False, "reason": "no passing cash level"})
                continue
            res = lab.challenge(ch, de, RULES[rule][1], "xs")
            res["gated"] = bool(set(w) & GATED_PODS)
            log.append(res)
            if res["passed"] and res["gated"]:
                if res["share"] > gated_share:
                    gated, gated_share = ch, res["share"]
            elif res["passed"] and res["share"] > best_share:
                best, best_share = ch, res["share"]
        return best, log, gated

    launch, log_l, launch_gated = pick("launch")
    nxt, log_n, _ = pick("next")
    target, log_t, _ = pick("target")
    out["challenges"].update({"launch": log_l, "next": log_n, "target": log_t})
    core_of = lambda n: blend((1.0, BASES[lab.cands[n]["base"]][1]))
    calm = lab.min_cash(lab.cands[launch]["base"], "CALM")
    calm_next = lab.min_cash(lab.cands[nxt]["base"], "CALM")

    # More return: the launch winner's core + growth slice g + cash c; highest CAGR passing DEF + floor.
    def richer(core: dict, tag: str) -> str | None:
        grid = []
        for g in [round(0.05 * k, 2) for k in range(1, 11)]:
            for c in [round(0.05 * k, 2) for k in range(11)]:
                w = with_cash(blend((1 - g, core), (g, GROWTH)), c)
                n = lab.add(f"RICH_{tag}|g{g:.2f}|c{c:.2f}", w, base=f"RICH_{tag}", stage="rich", cash=c, g=g)
                if lab.cands[n]["q"]["dd"] >= RULES["DEF"][0] and lab.floor_ok(n):
                    grid.append(n)
        grid.sort(key=lambda n: -lab.cands[n]["q"]["cagr"])
        # Exact: walk down the CAGR ranking; seed 0 above the cap rules a pair out for certain. The first pass is the
        # slot; the next two passes are reported as the frontier just below it.
        passing = []
        for i in range(0, len(grid), 8):
            chunk = grid[i:i + 8]
            lab.run_seeds(chunk, [0])
            live = [n for n in chunk if lab.seed0(n, "p10") <= RULES["DEF"][2]]
            lab.run_tails(live)
            passing += [n for n in chunk if n in live and lab.rule_ok(n, "DEF")]
            if len(passing) >= 3:
                break
        return (passing[0] if passing else None), passing[:3]

    def frontier_rows(names: list[str]) -> list[dict]:
        rows = []
        for n in names:
            q, t = lab.cands[n]["q"], lab.tails[n]
            rows.append({"name": n, "g": lab.cands[n]["g"], "cash": lab.cands[n]["cash"], "cagr": q["cagr"], "dd": q["dd"],
                         "gfc": q["crises"]["gfc"], "bear_2022": q["crises"]["bear_2022"], "worst": q["worst_crisis"], "p10": t["p10"], "p10_max": t["p10_max"],
                         "slack": {"dd": q["dd"] - RULES["DEF"][0], "gfc": q["crises"]["gfc"] - FLOOR["gfc"], "bear_2022": q["crises"]["bear_2022"] - FLOOR["bear_2022"],
                                   "worst": q["worst_crisis"] - FLOOR["worst"], "p10": RULES["DEF"][2] - t["p10_max"]}})
        return rows

    rich, front = richer(core_of(launch), "launch")
    rich_stage = "launch"
    out["notes"]["rich_frontier_launch"] = frontier_rows(front)
    if rich is None or lab.cands[rich]["q"]["cagr"] - lab.cands[launch]["q"]["cagr"] < 0.010:
        out["notes"]["rich_launch_candidate"] = rich
        rich, front = richer(core_of(nxt), "next")
        rich_stage = "next"
        out["notes"]["rich_frontier_next"] = frontier_rows(front)
    out["notes"]["rich_stage"] = rich_stage

    slots = {"launch": launch, "next": nxt, "calm": calm, "rich": rich, "target": target, "calm_next": calm_next,
             "launch_gated": launch_gated}
    out["next_vs_launch"] = lab.challenge(nxt, launch, "p10", "xs")
    if launch_gated:
        out["notes"]["launch_gated_shares"] = {"main": lab.share(launch_gated, launch),
                                               "plus5": lab.share(launch_gated, launch, "s3_plus_5bps"),
                                               "plus10": lab.share_plus10(launch_gated, launch)}
        print("GATED", launch_gated, out["notes"]["launch_gated_shares"], flush=True)
    for slot, n in slots.items():
        if n is None:
            out["defensive"][slot] = None
            continue
        c = lab.cands[n]
        out["defensive"][slot] = {"name": n, "weights": c["w"], "lever": 1.0, "cash": c.get("cash"), "g": c.get("g"),
                                  "q": c["q"], "tails": lab.tails[n], "floor_ok": lab.floor_ok(n),
                                  "frames": {f: lab.frame_stats(n, f) for f in ("s3_plus_5bps", "s6_exact", "s1_house_cash")},
                                  "halves_xs": list(lab.halves(n))}
        print("SLOT", slot, n, {k: round(v, 4) for k, v in c["q"].items() if isinstance(v, float)},
              {k: round(v, 4) for k, v in lab.tails[n].items() if not k.endswith("_max")}, {k: round(v, 3) for k, v in c["q"]["crises"].items()}, flush=True)
    for stage, log in (("launch", log_l), ("next", log_n), ("target", log_t)):
        for res in log:
            print("CHALLENGE", stage, res.get("challenger"), "vs", res.get("default"), "share", round(res.get("share", float("nan")), 3) if "share" in res else "-",
                  "passed", res["passed"], res.get("checks", res.get("reason")), flush=True)

    # Downshock sweep report: raw book and its smallest passing cash, per context and weight.
    for ctx in SWEEP_CONTEXT:
        for ds in (0.0,) + SWEEP_W:
            base = {"C_L": "C_L", "C_N": "C_N", "C_T": "C_T"}[ctx] if ds == 0.0 else f"SWEEP_{ctx}+DS_{int(ds * 100):02d}"
            raw = f"{base}|c0.00"
            mc = lab.min_cash(base, "DEF")
            T = lab.tails
            out["sweep"].append({"context": ctx, "ds": ds, "raw": {"q": lab.cands[raw]["q"], "tails": T.get(raw)},
                                 "min_cash": None if mc is None else {"name": mc, "cash": lab.cands[mc]["cash"], "q": lab.cands[mc]["q"], "tails": T[mc]},
                                 "floor_ok_raw": lab.floor_ok(raw)})
            q = lab.cands[raw]["q"]
            print("SWEEP", ctx, ds, "cagr", round(q["cagr"], 4), "xs", round(q["xs"], 3), "gfc", round(q["crises"]["gfc"], 3), "2022", round(q["crises"]["bear_2022"], 3),
                  "worst", round(q["worst_crisis"], 3), "p10", round(T[raw]["p10"], 4) if raw in T else "-", "min_cash", None if mc is None else lab.cands[mc]["cash"], flush=True)

    # ── Growth ──
    g_rows = {}
    for key, w in (("launch", GROWTH), ("plus", GROWTH_PLUS), ("mr", GROWTH_MR)):
        g_rows[key] = lab.add(f"G_{key}", w, base=f"G_{key}", stage="growth")
    route_names = []

    def lev(name: str, w: dict) -> str:
        L = lab.solve_L(w)
        return lab.add(name, w, L, base=name, stage="growth22")

    route_names.append(lev("G22_launch_alone", GROWTH))
    cores = {"launch": C_L, "next": BASES["C_N"][1], "target": BASES["C_T"][1]}     # literal cores (A6 text; A6-c fix)
    for stage, core in cores.items():
        for s in (1 / 3, 1 / 2, 2 / 3):
            route_names.append(lev(f"G22_{stage}_mix_{int(round(s * 100)):02d}", blend((s, GROWTH), (1 - s, core))))
    route_names.append(lev("G22_maxlev", blend((1.0, lab.cands[target]["w"]))))
    route_names.append(lev("G22_mr", GROWTH_MR))
    lab.run_tails(list(g_rows.values()) + route_names)

    def g_challenge(de: str, chs: list[str]) -> tuple[str, list]:
        best, best_share, log = de, -1.0, []
        for ch in chs:
            res = lab.challenge(ch, de, "p20", "cagr")
            log.append(res)
            if res["passed"] and res["share"] > best_share:
                best, best_share = ch, res["share"]
        return best, log

    g22_launch, log_gl = g_challenge("G22_launch_alone", [f"G22_launch_mix_{x}" for x in ("33", "50", "67")])
    g22_next, log_gn = g_challenge("G22_next_mix_50", ["G22_next_mix_33", "G22_next_mix_67"])
    g22_target, log_gt = g_challenge("G22_target_mix_50", ["G22_target_mix_33", "G22_target_mix_67"])
    out["challenges"].update({"g22_launch": log_gl, "g22_next": log_gn, "g22_target": log_gt})
    g_final = {"launch": g_rows["launch"], "plus": g_rows["plus"], "g22_launch": g22_launch, "g22_next": g22_next,
               "g22_target": g22_target, "g22_maxlev": "G22_maxlev", "mr": g_rows["mr"], "mr22": "G22_mr"}
    for slot, n in g_final.items():
        c = lab.cands[n]
        fin = {}
        if c["L"] != 1.0:
            for sp in (0.005, 0.025):
                r = lab.ret(c["w"], c["L"], spread=sp)
                fin[f"spread_{sp:.3f}"] = lab.quick(r)["cagr"]
        r5 = lab.ret(c["w"], c["L"], "s3_plus_5bps")
        r0 = lab.ret(c["w"], c["L"])
        r10 = r0 + 2 * (r5.reindex(r0.index) - r0)
        nav10 = np.prod(1 + r10.to_numpy())
        out["growth"][slot] = {"name": n, "weights": c["w"], "lever": c["L"], "q": c["q"], "tails": lab.tails[n],
                               "frames": {f: lab.frame_stats(n, f) for f in ("s3_plus_5bps", "s6_exact", "s1_house_cash", "s5_hpi_live_gap")},
                               "plus10_cagr": float(nav10 ** (252 / len(r10)) - 1), "financing": fin,
                               "reg_t": lab.reg_t(c["w"], c["L"]), "reg_t_flag": lab.reg_t(c["w"], c["L"]) > 0.90,
                               "halves_xs": list(lab.halves(n))}
        print("GROWTH", slot, n, "L", c["L"], {k: round(v, 4) for k, v in c["q"].items() if isinstance(v, float)},
              {k: round(v, 3) for k, v in lab.tails[n].items() if k in ("p20", "p25", "p30")}, "reg_t", round(lab.reg_t(c["w"], c["L"]), 2), flush=True)
    for stage, log in (("g22_launch", log_gl), ("g22_next", log_gn), ("g22_target", log_gt)):
        for res in log:
            print("CHALLENGE", stage, res["challenger"], "vs", res["default"], "share", round(res["share"], 3), "passed", res["passed"], res["checks"], flush=True)

    # Capacity (unlevered weights; levered rows divide by L).
    finals = {f"def_{k}": v["name"] for k, v in out["defensive"].items() if v} | {f"gro_{k}": v for k, v in g_final.items()}
    cap_books = {k: blend((1.0, lab.cands[n]["w"])) for k, n in finals.items()}
    excess = {k: (lab.frame_stats(n, "s6_exact")["cagr"] - 0.0161) / lab.cands[n]["L"] for k, n in finals.items()}
    cap = ev.rd.capacity(cap_books, lab.data, excess)
    for k, n in finals.items():
        rec = cap.get(k)
        val = rec["routes"]["worked+blocks"]["recommended"] if rec else None
        L = lab.cands[n]["L"]
        tgt = out["defensive"][k[4:]] if k.startswith("def_") else out["growth"][k[4:]]
        tgt["capacity"] = None if val is None else float(val) / L
        tgt["capacity_top"] = bool(val is not None and float(val) >= CAP_TOP)
    (fp.STUDY / "report" / "a6.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("a6_finished", slots={k: (v["name"] if v else None) for k, v in out["defensive"].items()}, growth=g_final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
