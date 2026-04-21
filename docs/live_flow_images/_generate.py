"""Generate PNG diagrams for LIVE_FLOW_SQL_SIMPLIFICATION_REVIEW.md."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).parent

# ---------- palette ----------
GREEN  = "#d4edda"
YELLOW = "#fff3cd"
BLUE   = "#d0e7ff"
ORANGE = "#ffe0b2"
RED    = "#f8d7da"
GREY   = "#e9ecef"
DARK   = "#343a40"
EDGE   = "#495057"


# ---------- drawing helpers ----------
def box(ax, xy, w, h, text, *, face=GREY, edge=EDGE, lw=1.4, fs=10, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs, fontweight=weight,
            color=DARK, zorder=3, wrap=True)


def diamond(ax, xy, w, h, text, *, face=YELLOW, edge=EDGE, lw=1.4, fs=9):
    x, y = xy
    cx, cy = x + w / 2, y + h / 2
    pts = [(cx, y + h), (x + w, cy), (cx, y), (x, cy)]
    poly = mpatches.Polygon(pts, closed=True,
                            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, color=DARK, zorder=3)


def arrow(ax, xy1, xy2, *, label=None, color=EDGE, lw=1.4, style="->", fs=8, curve=0.0):
    a = FancyArrowPatch(xy1, xy2,
                        arrowstyle=style, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={curve}",
                        mutation_scale=14, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
        ax.text(mx, my, label, fontsize=fs, color=DARK,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", ec="none", alpha=0.9), zorder=4)


def clean_axes(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ---------- 1. pie: keep/remove verdict ----------
def pie_verdict():
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sizes = [8, 4, 1, 1]
    labels = [
        "must_keep tables (8)",
        "candidate_remove tables (4)",
        "keep_if_daemon (1)",
        "useful_audit only (1)",
    ]
    colors = [GREEN, ORANGE, BLUE, YELLOW]
    total = sum(sizes)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f"{int(round(p * total / 100))}",
        startangle=90,
        wedgeprops=dict(edgecolor=EDGE, linewidth=1.2),
        textprops=dict(fontsize=11, color=DARK),
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Keep / Remove verdict across ~14 tables",
                 fontsize=13, fontweight="bold", color=DARK, pad=14)
    save(fig, "01_pie_verdict.png")


# ---------- 2. core formulas flow ----------
def core_formulas():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    clean_axes(ax, (0, 13), (0, 5.5))

    # inputs (left)
    box(ax, (0.1, 4.2), 2.3, 0.7, "net_liq_float\n(from broker)",        face=BLUE)
    box(ax, (0.1, 3.1), 2.3, 0.7, "pod_budget_fraction\n(from release)", face=BLUE)
    box(ax, (0.1, 2.0), 2.3, 0.7, "target_weight_float\n(from strategy)", face=BLUE)
    box(ax, (0.1, 0.9), 2.3, 0.7, "live_reference_price",                face=BLUE)
    box(ax, (0.1, -0.2), 2.3, 0.7, "broker_share\n(at submit)",          face=BLUE)

    # computed chain (middle/right)
    box(ax, (3.6, 3.65), 2.6, 0.7, "pod_budget_float",        face=YELLOW, weight="bold")
    box(ax, (6.9, 3.05), 2.6, 0.7, "target_notional_float",   face=YELLOW, weight="bold")
    box(ax, (6.9, 1.95), 2.6, 0.7, "target_share_float",      face=ORANGE, weight="bold")
    box(ax, (10.2, 1.35), 2.6, 0.7, "order_delta_share_float", face=GREEN, weight="bold")
    box(ax, (10.2, 0.25), 2.6, 0.7, "residual_share_float",    face=RED,   weight="bold")

    # post-exec input (bottom)
    box(ax, (6.9, -0.25), 2.6, 0.7, "broker_share\n(post-execution)",   face=BLUE)

    # arrows
    arrow(ax, (2.4, 4.55), (3.6, 4.0))   # net_liq -> pod_budget
    arrow(ax, (2.4, 3.45), (3.6, 3.9))   # budget_fraction -> pod_budget
    arrow(ax, (6.2, 4.0),  (6.9, 3.4))   # pod_budget -> target_notional
    arrow(ax, (2.4, 2.35), (6.9, 3.25))  # target_weight -> target_notional
    arrow(ax, (8.2, 3.05), (8.2, 2.65))  # target_notional -> target_share
    arrow(ax, (2.4, 1.25), (6.9, 2.15))  # live_price -> target_share
    arrow(ax, (9.5, 2.3),  (10.2, 1.7))  # target_share -> order_delta
    arrow(ax, (2.4, 0.15), (10.2, 1.55)) # broker_share submit -> order_delta
    arrow(ax, (9.5, 2.15), (10.2, 0.6))  # target_share -> residual
    arrow(ax, (9.5, 0.1),  (10.2, 0.4))  # broker post -> residual

    ax.set_title("Core formulas — the mathematical spine of the live layer",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)
    save(fig, "02_core_formulas.png")


# ---------- 3. end-to-end flow ----------
def end_to_end_flow():
    fig, ax = plt.subplots(figsize=(11, 10))
    clean_axes(ax, (0, 11), (0, 11))

    # column x positions
    xl, xc, xr = 1.0, 4.3, 7.6

    # nodes (bottom up roughly follows flow top-down)
    nodes = {
        "A":  (xc, 10.0, 2.8, 0.7, "LiveRelease YAML",                    YELLOW),
        "B":  (xc,  9.0, 2.8, 0.6, "live_release",                        BLUE),
        "C":  (xc,  8.0, 3.2, 0.6, "build_decision_plan_for_release()",  GREY),
        "D":  (xc,  7.0, 2.8, 0.6, "decision_plan",                       BLUE),
        "E":  (xc,  6.0, 2.8, 0.6, "build_vplans()",                      GREY),
        "F":  (xl - 0.2,  5.0, 3.0, 0.6, "broker_snapshot_cache",         BLUE),
        "G":  (xr + 0.2,  5.0, 3.0, 0.6, "live price snapshot",            BLUE),
        "H":  (xc,  4.0, 2.8, 0.6, "vplan + vplan_row",                   BLUE),
        "I":  (xc,  3.0, 3.2, 0.6, "submit_ready_vplans()",               GREY),
        "J":  (xl - 0.1, 2.0, 2.7, 0.6, "vplan_broker_order",             GREEN),
        "K":  (xc, 2.0, 2.8, 0.6, "vplan_broker_order_event\nvplan_broker_ack", ORANGE),
        "L":  (xr + 0.1, 2.0, 2.7, 0.6, "vplan_fill",                     GREEN),
        "M":  (xc,  1.0, 3.2, 0.6, "post_execution_reconcile()",          GREY),
        "N":  (xl - 0.1, 0.0, 3.0, 0.6, "vplan_reconciliation_snapshot",  BLUE),
        "O":  (xr + 0.1, 0.0, 2.7, 0.6, "pod_state",                      RED),
    }
    for k, (x, y, w, h, t, face) in nodes.items():
        box(ax, (x - w / 2, y), w, h, t, face=face, fs=9)

    def c(n): return (nodes[n][0], nodes[n][1] + nodes[n][3] / 2)
    def top(n): return (nodes[n][0], nodes[n][1] + nodes[n][3])
    def bot(n): return (nodes[n][0], nodes[n][1])

    arrow(ax, bot("A"), top("B"))
    arrow(ax, bot("B"), top("C"))
    arrow(ax, bot("C"), top("D"))
    arrow(ax, bot("D"), top("E"))
    arrow(ax, bot("E"), top("F"), curve=0.2)
    arrow(ax, bot("E"), top("G"), curve=-0.2)
    arrow(ax, bot("E"), top("H"))
    arrow(ax, bot("H"), top("I"))
    arrow(ax, bot("I"), top("J"), curve=0.2)
    arrow(ax, bot("I"), top("K"))
    arrow(ax, bot("I"), top("L"), curve=-0.2)
    arrow(ax, bot("J"), top("M"), curve=-0.1)
    arrow(ax, bot("L"), top("M"), curve=0.1)
    arrow(ax, (nodes["F"][0] + 1.3, nodes["F"][1] + 0.3), top("M"), curve=-0.4)
    arrow(ax, bot("M"), top("N"), curve=0.2)
    arrow(ax, bot("M"), top("O"), curve=-0.2)

    ax.set_title("End-to-end flow: manifest → decision → vplan → submit → reconcile → pod_state",
                 fontsize=12, fontweight="bold", color=DARK, y=0.98)
    save(fig, "03_end_to_end_flow.png")


# ---------- 4. decision tree ----------
def decision_tree():
    fig, ax = plt.subplots(figsize=(13, 10))
    clean_axes(ax, (0, 13), (0, 11))

    # top
    box(ax, (5.5, 10.0), 2.0, 0.6, "Cycle begins", face=GREY, weight="bold")

    # decision chain on the right side; outcomes fan out to left
    diamond(ax, (5.5, 8.8), 2.0, 0.9, "Snapshot\nready?")
    box(ax, (0.5, 8.9), 2.6, 0.7, "[WAIT]\nno DecisionPlan", face=YELLOW)

    diamond(ax, (5.5, 7.5), 2.0, 0.9, "Submission\nexpired?")
    box(ax, (0.5, 7.6), 2.6, 0.7, "[EXPIRED]\nDecisionPlan+VPlan=expired", face=RED)

    diamond(ax, (5.5, 6.2), 2.0, 0.9, "Broker ready?\nAccount OK?")
    box(ax, (0.5, 6.3), 2.6, 0.7, "[BLOCKED]", face=RED)

    diamond(ax, (5.5, 4.9), 2.0, 0.9, "net_liq > 0?")
    box(ax, (0.5, 5.0), 2.6, 0.7, "[BLOCKED]\nnon_positive_net_liq", face=RED)

    diamond(ax, (5.5, 3.6), 2.0, 0.9, "Live prices\navailable?")
    box(ax, (0.5, 3.7), 2.6, 0.7, "[BLOCKED]\nmissing_live_price", face=RED)

    diamond(ax, (5.5, 2.3), 2.0, 0.9, "Auto-submit\nenabled?")
    box(ax, (0.5, 2.4), 2.6, 0.7, "[REVIEW]\nVPlan=ready / review_vplan", face=YELLOW)

    box(ax, (5.8, 1.1), 1.6, 0.6, "[SEND] submitted", face=BLUE, weight="bold")

    diamond(ax, (8.8, 3.0), 2.0, 0.9, "Reconcile\npasses?")
    box(ax, (10.2, 1.1), 2.6, 0.7, "[OK] completed\nPodState updated", face=GREEN, weight="bold")

    diamond(ax, (8.8, 4.6), 2.0, 0.9, "Broker\nnon-terminal?")
    box(ax, (10.2, 4.7), 2.6, 0.7, "[RETRY] keep reconciling", face=YELLOW)
    box(ax, (10.2, 5.8), 2.6, 0.7, "[WARN] parked\nmanual_review", face=ORANGE)

    # arrows down the diamond chain
    arrow(ax, (6.5, 10.0), (6.5, 9.7))
    # Q1
    arrow(ax, (5.5, 9.25), (3.1, 9.25), label="no")
    arrow(ax, (6.5, 8.8),  (6.5, 8.4),  label="yes")
    # Q2
    arrow(ax, (7.5, 7.95), (3.1, 7.95), label="yes")
    arrow(ax, (6.5, 7.5),  (6.5, 7.1),  label="no")
    # Q3
    arrow(ax, (5.5, 6.65), (3.1, 6.65), label="no")
    arrow(ax, (6.5, 6.2),  (6.5, 5.8),  label="yes")
    # Q4
    arrow(ax, (5.5, 5.35), (3.1, 5.35), label="no")
    arrow(ax, (6.5, 4.9),  (6.5, 4.5),  label="yes")
    # Q5
    arrow(ax, (5.5, 4.05), (3.1, 4.05), label="no")
    arrow(ax, (6.5, 3.6),  (6.5, 3.2),  label="yes")
    # Q6
    arrow(ax, (5.5, 2.75), (3.1, 2.75), label="no")
    arrow(ax, (6.5, 2.3),  (6.5, 1.7),  label="yes")
    # submitted -> reconcile?
    arrow(ax, (7.4, 1.4),  (8.8, 3.0))
    # reconcile
    arrow(ax, (9.8, 3.0),  (10.2, 1.5), label="|res|≤eps")
    arrow(ax, (8.8, 3.45), (8.8, 4.15), label="no")
    # non-terminal?
    arrow(ax, (9.8, 5.05), (10.2, 5.05), label="PendingSubmit")
    arrow(ax, (9.8, 4.6),  (10.2, 6.15), label="Cancelled/Rejected")

    ax.set_title("What changes if X happens — decision tree through the cycle",
                 fontsize=12, fontweight="bold", color=DARK, y=0.99)
    save(fig, "04_decision_tree.png")


# ---------- 5. state machines ----------
def state_machine_decision_plan():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    clean_axes(ax, (0, 11), (0, 4.5))

    states = {
        "planned":     (1.0, 2.5),
        "vplan_ready": (3.5, 2.5),
        "submitted":   (6.0, 2.5),
        "completed":   (8.5, 2.5),
        "expired":     (3.5, 0.6),
        "blocked":     (1.0, 0.6),
    }
    face = {
        "planned": BLUE, "vplan_ready": BLUE,
        "submitted": YELLOW, "completed": GREEN,
        "expired": RED, "blocked": RED,
    }
    for name, (x, y) in states.items():
        box(ax, (x, y), 2.0, 0.7, name, face=face[name], weight="bold", fs=11)

    def pt(n, side="r"):
        x, y = states[n]
        if side == "r": return (x + 2.0, y + 0.35)
        if side == "l": return (x, y + 0.35)
        if side == "t": return (x + 1.0, y + 0.7)
        return (x + 1.0, y)

    arrow(ax, pt("planned", "r"),     pt("vplan_ready", "l"), label="VPlan built")
    arrow(ax, pt("vplan_ready", "r"), pt("submitted",   "l"), label="submit OK")
    arrow(ax, pt("submitted",   "r"), pt("completed",   "l"), label="reconcile OK")
    arrow(ax, pt("planned", "b"),     pt("blocked",    "t"),  label="broker/price")
    arrow(ax, pt("planned", "b"),     pt("expired",    "t"),  label="window", curve=-0.3)
    arrow(ax, pt("vplan_ready", "b"), pt("expired",    "t"),  label="window")

    ax.set_title("DecisionPlan.status_str", fontsize=13, fontweight="bold", color=DARK, y=0.98)
    save(fig, "05_state_decision_plan.png")


def state_machine_vplan():
    fig, ax = plt.subplots(figsize=(12, 5))
    clean_axes(ax, (0, 12), (0, 5))

    states = {
        "ready":        (0.5, 3.0),
        "submitting":   (3.0, 3.0),
        "submitted":    (5.8, 3.0),
        "completed":    (8.8, 3.8),
        "parked":       (8.8, 1.9),
        "expired":      (3.0, 0.6),
        "blocked":      (0.5, 0.6),
    }
    face = {
        "ready": BLUE, "submitting": YELLOW,
        "submitted": YELLOW, "completed": GREEN,
        "parked": ORANGE, "expired": RED, "blocked": RED,
    }
    label_map = {"parked": "parked_manual_review\n(derived, not persisted)"}
    for name, (x, y) in states.items():
        w = 2.4 if name != "parked" else 2.8
        box(ax, (x, y), w, 0.7, label_map.get(name, name),
            face=face[name], weight="bold", fs=10)

    def r(n): return (states[n][0] + (2.8 if n == "parked" else 2.4), states[n][1] + 0.35)
    def l(n): return (states[n][0], states[n][1] + 0.35)
    def t(n): return (states[n][0] + 1.2, states[n][1] + 0.7)
    def b(n): return (states[n][0] + 1.2, states[n][1])

    arrow(ax, r("ready"),      l("submitting"), label="claim")
    arrow(ax, r("submitting"), l("submitted"),  label="broker OK")
    arrow(ax, r("submitted"),  l("completed"),  label="residual≤eps")
    arrow(ax, r("submitted"),  l("parked"),     label="terminal+residual")
    arrow(ax, b("ready"),      t("blocked"),    label="broker/price")
    arrow(ax, b("ready"),      t("expired"),    label="window")

    ax.set_title("VPlan.status_str", fontsize=13, fontweight="bold", color=DARK, y=0.98)
    save(fig, "06_state_vplan.png")


# ---------- 7. ER-style diagram ----------
def er_diagram():
    fig, ax = plt.subplots(figsize=(13, 9))
    clean_axes(ax, (0, 13), (0, 9))

    tables = {
        "live_release":                   (0.5, 7.8, GREEN),
        "pod_state":                      (0.5, 6.4, GREEN),
        "broker_snapshot_cache":          (0.5, 5.0, GREEN),
        "scheduler_lease":                (0.5, 3.6, BLUE),
        "job_run":                        (0.5, 2.2, YELLOW),
        "session_open_price":             (0.5, 0.7, ORANGE),

        "decision_plan":                  (5.0, 7.8, GREEN),
        "vplan":                          (5.0, 5.8, GREEN),
        "vplan_row":                      (5.0, 3.8, ORANGE),

        "vplan_broker_order":             (9.5, 7.8, GREEN),
        "vplan_broker_order_event":       (9.5, 6.4, ORANGE),
        "vplan_broker_ack":               (9.5, 5.0, ORANGE),
        "vplan_fill":                     (9.5, 3.6, GREEN),
        "vplan_reconciliation_snapshot":  (9.5, 2.0, GREEN),
    }
    for name, (x, y, face) in tables.items():
        box(ax, (x, y), 3.0, 0.8, name, face=face, fs=10, weight="bold")

    def r(n): x, y, _ = tables[n]; return (x + 3.0, y + 0.4)
    def l(n): x, y, _ = tables[n]; return (x, y + 0.4)
    def b(n): x, y, _ = tables[n]; return (x + 1.5, y)
    def t(n): x, y, _ = tables[n]; return (x + 1.5, y + 0.8)

    arrow(ax, r("live_release"),         l("decision_plan"),  label="release_id")
    arrow(ax, r("decision_plan"),        l("vplan"), curve=0.1, label="1:1")
    arrow(ax, b("vplan"),                t("vplan_row"),      label="1:N")
    arrow(ax, r("vplan"),                l("vplan_broker_order"), curve=0.3, label="1:N")
    arrow(ax, r("vplan"),                l("vplan_broker_order_event"), curve=0.1, label="1:N")
    arrow(ax, r("vplan"),                l("vplan_broker_ack"), curve=-0.05, label="1:N")
    arrow(ax, r("vplan"),                l("vplan_fill"), curve=-0.2, label="1:N")
    arrow(ax, r("vplan"),                l("vplan_reconciliation_snapshot"), curve=-0.35, label="1:N")
    arrow(ax, r("broker_snapshot_cache"), l("vplan"), label="account_route", curve=-0.1)
    arrow(ax, r("pod_state"),            l("decision_plan"), label="pod_id", curve=-0.2)

    # legend
    handles = [
        mpatches.Patch(facecolor=GREEN,  edgecolor=EDGE, label="must_keep"),
        mpatches.Patch(facecolor=YELLOW, edgecolor=EDGE, label="useful_audit"),
        mpatches.Patch(facecolor=BLUE,   edgecolor=EDGE, label="keep_if_daemon"),
        mpatches.Patch(facecolor=ORANGE, edgecolor=EDGE, label="candidate_remove"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10,
              frameon=True, title="verdict")

    ax.set_title("Table relationships and simplification verdict",
                 fontsize=13, fontweight="bold", color=DARK, y=0.99)
    save(fig, "07_er_diagram.png")


# ---------- 8. core questions -> tables ----------
def core_questions():
    fig, ax = plt.subplots(figsize=(12, 7))
    clean_axes(ax, (0, 12), (0, 7))

    questions = [
        ("what was supposed\nto happen?",     6.0, "live_release +\ndecision_plan"),
        ("broker account\nstate?",            5.0, "broker_snapshot_cache"),
        ("what did we send?",                 4.0, "vplan +\nvplan_broker_order"),
        ("what filled?",                      3.0, "vplan_fill"),
        ("did we end\nat target?",            2.0, "vplan_reconciliation_snapshot"),
        ("continue from\nwhat state?",        1.0, "pod_state"),
        ("avoid double-submit?",              0.0, "submission_key_str"),
    ]
    for q, y, tbl in questions:
        box(ax, (0.5, y + 0.05), 3.4, 0.8, q,    face=YELLOW, fs=10)
        box(ax, (7.8, y + 0.05), 3.7, 0.8, tbl,  face=GREEN,  fs=10, weight="bold")
        arrow(ax, (3.9, y + 0.45), (7.8, y + 0.45))

    ax.set_title("Core live questions → the tables that answer them",
                 fontsize=13, fontweight="bold", color=DARK, y=0.99)
    save(fig, "08_core_questions.png")


# ---------- 9. simplification roadmap ----------
def roadmap():
    fig, ax = plt.subplots(figsize=(13, 8))
    clean_axes(ax, (0, 13), (0, 8))

    # phase headers
    box(ax, (0.3, 6.6), 4.0, 0.9,
        "Phase 1\nNo-semantics-change cleanup",
        face=YELLOW, weight="bold", fs=11)
    box(ax, (4.5, 6.6), 4.0, 0.9,
        "Phase 2\nOperator simplification",
        face=BLUE, weight="bold", fs=11)
    box(ax, (8.7, 6.6), 4.0, 0.9,
        "Phase 3\nInterface simplification",
        face=GREEN, weight="bold", fs=11)

    p1 = [
        "remove legacy alias fields\n(after explicit migration)",
        "decide: total_value vs net_liq",
        "one authoritative schema\nmigration path",
        "stop treating checked-in sqlite\nas schema contract",
    ]
    p2 = [
        "drop vplan_broker_order_event?",
        "drop vplan_broker_ack?",
        "drop session_open_price?",
        "derive vplan_row on read\ninstead of storing",
    ]
    p3 = [
        "one status view",
        "one vplan detail view",
        "one execution-exception view",
        "one manual submit path",
        "one reconcile path",
    ]

    def column(items, x0, face):
        y = 5.6
        for it in items:
            box(ax, (x0, y), 4.0, 0.9, it, face=face, fs=9)
            y -= 1.1

    column(p1, 0.3, YELLOW)
    column(p2, 4.5, BLUE)
    column(p3, 8.7, GREEN)

    # horizontal flow between phases
    arrow(ax, (4.3, 7.05), (4.5, 7.05), lw=2.2)
    arrow(ax, (8.5, 7.05), (8.7, 7.05), lw=2.2)

    ax.set_title("Safe simplification roadmap — in order",
                 fontsize=13, fontweight="bold", color=DARK, y=0.99)
    save(fig, "09_roadmap.png")


# ---------- 10. sequence diagram ----------
def sequence_diagram():
    fig, ax = plt.subplots(figsize=(14, 9))
    clean_axes(ax, (0, 14), (0, 10))

    actors = ["Runner", "Strategy\nHost", "Broker\nAdapter", "Execution\nEngine", "state_store_v2", "IBKR"]
    positions = [1.0, 3.3, 5.6, 7.9, 10.2, 12.5]
    colors = [BLUE, YELLOW, ORANGE, GREEN, GREY, RED]

    for name, x, c in zip(actors, positions, colors):
        box(ax, (x - 0.9, 9.0), 1.8, 0.7, name, face=c, weight="bold", fs=10)
        # lifeline
        ax.plot([x, x], [0.2, 9.0], color=EDGE, lw=0.8, linestyle="--", zorder=0)

    # messages: (from, to, y, label)
    msgs = [
        (0, 4, 8.4, "load live_release"),
        (0, 1, 7.9, "build DecisionPlan(release, snap, pod_state)"),
        (1, 4, 7.4, "persist decision_plan"),
        (0, 2, 6.9, "get broker_snapshot + live prices"),
        (2, 4, 6.4, "upsert broker_snapshot_cache"),
        (0, 3, 5.9, "build VPlan(decision, snap, prices)"),
        (3, 4, 5.4, "persist vplan + vplan_row"),
        (0, 3, 4.9, "submit_ready_vplans()"),
        (3, 5, 4.4, "place orders"),
        (5, 3, 3.9, "ACK + events + fills"),
        (3, 4, 3.4, "vplan_broker_order / event / ack / fill"),
        (0, 3, 2.9, "post_execution_reconcile()"),
        (3, 4, 2.4, "vplan_reconciliation_snapshot"),
        (3, 4, 1.9, "update pod_state from broker truth"),
    ]
    for i, (a, b, y, lbl) in enumerate(msgs, 1):
        x1, x2 = positions[a], positions[b]
        arrow(ax, (x1, y), (x2, y), lw=1.3)
        mx = (x1 + x2) / 2
        ax.text(mx, y + 0.12, f"{i}. {lbl}", ha="center", va="bottom",
                fontsize=8.5, color=DARK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.95))

    ax.set_title("Sequence view — who calls whom during a cycle",
                 fontsize=13, fontweight="bold", color=DARK, y=0.99)
    save(fig, "10_sequence.png")


# ---------- run all ----------
if __name__ == "__main__":
    pie_verdict()
    core_formulas()
    end_to_end_flow()
    decision_tree()
    state_machine_decision_plan()
    state_machine_vplan()
    er_diagram()
    core_questions()
    roadmap()
    sequence_diagram()
    print("\nAll diagrams written to:", OUT_DIR)
