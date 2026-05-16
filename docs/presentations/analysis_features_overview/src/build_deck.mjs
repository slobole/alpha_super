import fs from "node:fs/promises";
import path from "node:path";

const artifactToolUrl =
  "file:///C:/Users/User/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
} = await import(artifactToolUrl);

const repoRootPath = "C:/Users/User/Documents/workspace/alpha_super";
const deckRootPath = path.join(
  repoRootPath,
  "docs/presentations/analysis_features_overview",
);
const outputDirPath = path.join(deckRootPath, "output");
const previewDirPath = path.join(deckRootPath, "scratch/previews");
const outputPptxPath = path.join(outputDirPath, "analysis_features_overview.pptx");

await fs.mkdir(outputDirPath, { recursive: true });
await fs.mkdir(previewDirPath, { recursive: true });

const W = 1920;
const H = 1080;
const FONT = "Aptos";
const MONO = "Cascadia Mono";

const C = {
  bg: "#F6F2EA",
  paper: "#FFFDF8",
  ink: "#15181D",
  muted: "#636B74",
  faint: "#D8D0C2",
  black: "#11151B",
  teal: "#0F766E",
  blue: "#2563EB",
  amber: "#B7791F",
  red: "#B42318",
  purple: "#6D28D9",
  green: "#15803D",
};

const presentation = Presentation.create({
  slideSize: { width: W, height: H },
});

function t(value, options = {}) {
  return text(value, {
    name: options.name,
    width: options.width ?? fill,
    height: options.height ?? hug,
    columnSpan: options.columnSpan,
    rowSpan: options.rowSpan,
    style: {
      fontFamily: options.fontFamily ?? FONT,
      fontSize: options.fontSize ?? 26,
      bold: options.bold ?? false,
      color: options.color ?? C.ink,
      ...options.style,
    },
  });
}

function simpleSlide(kicker, title, subtitle, bodyChildren, options = {}) {
  const slide = presentation.slides.add();
  const bg = options.bg ?? C.bg;
  slide.compose(
    layers(
      { name: "slide", width: fill, height: fill },
      [
        shape({ name: "background", width: fill, height: fill, fill: bg }),
        column(
          {
            name: "root",
            width: fill,
            height: fill,
            padding: { x: 96, y: 64 },
            gap: options.gap ?? 30,
          },
          [
            column({ name: "title-stack", width: fill, height: hug, gap: 10 }, [
              t(kicker.toUpperCase(), {
                name: "kicker",
                width: wrap(780),
                fontSize: 17,
                bold: true,
                color: options.accent ?? C.teal,
                style: { letterSpacing: 1.1 },
              }),
              t(title, {
                name: "slide-title",
                width: wrap(options.titleWidth ?? 1320),
                fontSize: options.titleSize ?? 48,
                bold: true,
                color: C.ink,
                style: { lineSpacingMultiple: 0.95 },
              }),
              subtitle
                ? t(subtitle, {
                    name: "subtitle",
                    width: wrap(options.subtitleWidth ?? 1220),
                    fontSize: 23,
                    color: C.muted,
                  })
                : shape({ width: fixed(1), height: fixed(1), fill: bg }),
            ]),
            ...bodyChildren,
            row({ name: "footer", width: fill, height: hug, gap: 18, align: "center" }, [
              rule({ width: grow(1), stroke: C.faint, weight: 1 }),
              t(options.footer ?? "Source: current alpha_super source inspection, 2026-05-16.", {
                name: "source",
                width: wrap(940),
                fontSize: 12,
                color: C.muted,
                style: { italic: true, alignment: "right" },
              }),
            ]),
          ],
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

function command(commandText, noteText, color = C.black) {
  return panel(
    {
      name: "command",
      width: fill,
      height: hug,
      padding: { x: 26, y: 20 },
      fill: color,
      borderRadius: 8,
    },
    column({ width: fill, height: hug, gap: 8 }, [
      t(commandText, {
        fontFamily: MONO,
        fontSize: 19,
        color: "#FFFFFF",
      }),
      noteText
        ? t(noteText, {
            fontSize: 16,
            color: "#D6DEE8",
          })
        : shape({ width: fixed(1), height: fixed(1), fill: color }),
    ]),
  );
}

function labeledText(label, body, accent = C.teal) {
  return column({ width: fill, height: hug, gap: 7 }, [
    t(label.toUpperCase(), {
      fontSize: 15,
      bold: true,
      color: accent,
      style: { letterSpacing: 0.8 },
    }),
    t(body, {
      fontSize: 22,
      color: C.ink,
      width: fill,
    }),
  ]);
}

function metric(value, label, note, color) {
  return column({ width: fill, height: hug, gap: 5 }, [
    t(value, { fontSize: 42, bold: true, color }),
    t(label, { fontSize: 17, bold: true, color: C.ink }),
    t(note, { fontSize: 14, color: C.muted }),
  ]);
}

function featureCard(name, accent, oneLine, commandText, caution) {
  return panel(
    {
      width: fill,
      height: fixed(220),
      padding: { x: 24, y: 22 },
      fill: C.paper,
      line: { color: C.faint, width: 1 },
      borderRadius: 8,
    },
    column({ width: fill, height: hug, gap: 11 }, [
      row({ width: fill, height: hug, gap: 12, align: "center" }, [
        shape({ width: fixed(12), height: fixed(12), fill: accent, borderRadius: 6 }),
        t(name, { fontSize: 25, bold: true, color: C.ink }),
      ]),
      t(oneLine, { fontSize: 18, color: C.muted, width: fill }),
      t(commandText, {
        fontFamily: MONO,
        fontSize: 14,
        color: C.black,
        width: fill,
      }),
      t(caution, { fontSize: 15, color: accent, width: fill, bold: true }),
    ]),
  );
}

function compactTable(headers, rows, widths, options = {}) {
  const children = [];
  headers.forEach((header) => {
    children.push(
      panel(
        {
          width: fill,
          height: fixed(options.headerHeight ?? 44),
          padding: { x: 14, y: 10 },
          fill: options.headerFill ?? C.black,
        },
        t(header, {
          fontSize: options.headerFontSize ?? 16,
          bold: true,
          color: "#FFFFFF",
        }),
      ),
    );
  });
  rows.forEach((rowValues, rowIndex) => {
    rowValues.forEach((cellValue, cellIndex) => {
      children.push(
        panel(
          {
            width: fill,
            height: fixed(options.rowHeight ?? 58),
            padding: { x: 14, y: 10 },
            fill: rowIndex % 2 === 0 ? C.paper : "#FAF7EF",
          },
          t(String(cellValue), {
            fontSize: options.bodyFontSize ?? 16,
            bold: cellIndex === 0,
            color: cellIndex === 0 ? C.ink : C.muted,
          }),
        ),
      );
    });
  });
  return grid(
    {
      width: fill,
      height: hug,
      columns: widths.map((width) => fr(width)),
      columnGap: 2,
      rowGap: 2,
    },
    children,
  );
}

function coverSlide() {
  const slide = presentation.slides.add();
  slide.compose(
    layers(
      { width: fill, height: fill },
      [
        shape({ width: fill, height: fill, fill: C.black }),
        shape({ width: fixed(470), height: fill, fill: "#1C282B" }),
        column(
          {
            width: fill,
            height: fill,
            padding: { x: 92, y: 76 },
            gap: 40,
          },
          [
            row({ width: fill, height: hug, justify: "between", align: "center" }, [
              t("alpha_super", { width: wrap(420), fontSize: 24, bold: true, color: "#DCE8E4" }),
              t("Analysis feature map", {
                width: wrap(460),
                fontSize: 18,
                color: "#B7C5C1",
                style: { alignment: "right" },
              }),
            ]),
            row({ width: fill, height: fixed(770), gap: 70, align: "center" }, [
              column({ width: fixed(410), height: hug, gap: 18 }, [
                t("Currently wired", { fontSize: 28, bold: true, color: "#E9F3EF" }),
                featurePill("Vanilla Backtest", C.teal),
                featurePill("Portfolio", C.blue),
                featurePill("Timing", C.amber),
                featurePill("Stress", C.red),
                featurePill("Friction", C.purple),
              ]),
              column({ width: grow(1), height: hug, gap: 26 }, [
                t("What analysis features exist today?", {
                  width: wrap(980),
                  fontSize: 72,
                  bold: true,
                  color: "#FFFFFF",
                  style: { lineSpacingMultiple: 0.9 },
                }),
                t(
                  "A clean operator guide: what each feature answers, how to run it, what example output looks like, and where the realism boundary sits.",
                  {
                    width: wrap(960),
                    fontSize: 27,
                    color: "#D5DFE7",
                  },
                ),
                rule({ width: fixed(390), stroke: C.teal, weight: 5 }),
                t("Prepared from current repo source and saved artifacts.", {
                  width: wrap(760),
                  fontSize: 18,
                  color: "#AAB8C3",
                }),
              ]),
            ]),
          ],
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
}

function featurePill(label, color) {
  return panel(
    {
      width: fixed(280),
      height: fixed(48),
      padding: { x: 18, y: 10 },
      fill: color,
      borderRadius: 8,
    },
    t(label, {
      fontSize: 18,
      bold: true,
      color: "#FFFFFF",
      style: { alignment: "center" },
    }),
  );
}

function inventorySlide() {
  simpleSlide(
    "TL;DR",
    "The stack has five practical analysis surfaces",
    "Use them as lenses, not as proof. Each one exposes a different failure mode.",
    [
      grid(
        {
          width: fill,
          height: fixed(490),
          columns: [fr(1), fr(1), fr(1)],
          columnGap: 24,
          rowGap: 24,
        },
        [
          featureCard(
            "Vanilla Backtest",
            C.teal,
            "Baseline engine contract and standard report.",
            "strategies/run_strategy.py",
            "Question: does it work under next-open fills?",
          ),
          featureCard(
            "Portfolio Analysis",
            C.blue,
            "Combines completed pods as independent sleeves.",
            "strategies/run_portfolio.py",
            "Question: do pods diversify or concentrate risk?",
          ),
          featureCard(
            "Execution Timing",
            C.amber,
            "Replays intent under alternate fill timing.",
            "execution_timing_analyzer.py",
            "Question: is the edge timing fragile?",
          ),
          featureCard(
            "Stress Replay",
            C.red,
            "Fresh-capital replay through named crises.",
            "strategies/run_crisis_replay.py",
            "Question: what breaks in known shocks?",
          ),
          featureCard(
            "FrictionAnalysis",
            C.purple,
            "Auction capacity and cost proxy overlay.",
            "strategies/run_friction_analysis.py",
            "Question: are orders believable at scale?",
          ),
          panel(
            {
              width: fill,
              height: fixed(220),
              padding: { x: 24, y: 22 },
              fill: "#EDE7DA",
              borderRadius: 8,
            },
            column({ width: fill, height: hug, gap: 12 }, [
              t("Shared artifact tree", { fontSize: 25, bold: true }),
              t("results/research/{entity}/{id}/{analysis}/{timestamp}", {
                fontFamily: MONO,
                fontSize: 16,
                color: C.black,
              }),
              t("Every saved run should be easy to inspect and compare.", {
                fontSize: 18,
                color: C.muted,
              }),
            ]),
          ),
        ],
      ),
      row({ width: fill, height: hug, gap: 34 }, [
        labeledText("One sentence rule", "Run Vanilla first. Add timing, stress, portfolio, or friction only when that specific question matters.", C.teal),
        labeledText("Important caveat", "None of these tools turns a backtest into live readiness by itself.", C.red),
      ]),
    ],
    { accent: C.teal, footer: "Core code paths: alpha/engine/backtester.py, portfolio.py, execution_timing.py, crisis.py, friction_analysis.py" },
  );
}

function vanillaSlide() {
  simpleSlide(
    "Vanilla Backtest",
    "Baseline: prior information, next-open fills",
    "This is the default truth path for daily strategies. It is the first run before any extra diagnostics.",
    [
      grid(
        { width: fill, height: fixed(600), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 24 }, [
            command(
              "uv run python strategies/run_strategy.py strategy_mr_dv2.py",
              "Works for modules exposing run_variant(...).",
              C.black,
            ),
            labeledText(
              "Timing contract",
              "decision_t = f(I_{t-1}); order is created inside iterate(); fill happens at current_bar Open.",
              C.teal,
            ),
            compactTable(
              ["Artifact", "Use"],
              [
                ["report.html", "human-readable report"],
                ["transactions.csv", "executed order ledger"],
                ["run_info.json", "identity and parameters"],
                ["summary.json", "small top-line metrics"],
              ],
              [0.9, 1.7],
              { rowHeight: 50, bodyFontSize: 16 },
            ),
          ]),
          column({ width: fill, height: hug, gap: 22 }, [
            t("Example saved summary", { fontSize: 27, bold: true }),
            grid({ width: fill, height: hug, columns: [fr(1), fr(1)], columnGap: 34, rowGap: 28 }, [
              metric("18.07%", "Annual return", "strategy_mo_atr_normalized_ndx_vxn_scaled", C.teal),
              metric("1.29", "Sharpe", "repo helper convention", C.blue),
              metric("-22.58%", "Max drawdown", "full sample", C.red),
              metric("840", "Trades", "closed trade count", C.purple),
            ]),
            labeledText(
              "Read this as",
              "A baseline research result under the repo's current engine assumptions, not a live execution claim.",
              C.amber,
            ),
          ]),
        ],
      ),
    ],
    { accent: C.teal, footer: "Code: alpha/engine/backtester.py, alpha/engine/report.py, strategies/run_strategy.py" },
  );
}

function portfolioSlide() {
  simpleSlide(
    "Portfolio Analysis",
    "Pods compound independently",
    "The portfolio report is a read-only aggregation of completed strategy runs, matching the pod model.",
    [
      grid(
        { width: fill, height: fixed(600), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 24 }, [
            command(
              "uv run python strategies/run_portfolio.py portfolios/current_multipod_all.yaml",
              "Loads pod pickles, validates metadata, and writes a portfolio report.",
              C.black,
            ),
            labeledText(
              "Portfolio math",
              "E_portfolio_t = sum_i E_pod_i,t. Do not replace this with daily weighted returns unless daily rebalancing is explicitly the model.",
              C.blue,
            ),
            compactTable(
              ["Diagnostic", "Why it matters"],
              [
                ["Correlation", "diversification check"],
                ["Tail contribution", "worst-day pod impact"],
                ["Pod sections", "standalone behavior inside allocation"],
              ],
              [0.9, 1.8],
              { rowHeight: 58, bodyFontSize: 17 },
            ),
          ]),
          column({ width: fill, height: hug, gap: 22 }, [
            t("Example: current_multipod_all", { fontSize: 27, bold: true }),
            grid({ width: fill, height: hug, columns: [fr(1), fr(1)], columnGap: 34, rowGap: 28 }, [
              metric("20.99%", "Annual return", "portfolio vanilla", C.teal),
              metric("1.46", "Sharpe", "combined curve", C.blue),
              metric("-13.51%", "Max drawdown", "portfolio drawdown", C.red),
              metric("17,685", "Trades", "pooled pod trades", C.purple),
            ]),
            labeledText(
              "Operator read",
              "This tells whether the book improves after combining sleeves, not whether one sleeve is live-ready.",
              C.amber,
            ),
          ]),
        ],
      ),
    ],
    { accent: C.blue, footer: "Code: alpha/engine/portfolio.py, alpha/engine/report.py, strategies/run_portfolio.py" },
  );
}

function timingSlide() {
  simpleSlide(
    "ExecutionTimingAnalyzer",
    "Timing robustness, not a new engine contract",
    "It replays the same order intent under alternate entry and exit fill rules.",
    [
      grid(
        { width: fill, height: fixed(600), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 24 }, [
            command(
              "uv run python execution_timing_analyzer.py strategies/dv2/strategy_mr_dv2.py",
              "The strategy must expose build_execution_timing_analysis_inputs().",
              C.black,
            ),
            labeledText(
              "Formula",
              "entry_fill = signal_bar_t + entry_lag at Open or Close. Exit uses the selected exit lag and price field.",
              C.amber,
            ),
            compactTable(
              ["Mode", "Risk read"],
              [
                ["same_open", "diagnostic for daily OHLC signals"],
                ["same_close_moc", "MOC only if known before cutoff"],
                ["next_open", "Vanilla daily default"],
                ["next_close", "close/MOC diagnostic"],
              ],
              [0.9, 1.8],
              { rowHeight: 50, bodyFontSize: 16, headerFill: C.amber },
            ),
          ]),
          column({ width: fill, height: hug, gap: 22 }, [
            t("Example timing matrix rows", { fontSize: 27, bold: true }),
            compactTable(
              ["Entry", "Exit", "Sharpe", "Risk"],
              [
                ["T+1 Open", "T+1 Open", "1.19", "Clean"],
                ["T Close", "T Close", "1.21", "Biased MOC"],
                ["T+1 Open", "T+1 Close", "1.13", "MOC Assumption"],
                ["T+1 Close", "T+1 Open", "1.20", "MOC Assumption"],
              ],
              [1.0, 1.0, 0.6, 1.1],
              { rowHeight: 52, bodyFontSize: 16, headerFill: C.amber },
            ),
            labeledText(
              "Saved outputs",
              "execution_timing_metrics.csv plus return, CVaR, Sharpe, and drawdown matrices.",
              C.blue,
            ),
          ]),
        ],
      ),
    ],
    { accent: C.amber, footer: "Code: alpha/engine/execution_timing.py, execution_timing_analyzer.py" },
  );
}

function stressSlide() {
  simpleSlide(
    "Stress / Crisis Replay",
    "Fresh-capital replay through named shocks",
    "The analyzer restricts the execution calendar to each crisis while preserving pre-crisis history for causal signals.",
    [
      grid(
        { width: fill, height: fixed(600), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 24 }, [
            command(
              "uv run python strategies/run_crisis_replay.py strategy_mr_dv2",
              "Supported keys are listed in alpha.engine.crisis.",
              C.black,
            ),
            labeledText(
              "Formula",
              "For crisis c: R_c = V_end / V_0 - 1. Each window starts from fresh capital.",
              C.red,
            ),
            compactTable(
              ["Window", "Dates"],
              [
                ["gfc_2008", "2008-09-15 to 2009-03-09"],
                ["covid_crash", "2020-02-20 to 2020-04-07"],
                ["inflation_bear_2022", "2022-01-03 to 2022-10-12"],
                ["trump_tariffs_2025", "2025-02-01 to 2025-04-30"],
              ],
              [1.0, 1.7],
              { rowHeight: 50, bodyFontSize: 16, headerFill: C.red },
            ),
          ]),
          column({ width: fill, height: hug, gap: 22 }, [
            t("Example: archived strategy_mr_dv2", { fontSize: 27, bold: true }),
            compactTable(
              ["Crisis", "Strategy", "Relative"],
              [
                ["gfc_2008", "+3.0%", "+46.3 pp"],
                ["covid_crash", "-13.1%", "+8.1 pp"],
                ["inflation_bear_2022", "-10.2%", "+15.3 pp"],
                ["trump_tariffs_2025", "-4.4%", "+2.7 pp"],
              ],
              [1.25, 0.8, 0.9],
              { rowHeight: 52, bodyFontSize: 17, headerFill: C.red },
            ),
            labeledText(
              "Caveat",
              "Historical crises are useful samples. They are not complete future scenario coverage.",
              C.amber,
            ),
          ]),
        ],
      ),
    ],
    { accent: C.red, footer: "Code: alpha/engine/crisis.py, alpha/engine/report.py, strategies/run_crisis_replay.py" },
  );
}

function frictionSlide() {
  simpleSlide(
    "FrictionAnalysis",
    "Order realism overlay for auction capacity and cost",
    "It reads completed orders and pricing data. It does not change fills, PnL, sizing, or Vanilla reports.",
    [
      grid(
        { width: fill, height: fixed(600), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 24 }, [
            command(
              "uv run python strategies/run_friction_analysis.py dv2/strategy_mr_dv2.py",
              "The strategy must expose run_friction_analysis(...).",
              C.black,
            ),
            labeledText(
              "Core calculation",
              "auction_proxy_t = lagged 20-day median dollar ADV * auction_fraction. participation_t = abs(order_notional_t) / auction_proxy_t.",
              C.purple,
            ),
            compactTable(
              ["Policy", "Assumption"],
              [
                ["MOO", "2% of lagged dollar ADV proxy"],
                ["MOC", "10% of lagged dollar ADV proxy"],
                ["Default cost", "2.5 bps plus commission model"],
                ["Impact", "adverse proxy, not broker truth"],
              ],
              [0.75, 1.8],
              { rowHeight: 50, bodyFontSize: 16, headerFill: C.purple },
            ),
          ]),
          column({ width: fill, height: hug, gap: 22 }, [
            t("Example: NDX VXN scaled friction run", { fontSize: 27, bold: true }),
            grid({ width: fill, height: hug, columns: [fr(1), fr(1)], columnGap: 34, rowGap: 28 }, [
              metric("Watch", "Friction verdict", "auction verdict: Stressed", C.amber),
              metric("3,126", "Orders", "$303.7M assessed", C.blue),
              metric("8.64%", "Auction p95", "participation proxy", C.red),
              metric("-0.12 pp", "Return impact", "18.07% to 17.95%", C.purple),
            ]),
            labeledText(
              "Boundary",
              "No rescale, no retrade, no fill changes. This is a diagnostic overlay only.",
              C.red,
            ),
          ]),
        ],
      ),
    ],
    { accent: C.purple, footer: "Code: alpha/engine/friction_analysis.py, strategies/run_friction_analysis.py" },
  );
}

function artifactsSlide() {
  simpleSlide(
    "Artifact Contract",
    "Saved runs have a predictable folder shape",
    "The goal is simple inspection: know what entity ran, what analysis type it used, and which files carry the evidence.",
    [
      grid(
        { width: fill, height: fixed(590), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 24 }, [
            command(
              "results/research/{entity_type}/{entity_id}/{analysis_type}/{timestamp}/",
              "Example: results/research/strategy/strategy_mr_dv2/friction_analysis/...",
              C.black,
            ),
            compactTable(
              ["Analysis type", "Entity"],
              [
                ["vanilla_backtest", "strategy or portfolio"],
                ["execution_timing_analyzer", "strategy"],
                ["stress_analysis", "strategy"],
                ["friction_analysis", "strategy"],
              ],
              [1.25, 1.3],
              { rowHeight: 54, bodyFontSize: 17 },
            ),
          ]),
          column({ width: fill, height: hug, gap: 24 }, [
            t("Minimum files", { fontSize: 27, bold: true }),
            compactTable(
              ["File", "Reason"],
              [
                ["run_info.json", "identity and parameters"],
                ["summary.json", "small top-line metrics"],
                ["report.html", "human-readable report"],
                ["*.csv", "analysis-specific evidence"],
              ],
              [0.9, 1.7],
              { rowHeight: 54, bodyFontSize: 17 },
            ),
            labeledText(
              "Why it helps",
              "A future review can tell whether a number came from Vanilla, timing, stress, portfolio, or friction work.",
              C.teal,
            ),
          ]),
        ],
      ),
    ],
    { accent: C.teal, footer: "Code: alpha/engine/report.py, scripts/archive_research_results.py" },
  );
}

function runbookSlide() {
  simpleSlide(
    "Runbook",
    "Pick the analyzer by the question",
    "The question determines the tool. The tool determines the caveat.",
    [
      compactTable(
        ["Question", "Use", "Main thing to inspect"],
        [
          ["Does it work under the house contract?", "Vanilla Backtest", "summary, report.html, transactions.csv"],
          ["Is performance timing fragile?", "ExecutionTimingAnalyzer", "risk labels and timing matrices"],
          ["Does it survive known shocks?", "Stress Replay", "crisis_metrics.csv and paths"],
          ["Do pods diversify?", "Portfolio Analysis", "correlation and tail contribution"],
          ["Are orders believable?", "FrictionAnalysis", "verdict, p95 participation, scale limits"],
        ],
        [1.5, 1.0, 1.55],
        { rowHeight: 66, bodyFontSize: 17 },
      ),
      grid(
        { width: fill, height: fixed(160), columns: [fr(1), fr(1), fr(1)], columnGap: 30 },
        [
          labeledText("Green", "Use diagnostics before promotion.", C.green),
          labeledText("Yellow", "Rerun after data, cost, or engine changes.", C.amber),
          labeledText("Red", "Do not turn biased cells into live claims.", C.red),
        ],
      ),
    ],
    { accent: C.teal, footer: "Repo doctrine: QUANT_PHILOSOPHY.md and ASSUMPTIONS_AND_GAPS.md" },
  );
}

function bottomLineSlide() {
  simpleSlide(
    "Bottom Line",
    "The tools exist. The discipline is using the right one.",
    "The stack already covers returns, timing, stress, portfolio composition, and execution friction. Each diagnostic must keep its assumption boundary visible.",
    [
      grid(
        { width: fill, height: fixed(610), columns: [fr(1), fr(1)], columnGap: 58 },
        [
          column({ width: fill, height: hug, gap: 28 }, [
            labeledText(
              "Mature surfaces",
              "Vanilla reports, portfolio aggregation, FrictionAnalysis, and the artifact tree are directly wired.",
              C.green,
            ),
            labeledText(
              "Diagnostic surfaces",
              "Timing and crisis replay are powerful lenses, but they must not silently change execution semantics.",
              C.amber,
            ),
            labeledText(
              "Before live claims",
              "Check data timing, cutoff, order type, auction participation, partial fills, broker constraints, and reconciliation.",
              C.red,
            ),
          ]),
          column({ width: fill, height: hug, gap: 22 }, [
            t("Trust checklist", { fontSize: 30, bold: true }),
            compactTable(
              ["Before trusting a result", "Check"],
              [
                ["Data known at decision time?", "Lookahead boundary"],
                ["Engine timing preserved?", "Vanilla vs timing"],
                ["Stress reviewed?", "Crisis replay"],
                ["Pod math realistic?", "Independent compounding"],
                ["Execution scale believable?", "FrictionAnalysis"],
              ],
              [1.3, 1.0],
              { rowHeight: 54, bodyFontSize: 17 },
            ),
          ]),
        ],
      ),
    ],
    { accent: C.teal, footer: "This is a feature map, not a strategy recommendation." },
  );
}

coverSlide();
inventorySlide();
vanillaSlide();
portfolioSlide();
timingSlide();
stressSlide();
frictionSlide();
artifactsSlide();
runbookSlide();
bottomLineSlide();

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save(outputPptxPath);

const previewPathList = [];
for (let slideIndex = 0; slideIndex < presentation.slides.count; slideIndex += 1) {
  const slide = presentation.slides.getItem(slideIndex);
  const pngBlob = await slide.export({ format: "png" });
  const previewPath = path.join(
    previewDirPath,
    `slide_${String(slideIndex + 1).padStart(2, "0")}.png`,
  );
  await fs.writeFile(previewPath, Buffer.from(await pngBlob.arrayBuffer()));
  previewPathList.push(previewPath);
}

console.log(
  JSON.stringify(
    {
      pptxPath: outputPptxPath,
      previewDirPath,
      previewPathList,
      slideCount: presentation.slides.count,
    },
    null,
    2,
  ),
);
