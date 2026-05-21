import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main";
import type { DashboardSummary, PodDetail, PodRow } from "./types";

const lifecycleStepList = [
  { step_key_str: "db", label_str: "DB", status_str: "complete", severity_str: "green" },
  { step_key_str: "decision", label_str: "Decision", status_str: "complete", severity_str: "green" },
  { step_key_str: "vplan", label_str: "VPlan", status_str: "submitted", severity_str: "yellow" },
  { step_key_str: "ack", label_str: "ACK", status_str: "complete", severity_str: "green", detail_str: "rows=1" },
  { step_key_str: "fill", label_str: "Fill", status_str: "recorded", severity_str: "green", detail_str: "fills=1" },
  { step_key_str: "reconcile", label_str: "Reconcile", status_str: "pending", severity_str: "yellow" },
  { step_key_str: "eod", label_str: "EOD", status_str: "waiting", severity_str: "gray" },
  { step_key_str: "diff", label_str: "DIFF", status_str: "not_run", severity_str: "gray" }
];

const livePodRow: PodRow = {
  pod_id_str: "pod_live_01",
  mode_str: "live",
  account_route_str: "U123",
  strategy_import_str: "strategies.example",
  health_str: "yellow",
  next_action_str: "post_execution_reconcile",
  reason_code_str: "waiting_for_post_execution_reconcile",
  equity_float: 100000,
  cash_float: 50000,
  position_count_int: 2,
  latest_event_timestamp_str: "2026-05-20T12:00:00Z",
  latest_decision_plan_status_str: "complete",
  latest_decision_plan_submission_timestamp_str: "2026-05-20T11:55:00Z",
  latest_decision_plan_target_execution_timestamp_str: "2026-05-21T13:30:00Z",
  latest_vplan_status_str: "submitted",
  latest_vplan_id_int: 7,
  latest_vplan_submission_timestamp_str: "2026-05-20T12:00:00Z",
  latest_vplan_target_execution_timestamp_str: "2026-05-21T13:30:00Z",
  latest_submit_ack_status_str: "complete",
  latest_reconciliation_status_str: "pending",
  latest_reconciliation_timestamp_str: "2026-05-20T12:05:00Z",
  latest_diff_status_str: "green",
  latest_diff_open_issue_count_int: 0,
  broker_ack_count_int: 1,
  broker_order_count_int: 1,
  fill_count_int: 1,
  missing_ack_count_int: 0,
  eod_snapshot_dict: { status_str: "waiting", severity_str: "gray" },
  lifecycle_step_dict_list: lifecycleStepList,
  required_action_dict: {
    label_str: "Waiting reconcile",
    severity_str: "yellow",
    detail_str: "Broker fills exist; reconcile is pending."
  },
  debug_summary_dict: {
    severity_str: "yellow",
    verdict_label_str: "Waiting reconcile",
    primary_reason_str: "Broker fills exist; reconcile is pending."
  }
};

const paperPodRow: PodRow = {
  ...livePodRow,
  pod_id_str: "pod_paper_01",
  mode_str: "paper",
  account_route_str: "PAPER",
  health_str: "green",
  next_action_str: "No action",
  reason_code_str: "ok",
  required_action_dict: {
    label_str: "No action",
    severity_str: "green",
    detail_str: "POD is idle."
  },
  debug_summary_dict: {
    severity_str: "green",
    verdict_label_str: "Healthy",
    primary_reason_str: "POD is idle."
  }
};

const summaryPayload: DashboardSummary = {
  as_of_timestamp_str: "2026-05-20T12:00:00Z",
  pod_row_dict_list: [livePodRow, paperPodRow],
  alert_dict_list: [],
  alert_summary_dict: { total_count_int: 0 },
  mode_list: ["live", "paper"],
  combined_book_dict: {
    environment_dict_list: [
      {
        mode_str: "live",
        latest_equity_float: 100750,
        daily_pnl_float: 250,
        daily_pnl_pct_float: 0.00249,
        since_start_pnl_float: 750,
        since_start_pnl_pct_float: 0.0075,
        carry_forward_point_count_int: 3
      },
      {
        mode_str: "paper",
        latest_equity_float: 50750,
        daily_pnl_float: -50,
        daily_pnl_pct_float: -0.00098,
        since_start_pnl_float: 750,
        since_start_pnl_pct_float: 0.015,
        carry_forward_point_count_int: 1
      }
    ]
  }
};

const detailPayload: PodDetail = {
  pod_row_dict: livePodRow,
  required_action_dict: livePodRow.required_action_dict,
  lifecycle_step_dict_list: lifecycleStepList,
  event_dict_list: [],
  pod_pnl_dict: {
    status_str: "available",
    source_str: "pod_state_history.eod",
    point_count_int: 3,
    latest_market_date_str: "2026-05-20",
    latest_equity_float: 100750,
    daily_pnl_float: 250,
    daily_pnl_pct_float: 0.00249,
    since_start_pnl_float: 750,
    since_start_pnl_pct_float: 0.0075,
    equity_point_dict_list: [
      { market_date_str: "2026-05-18", equity_float: 100000, cash_float: 50000, since_start_pnl_float: 0 },
      { market_date_str: "2026-05-19", equity_float: 100500, cash_float: 50200, daily_pnl_float: 500, since_start_pnl_float: 500 },
      { market_date_str: "2026-05-20", equity_float: 100750, cash_float: 50350, daily_pnl_float: 250, since_start_pnl_float: 750 }
    ]
  },
  debug_story_dict: {
    timeline_event_dict_list: [
      {
        source_str: "Fill",
        label_str: "AAPL",
        status_str: "recorded",
        severity_str: "green",
        timestamp_str: "2026-05-20T12:03:00Z",
        detail_str: "shares=28, price=101"
      }
    ]
  },
  latest_decision_plan_dict: {
    status_str: "complete",
    decision_book_type_str: "full_target_weight_book",
    target_execution_timestamp_str: "2026-05-21T13:30:00Z",
    exit_asset_list: ["MSFT"],
    entry_target_weight_map_dict: { AAPL: 0.6 },
    full_target_weight_map_dict: { AAPL: 0.6, MSFT: 0 }
  },
  latest_vplan_dict: {
    vplan_id_int: 7,
    status_str: "submitted",
    submit_ack_status_str: "complete",
    ack_coverage_ratio_float: 1,
    vplan_row_dict_list: [
      {
        asset_str: "AAPL",
        current_share_float: 2,
        target_share_float: 30,
        order_delta_share_float: 28,
        live_reference_price_float: 100
      }
    ],
    broker_ack_row_dict_list: [
      {
        asset_str: "AAPL",
        ack_status_str: "broker_acked",
        ack_source_str: "fill",
        broker_response_ack_bool: true,
        response_timestamp_str: "2026-05-20T12:02:00Z"
      }
    ],
    fill_row_dict_list: [
      {
        asset_str: "AAPL",
        fill_amount_float: 28,
        fill_price_float: 101,
        fill_timestamp_str: "2026-05-20T12:03:00Z",
        official_open_price_float: 100.5
      }
    ]
  },
  latest_execution_report_dict: {
    fill_count_int: 1,
    fill_with_official_open_count_int: 1,
    broker_order_count_int: 1,
    broker_ack_count_int: 1,
    residual_count_int: 1,
    official_open_slippage_bps_float: 49.7512,
    official_open_slippage_notional_float: 14,
    vplan_reference_slippage_bps_float: 100,
    vplan_reference_slippage_notional_float: 28,
    execution_row_dict_list: [
      {
        asset_str: "AAPL",
        side_str: "buy",
        planned_order_delta_share_float: 28,
        filled_share_float: 28,
        fill_price_float: 101,
        official_open_price_float: 100.5,
        vplan_reference_price_float: 100,
        target_share_float: 30,
        broker_share_float: 29,
        residual_share_float: 1,
        official_open_slippage_bps_float: 49.7512,
        official_open_slippage_notional_float: 14,
        vplan_reference_slippage_bps_float: 100,
        vplan_reference_slippage_notional_float: 28,
        latest_broker_order_status_str: "Filled"
      }
    ]
  },
  latest_diff_dict: { status_str: "green", open_issue_count_int: 0 }
};

function jsonResponse(payload: unknown, init?: ResponseInit) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status: init?.status || 200,
      headers: { "Content-Type": "application/json" },
      ...init
    })
  );
}

function installFetchMock(summary: DashboardSummary = summaryPayload, detail: PodDetail = detailPayload) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/pods") return jsonResponse(summary);
    if (url === "/api/pods/pod_live_01") return jsonResponse(detail);
    if (url === "/api/pods/pod_paper_01") return jsonResponse({ ...detail, pod_row_dict: paperPodRow });
    if (url === "/api/action-token") return jsonResponse({ action_token_str: "test-token" });
    if (url === "/api/pods/pod_live_01/actions/tick" && init?.method === "POST") {
      return jsonResponse({
        job_id_str: "job-1",
        pod_id_str: "pod_live_01",
        mode_str: "live",
        action_name_str: "tick",
        status_str: "running",
        created_timestamp_str: "2026-05-20T12:10:00Z"
      });
    }
    if (url === "/api/jobs/job-1") {
      return jsonResponse({
        job_id_str: "job-1",
        pod_id_str: "pod_live_01",
        mode_str: "live",
        action_name_str: "tick",
        status_str: "succeeded",
        created_timestamp_str: "2026-05-20T12:10:00Z",
        completed_timestamp_str: "2026-05-20T12:10:02Z"
      });
    }
    return jsonResponse({ message_str: `unexpected ${url}` }, { status: 404 });
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

async function openInlineDetail() {
  render(<App />);
  const card = await livePodCard();
  fireEvent.click(within(card).getByRole("button", { name: "Inspect" }));
  await screen.findByText("Enable controls");
  return screen.getByLabelText("Inline POD detail");
}

async function livePodCard() {
  const livePodLabelList = await screen.findAllByText("pod_live_01");
  const card = livePodLabelList.map((label) => label.closest(".pod-row-card")).find(Boolean) as HTMLElement | undefined;
  if (!card) throw new Error("live POD row was not rendered");
  return card;
}

async function clickLiveStage(stageNamePattern: RegExp) {
  const card = await livePodCard();
  fireEvent.click(within(card).getByRole("button", { name: stageNamePattern }));
}

describe("Dashboard V2 operator cockpit", () => {
  beforeEach(() => {
    installFetchMock();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders Attention above separate LIVE and PAPER sections", async () => {
    render(<App />);

    const attentionHeading = await screen.findByRole("heading", { name: "Attention Queue" });
    const liveHeading = screen.getByRole("heading", { name: "LIVE PODs" });
    expect(screen.getByRole("heading", { name: "PAPER PODs" })).toBeInTheDocument();
    expect(screen.queryByText("Live / Paper PODs")).not.toBeInTheDocument();
    expect(attentionHeading.compareDocumentPosition(liveHeading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    const attention = attentionHeading.closest("section") as HTMLElement;
    expect(within(attention).getByText("pod_live_01")).toBeInTheDocument();
    expect(within(attention).getByText("live / Reconcile")).toBeInTheDocument();
    expect(within(attention).getByText("ACK 1 / Fill 1 / DIFF ok")).toBeInTheDocument();
    expect(within(attention).getByText(/Target/)).toBeInTheDocument();
  });

  it("keeps dangerous controls inside inline detail and disabled until enabled", async () => {
    render(<App />);

    expect(screen.queryByText("Recent Actions")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Action status")).not.toBeInTheDocument();
    const card = await livePodCard();
    fireEvent.click(within(card).getByRole("button", { name: "Inspect" }));

    const detail = await screen.findByLabelText("Inline POD detail");
    expect(screen.queryByLabelText("POD detail")).not.toBeInTheDocument();
    const actionLabelList = ["Run DIFF", "Tick", "Submit", "Reconcile", "EOD"];

    for (const labelStr of actionLabelList) {
      expect(within(detail).getByRole("button", { name: labelStr })).toBeDisabled();
    }

    fireEvent.click(within(detail).getByLabelText("Enable controls"));

    for (const labelStr of actionLabelList) {
      expect(within(detail).getByRole("button", { name: labelStr })).toBeEnabled();
    }
  });

  it("opens Fill stage directly into the execution receipt with BPS and dollar impact", async () => {
    render(<App />);

    await clickLiveStage(/Fill 1 fill/i);
    const detail = await screen.findByLabelText("Inline POD detail");

    expect(within(detail).getByLabelText("Execution receipt")).toBeInTheDocument();
    expect(within(detail).getByText("BPS vs Ref")).toBeInTheDocument();
    expect(within(detail).getByText("BPS vs Open")).toBeInTheDocument();
    expect(within(detail).getByText("$ vs Ref")).toBeInTheDocument();
    expect(within(detail).getByText("$ vs Open")).toBeInTheDocument();
    expect(within(detail).getByText("Positive BPS means worse execution cost; negative BPS means price improvement.")).toBeInTheDocument();
    expect(within(detail).getByText("AAPL")).toBeInTheDocument();
    expect(within(detail).getAllByText("100 bps").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("49.7512 bps").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("$28").length).toBeGreaterThan(0);
  });

  it("shows assets to exit in the Decision stage", async () => {
    render(<App />);

    await clickLiveStage(/Decision complete/i);
    const detail = await screen.findByLabelText("Inline POD detail");

    expect(within(detail).getByText("Assets to exit")).toBeInTheDocument();
    expect(within(detail).getAllByText("MSFT").length).toBeGreaterThan(0);
    expect(within(detail).getByText("Entry targets")).toBeInTheDocument();
    expect(within(detail).getByText("Full targets")).toBeInTheDocument();
  });

  it("separates a current planned DecisionPlan from previous execution evidence", async () => {
    const mixedRow: PodRow = {
      ...livePodRow,
      next_action_str: "wait",
      reason_code_str: "waiting_for_submission_window",
      latest_decision_plan_id_int: 2,
      latest_decision_plan_status_str: "planned",
      latest_vplan_id_int: 1,
      latest_vplan_decision_plan_id_int: 1,
      latest_vplan_is_for_latest_decision_bool: false,
      latest_vplan_cycle_role_str: "previous",
      latest_vplan_status_str: "completed",
      latest_reconciliation_status_str: "passed",
      required_action_dict: {
        label_str: "Wait submission window",
        severity_str: "yellow",
        detail_str: "waiting_for_submission_window"
      },
      debug_summary_dict: {
        severity_str: "yellow",
        verdict_label_str: "Wait submission window",
        primary_reason_str: "waiting_for_submission_window"
      },
      lifecycle_step_dict_list: [
        { step_key_str: "db", label_str: "DB", status_str: "ok", severity_str: "green" },
        { step_key_str: "decision", label_str: "Decision", status_str: "planned", severity_str: "yellow" },
        { step_key_str: "vplan", label_str: "VPlan", status_str: "completed", severity_str: "green", detail_str: "previous cycle #1" },
        { step_key_str: "ack", label_str: "ACK", status_str: "complete", severity_str: "green", detail_str: "previous cycle, rows=10" },
        { step_key_str: "fill", label_str: "Fill", status_str: "recorded", severity_str: "green", detail_str: "previous cycle, fill_records=16" },
        { step_key_str: "reconcile", label_str: "Reconcile", status_str: "passed", severity_str: "green" },
        { step_key_str: "eod", label_str: "EOD", status_str: "waiting", severity_str: "gray" }
      ]
    };
    const mixedSummary: DashboardSummary = {
      ...summaryPayload,
      pod_row_dict_list: [mixedRow],
      alert_summary_dict: { total_count_int: 1, red_count_int: 0, yellow_count_int: 1, gray_count_int: 0 },
      combined_book_dict: { environment_dict_list: [] }
    };
    const mixedDetail: PodDetail = {
      ...detailPayload,
      pod_row_dict: mixedRow,
      required_action_dict: mixedRow.required_action_dict,
      lifecycle_step_dict_list: mixedRow.lifecycle_step_dict_list,
      latest_decision_plan_dict: {
        decision_plan_id_int: 2,
        status_str: "planned",
        signal_timestamp_str: "2026-05-20T20:00:00Z",
        submission_timestamp_str: "2026-05-21T13:30:00Z",
        target_execution_timestamp_str: "2026-05-21T13:30:00Z",
        execution_policy_str: "next_open_market",
        decision_book_type_str: "incremental_entry_exit_book",
        exit_asset_list: ["CDNS", "GL"],
        entry_target_weight_map_dict: { NEE: 0.1, VRT: 0.1 },
        display_target_weight_map_dict: { NEE: 0.1, VRT: 0.1 },
        decision_base_position_map_dict: { CDNS: 28, GL: 63 },
        snapshot_metadata_dict: {
          norgate_data_profile_str: "norgate_eod_sp500_pit",
          norgate_snapshot_date_str: "2026-05-20"
        }
      },
      latest_vplan_dict: {
        ...detailPayload.latest_vplan_dict,
        vplan_id_int: 1,
        decision_plan_id_int: 1,
        status_str: "completed"
      },
      debug_story_dict: {
        verdict_dict: {
          severity_str: "yellow",
          verdict_label_str: "Wait submission window",
          primary_reason_str: "waiting_for_submission_window"
        },
        timeline_event_dict_list: [
          {
            source_str: "DecisionPlan",
            label_str: "DecisionPlan",
            status_str: "planned",
            severity_str: "yellow",
            timestamp_str: "2026-05-20T20:00:00Z",
            detail_str: "execute=2026-05-21T13:30:00Z",
            decision_plan_id_int: 2,
            cycle_role_str: "current"
          },
          {
            source_str: "VPlan",
            label_str: "VPlan",
            status_str: "completed",
            severity_str: "green",
            timestamp_str: "2026-05-20T13:30:00Z",
            detail_str: "vplan_id=1",
            decision_plan_id_int: 1,
            vplan_id_int: 1,
            cycle_role_str: "previous"
          },
          {
            source_str: "Norgate",
            label_str: "Sync",
            status_str: "ready",
            severity_str: "green",
            timestamp_str: null,
            detail_str: "<script>alert(1)</script>",
            decision_plan_id_int: null,
            vplan_id_int: null,
            cycle_role_str: null
          }
        ]
      }
    };
    installFetchMock(mixedSummary, mixedDetail);

    const detail = await openInlineDetail();

    expect(within(detail).getByText("Current planned cycle")).toBeInTheDocument();
    expect(within(detail).getByText("Wait submission window")).toBeInTheDocument();
    expect(within(detail).getAllByText("CDNS").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("GL").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("NEE").length).toBeGreaterThan(0);
    expect(within(detail).getAllByText("VRT").length).toBeGreaterThan(0);
    expect(within(detail).getByText(/Previous execution cycle/)).toBeInTheDocument();
    expect(within(detail).getAllByText(/prev completed/).length).toBeGreaterThan(0);

    fireEvent.click(within(detail).getByRole("button", { name: "Operator Log" }));
    expect(within(detail).getByLabelText("Operator Log")).toBeInTheDocument();
    expect(within(detail).getByText("plan=2 / current")).toBeInTheDocument();
    expect(within(detail).getByText("plan=1 / vplan=1 / previous")).toBeInTheDocument();
    expect(within(detail).getByText("<script>alert(1)</script>")).toBeInTheDocument();
  });

  it("handles empty, single-point, and multi-point equity states", async () => {
    installFetchMock(summaryPayload, { ...detailPayload, pod_pnl_dict: { status_str: "unavailable", equity_point_dict_list: [] } });
    const emptyDetail = await openInlineDetail();
    fireEvent.click(within(emptyDetail).getByRole("button", { name: "equity" }));
    expect(within(emptyDetail).getByText("No EOD equity samples yet. The equity curve appears after the first EOD snapshot is written.")).toBeInTheDocument();

    cleanup();
    installFetchMock(summaryPayload, {
      ...detailPayload,
      pod_pnl_dict: {
        status_str: "available",
        point_count_int: 1,
        latest_equity_float: 100000,
        equity_point_dict_list: [{ market_date_str: "2026-05-20", equity_float: 100000 }]
      }
    });
    const singleDetail = await openInlineDetail();
    fireEvent.click(within(singleDetail).getByRole("button", { name: "equity" }));
    expect(within(singleDetail).getByText("One EOD sample")).toBeInTheDocument();

    cleanup();
    installFetchMock();
    const curveDetail = await openInlineDetail();
    fireEvent.click(within(curveDetail).getByRole("button", { name: "equity" }));
    expect(within(curveDetail).getByLabelText("POD equity curve")).toBeInTheDocument();
  });

  it("shows a compact action status strip after a session action starts", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const detail = await openInlineDetail();

    fireEvent.click(within(detail).getByLabelText("Enable controls"));
    fireEvent.click(within(detail).getByRole("button", { name: "Tick" }));

    const strip = await screen.findByLabelText("Action status");
    expect(within(strip).getByText("Tick")).toBeInTheDocument();
    expect(within(strip).getByText("pod_live_01")).toBeInTheDocument();
    expect(within(strip).queryByText("Recent Actions")).not.toBeInTheDocument();
  });
});
