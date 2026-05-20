import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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

const podRow: PodRow = {
  pod_id_str: "pod_live_01",
  mode_str: "live",
  account_route_str: "U123",
  strategy_import_str: "strategies.example",
  health_str: "green",
  next_action_str: "No action",
  reason_code_str: "ok",
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
  latest_submit_ack_status_str: "complete",
  latest_reconciliation_status_str: "pending",
  latest_reconciliation_timestamp_str: "2026-05-20T12:05:00Z",
  broker_ack_count_int: 1,
  broker_order_count_int: 1,
  fill_count_int: 1,
  missing_ack_count_int: 0,
  lifecycle_step_dict_list: lifecycleStepList,
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
  pod_row_dict_list: [podRow],
  alert_dict_list: [],
  alert_summary_dict: { total_count_int: 0 },
  mode_list: ["live"]
};

const detailPayload: PodDetail = {
  pod_row_dict: podRow,
  required_action_dict: podRow.required_action_dict,
  lifecycle_step_dict_list: lifecycleStepList,
  event_dict_list: [],
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
    display_target_weight_map_dict: { AAPL: 0.6 }
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
    official_open_slippage_bps_float: 49.75,
    vplan_reference_slippage_bps_float: 100,
    fill_row_dict_list: [
      {
        asset_str: "AAPL",
        fill_amount_float: 28,
        fill_price_float: 101,
        fill_timestamp_str: "2026-05-20T12:03:00Z",
        official_open_price_float: 100.5
      }
    ],
    execution_row_dict_list: [
      {
        asset_str: "AAPL",
        target_share_float: 30,
        broker_share_float: 29,
        residual_share_float: 1,
        latest_broker_order_status_str: "Filled"
      }
    ]
  },
  latest_diff_dict: { status_str: "not_run" }
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

function installFetchMock() {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/pods") return jsonResponse(summaryPayload);
    if (url === "/api/pods/pod_live_01") return jsonResponse(detailPayload);
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

function firePointer(target: Document | Element, typeStr: string, clientX: number, clientY: number) {
  const event = new Event(typeStr, { bubbles: true });
  Object.defineProperty(event, "clientX", { value: clientX });
  Object.defineProperty(event, "clientY", { value: clientY });
  fireEvent(target, event);
}

async function openDetailPanel() {
  render(<App />);
  const inspectButton = await screen.findByRole("button", { name: "Inspect" });
  fireEvent.click(inspectButton);
  await screen.findByText("Enable controls");
  return screen.getByLabelText("POD detail");
}

describe("Dashboard V2 operator controls", () => {
  beforeEach(() => {
    installFetchMock();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("keeps row actions inspect-only and gates detail actions", async () => {
    render(<App />);

    const inspectButton = await screen.findByRole("button", { name: "Inspect" });
    expect(screen.queryByText("Recent Actions")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Action status")).not.toBeInTheDocument();

    const rowActions = inspectButton.closest(".row-actions");
    expect(rowActions).not.toBeNull();
    expect(within(rowActions as HTMLElement).getAllByRole("button")).toHaveLength(1);
    expect(within(rowActions as HTMLElement).queryByText("DIFF")).not.toBeInTheDocument();
    expect(within(rowActions as HTMLElement).queryByText("Tick")).not.toBeInTheDocument();
    expect(within(rowActions as HTMLElement).queryByText("Reconcile")).not.toBeInTheDocument();

    fireEvent.click(inspectButton);
    const drawer = await screen.findByLabelText("POD detail");
    await screen.findByText("Enable controls");
    const actionLabelList = ["Run DIFF", "Tick", "Submit", "Reconcile", "EOD"];

    for (const labelStr of actionLabelList) {
      expect(within(drawer).getByRole("button", { name: labelStr })).toBeDisabled();
    }

    fireEvent.click(within(drawer).getByLabelText("Enable controls"));

    for (const labelStr of actionLabelList) {
      expect(within(drawer).getByRole("button", { name: labelStr })).toBeEnabled();
    }
  });

  it("opens Fill evidence from one stage chip click", async () => {
    render(<App />);

    const fillChip = await screen.findByRole("button", { name: /Fill 1 fill/i });
    fireEvent.click(fillChip);

    const drawer = await screen.findByLabelText("POD detail");
    expect(within(drawer).getByLabelText("Stage Inspector evidence")).toBeInTheDocument();
    expect(within(drawer).getByRole("heading", { name: "Fill" })).toBeInTheDocument();
    expect(within(drawer).getAllByText("Fills").length).toBeGreaterThan(0);
    expect(within(drawer).getByText("Open Coverage")).toBeInTheDocument();
    expect(within(drawer).getByText("AAPL")).toBeInTheDocument();
    expect(within(drawer).getByText("101")).toBeInTheDocument();
  });

  it("opens ACK evidence and missing ACK state from one stage chip click", async () => {
    const ackMissingRow: PodRow = {
      ...podRow,
      latest_submit_ack_status_str: "missing_ack",
      missing_ack_count_int: 1,
      lifecycle_step_dict_list: lifecycleStepList.map((step) =>
        step.step_key_str === "ack" ? { ...step, status_str: "missing_ack", severity_str: "red" } : step
      )
    };
    const ackMissingDetail: PodDetail = {
      ...detailPayload,
      pod_row_dict: ackMissingRow,
      lifecycle_step_dict_list: ackMissingRow.lifecycle_step_dict_list
    };
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/pods") return jsonResponse({ ...summaryPayload, pod_row_dict_list: [ackMissingRow] });
      if (url === "/api/pods/pod_live_01") return jsonResponse(ackMissingDetail);
      return jsonResponse({ message_str: `unexpected ${url}` }, { status: 404 });
    });
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /ACK missing 1/i }));
    const drawer = await screen.findByLabelText("POD detail");

    expect(within(drawer).getByRole("heading", { name: "ACK" })).toBeInTheDocument();
    expect(within(drawer).getByText("Broker ACK evidence")).toBeInTheDocument();
    expect(within(drawer).getByText("Missing")).toBeInTheDocument();
    expect(within(drawer).getByText("broker_acked")).toBeInTheDocument();
  });

  it("opens Reconcile evidence from one stage chip click", async () => {
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: /Reconcile pending/i }));
    const drawer = await screen.findByLabelText("POD detail");

    expect(within(drawer).getByRole("heading", { name: "Reconcile" })).toBeInTheDocument();
    expect(within(drawer).getByText("Model vs broker")).toBeInTheDocument();
    expect(within(drawer).getByText("Residual")).toBeInTheDocument();
    expect(within(drawer).getByText("Filled")).toBeInTheDocument();
  });

  it("shows a compact action status strip after a session action starts", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<App />);

    fireEvent.click(await screen.findByRole("button", { name: "Inspect" }));
    const drawer = await screen.findByLabelText("POD detail");
    fireEvent.click(within(drawer).getByLabelText("Enable controls"));
    fireEvent.click(within(drawer).getByRole("button", { name: "Tick" }));

    const strip = await screen.findByLabelText("Action status");
    expect(within(strip).getByText("Tick")).toBeInTheDocument();
    expect(within(strip).getByText("pod_live_01")).toBeInTheDocument();
    expect(within(strip).queryByText("Recent Actions")).not.toBeInTheDocument();
  });

  it("moves and resizes the POD detail panel", async () => {
    const drawer = await openDetailPanel();
    const header = drawer.querySelector(".drawer-header") as HTMLElement;
    const resizeHandle = drawer.querySelector(".drawer-resize-handle") as HTMLElement;

    const startLeftStr = (drawer as HTMLElement).style.left;
    const startTopStr = (drawer as HTMLElement).style.top;
    firePointer(header, "pointerdown", 120, 120);
    firePointer(document, "pointermove", 170, 155);
    firePointer(document, "pointerup", 170, 155);

    await waitFor(() => expect((drawer as HTMLElement).style.left).not.toBe(startLeftStr));
    expect((drawer as HTMLElement).style.top).not.toBe(startTopStr);

    const startWidthStr = (drawer as HTMLElement).style.width;
    const startHeightStr = (drawer as HTMLElement).style.height;
    firePointer(resizeHandle, "pointerdown", 500, 500);
    firePointer(document, "pointermove", 560, 545);
    firePointer(document, "pointerup", 560, 545);

    await waitFor(() => expect((drawer as HTMLElement).style.width).not.toBe(startWidthStr));
    expect((drawer as HTMLElement).style.height).not.toBe(startHeightStr);
  });
});
