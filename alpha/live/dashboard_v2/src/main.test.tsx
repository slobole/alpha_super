import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./main";
import type { DashboardSummary, PodDetail, PodRow } from "./types";

const lifecycleStepList = [
  { step_key_str: "db", label_str: "DB", status_str: "complete", severity_str: "green" },
  { step_key_str: "decision", label_str: "Decision", status_str: "waiting", severity_str: "yellow" },
  { step_key_str: "reconcile", label_str: "Reconcile", status_str: "pending", severity_str: "gray" }
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
  latest_vplan_status_str: "waiting",
  latest_reconciliation_status_str: "pending",
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
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/pods") return jsonResponse(summaryPayload);
    if (url === "/api/pods/pod_live_01") return jsonResponse(detailPayload);
    if (url === "/api/action-token") return jsonResponse({ action_token_str: "test-token" });
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
