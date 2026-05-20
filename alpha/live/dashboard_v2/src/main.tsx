import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Gauge,
  History,
  ListChecks,
  Loader2,
  Move,
  PauseCircle,
  RefreshCcw,
  Search,
  TerminalSquare,
  X
} from "lucide-react";
import "./styles.css";
import type {
  ActionToken,
  DashboardJob,
  DashboardSummary,
  DataFreshness,
  EventRow,
  LifecycleStep,
  PodDetail,
  PodRow,
  RequiredAction,
  Severity
} from "./types";

const OPS_ACTION_LIST = ["tick", "submit_vplan", "post_execution_reconcile", "eod_snapshot"] as const;
const REFRESH_MS = 4000;

type ViewName = "ops" | "incubation";
type DetailTab = "summary" | "logs" | "lifecycle" | "broker" | "freshness" | "raw";

interface PanelRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function App() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string>("loading");
  const [viewName, setViewName] = useState<ViewName>(currentViewName());
  const [selectedPodId, setSelectedPodId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<PodDetail | null>(null);
  const [detailTab, setDetailTab] = useState<DetailTab>("summary");
  const [detailLoading, setDetailLoading] = useState(false);
  const [jobMap, setJobMap] = useState<Record<string, DashboardJob>>({});

  useEffect(() => {
    void refreshSummary();
    const intervalId = window.setInterval(() => void refreshSummary(), REFRESH_MS);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const onPopState = () => setViewName(currentViewName());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (!selectedPodId) return;
    let activeBool = true;
    void loadPodDetail(selectedPodId, () => activeBool);
    return () => {
      activeBool = false;
    };
  }, [selectedPodId]);

  async function refreshSummary() {
    try {
      const payload = await fetchJson<DashboardSummary>("/api/pods");
      setSummary(payload);
      setLoadError(null);
      setLastRefresh(formatTime(new Date().toISOString()));
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }

  async function loadPodDetail(podId: string, shouldApplyFn: () => boolean = () => true) {
    setDetailLoading(true);
    try {
      const payload = await fetchJson<PodDetail>(`/api/pods/${encodeURIComponent(podId)}`);
      if (shouldApplyFn()) setSelectedDetail(payload);
    } catch (error) {
      if (shouldApplyFn()) {
        setSelectedDetail({
          pod_row_dict: rowByPodId(summary?.pod_row_dict_list || [], podId) || {
            pod_id_str: podId,
            mode_str: "unknown",
            account_route_str: "-",
            strategy_import_str: "-"
          },
          event_dict_list: [
            {
              level_str: "error",
              event_name_str: "detail.fetch_failed",
              message_str: error instanceof Error ? error.message : String(error)
            }
          ]
        });
      }
    } finally {
      if (shouldApplyFn()) setDetailLoading(false);
    }
  }

  function selectPod(row: PodRow) {
    setSelectedDetail(null);
    setDetailTab("summary");
    setSelectedPodId(row.pod_id_str);
  }

  function navigate(view: ViewName) {
    const path = view === "incubation" ? "/incubation" : "/";
    window.history.pushState({}, "", path);
    setViewName(view);
    setSelectedPodId(null);
    setSelectedDetail(null);
  }

  async function startAction(row: PodRow, actionName: string) {
    const confirmed = window.confirm(`Run ${actionName} for ${row.mode_str}/${row.pod_id_str}?`);
    if (!confirmed) return;
    const url =
      actionName === "compare_reference"
        ? `/api/pods/${encodeURIComponent(row.pod_id_str)}/diff/run`
        : `/api/pods/${encodeURIComponent(row.pod_id_str)}/actions/${encodeURIComponent(actionName)}`;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    try {
      headers["X-Alpha-Action-Token"] = await fetchActionToken();
      const job = await fetchJson<DashboardJob>(url, {
        method: "POST",
        headers,
        body: JSON.stringify({ confirmed_bool: true })
      });
      setJobMap((current) => ({ ...current, [job.job_id_str]: job }));
      void pollJob(job.job_id_str);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : String(error));
    }
  }

  async function pollJob(jobId: string) {
    try {
      const job = await fetchJson<DashboardJob>(`/api/jobs/${encodeURIComponent(jobId)}`);
      setJobMap((current) => ({ ...current, [jobId]: job }));
      if (["queued", "running"].includes(job.status_str)) {
        window.setTimeout(() => void pollJob(jobId), 1400);
      } else {
        void refreshSummary();
        if (selectedPodId) void loadPodDetail(selectedPodId);
      }
    } catch (error) {
      setJobMap((current) => ({
        ...current,
        [jobId]: {
          job_id_str: jobId,
          pod_id_str: "-",
          mode_str: "-",
          status_str: "failed",
          created_timestamp_str: new Date().toISOString(),
          error_str: error instanceof Error ? error.message : String(error)
        }
      }));
    }
  }

  const podRows = summary?.pod_row_dict_list || [];
  const opsRows = useMemo(() => podRows.filter((row) => row.mode_str === "live" || row.mode_str === "paper"), [podRows]);
  const incubationRows = useMemo(() => podRows.filter((row) => row.mode_str === "incubation"), [podRows]);
  const visibleRows = viewName === "incubation" ? incubationRows : opsRows;
  const attentionRows = useMemo(() => buildAttentionRows(visibleRows), [visibleRows]);
  const topVerdict = resolveTopVerdict(viewName === "incubation" ? incubationRows : opsRows);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">Alpha Live</div>
          <h1>Operator Console</h1>
        </div>
        <nav className="nav-tabs" aria-label="Dashboard views">
          <button className={viewName === "ops" ? "active" : ""} onClick={() => navigate("ops")}>
            <Gauge size={16} /> Operations
          </button>
          <button className={viewName === "incubation" ? "active" : ""} onClick={() => navigate("incubation")}>
            <ClipboardList size={16} /> Incubation
          </button>
        </nav>
      </header>

      <main className="workspace">
        <section className={`status-hero ${severityClass(topVerdict.severity)}`}>
          <div className="hero-main">
            {topVerdict.severity === "green" ? <CheckCircle2 size={26} /> : <AlertTriangle size={26} />}
            <div>
              <div className="hero-title">{topVerdict.title}</div>
              <div className="hero-subtitle">{topVerdict.subtitle}</div>
            </div>
          </div>
          <div className="hero-metrics">
            <Metric label="Visible" value={visibleRows.length} />
            <Metric label="Needs Action" value={countByGroup(visibleRows, "Needs Action")} tone="red" />
            <Metric label="Waiting" value={countByGroup(visibleRows, "Waiting")} tone="yellow" />
            <Metric label="Last Refresh" value={lastRefresh} />
          </div>
          <button className="icon-button" onClick={() => void refreshSummary()} title="Refresh">
            <RefreshCcw size={17} />
          </button>
        </section>

        {loadError && <div className="inline-alert red">Refresh failed: {loadError}</div>}

        {viewName === "ops" ? (
          <OpsHome
            rows={opsRows}
            attentionRows={attentionRows}
            jobMap={jobMap}
            onSelect={selectPod}
          />
        ) : (
          <IncubationHome
            rows={incubationRows}
            attentionRows={attentionRows}
            onSelect={selectPod}
          />
        )}
      </main>

      {selectedPodId && (
        <DetailDrawer
          detail={selectedDetail}
          loading={detailLoading}
          tab={detailTab}
          onTab={setDetailTab}
          onClose={() => {
            setSelectedPodId(null);
            setSelectedDetail(null);
          }}
          onAction={startAction}
        />
      )}
    </div>
  );
}

function OpsHome({
  rows,
  attentionRows,
  jobMap,
  onSelect
}: {
  rows: PodRow[];
  attentionRows: AttentionRow[];
  jobMap: Record<string, DashboardJob>;
  onSelect: (row: PodRow) => void;
}) {
  return (
    <div className="content-grid">
      <section className="surface attention-surface">
        <SectionTitle icon={<ListChecks size={18} />} title="Attention Queue" aside={`${attentionRows.length} items`} />
        {attentionRows.length === 0 ? (
          <EmptyState text="No live or paper PODs are enabled." />
        ) : (
          <div className="attention-list">
            {attentionRows.map((item) => (
              <button
                key={item.row.pod_id_str}
                className={`attention-item ${severityClass(item.severity)}`}
                onClick={() => onSelect(item.row)}
              >
                <StatusDot severity={item.severity} />
                <div>
                  <strong>{item.row.pod_id_str}</strong>
                  <span>{item.state}</span>
                </div>
                <small>{item.reason}</small>
                <ArrowRight size={16} />
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="surface pods-surface">
        <SectionTitle icon={<Activity size={18} />} title="Live / Paper PODs" aside={`${rows.length} pods`} />
        <div className="pod-table">
          {rows.map((row) => (
            <PodRowCard key={row.pod_id_str} row={row} onSelect={() => onSelect(row)} />
          ))}
          {!rows.length && <EmptyState text="No live or paper PODs are enabled." />}
        </div>
      </section>

      <section className="surface jobs-surface">
        <SectionTitle icon={<History size={18} />} title="Recent Actions" aside="this session" />
        <JobList jobMap={jobMap} />
      </section>
    </div>
  );
}

function IncubationHome({
  rows,
  attentionRows,
  onSelect
}: {
  rows: PodRow[];
  attentionRows: AttentionRow[];
  onSelect: (row: PodRow) => void;
}) {
  return (
    <div className="content-grid incubation-grid">
      <section className="surface pods-surface wide">
        <SectionTitle icon={<ClipboardList size={18} />} title="Incubation Rehearsal" aside={`${rows.length} pods`} />
        <div className="pod-table">
          {rows.map((row) => (
            <PodRowCard key={row.pod_id_str} row={row} onSelect={() => onSelect(row)} incubation />
          ))}
          {!rows.length && <EmptyState text="No incubation PODs are enabled." />}
        </div>
      </section>
      <section className="surface">
        <SectionTitle icon={<ListChecks size={18} />} title="Promotion Queue" aside={`${attentionRows.length} items`} />
        <div className="attention-list">
          {attentionRows.map((item) => (
            <button key={item.row.pod_id_str} className="attention-item gray" onClick={() => onSelect(item.row)}>
              <StatusDot severity={item.severity} />
              <div>
                <strong>{item.row.pod_id_str}</strong>
                <span>{rehearsalLabel(item.row)}</span>
              </div>
              <small>{item.reason}</small>
            </button>
          ))}
          {!attentionRows.length && <EmptyState text="No rehearsal state to review." />}
        </div>
      </section>
    </div>
  );
}

function PodRowCard({
  row,
  incubation,
  onSelect
}: {
  row: PodRow;
  incubation?: boolean;
  onSelect: () => void;
}) {
  const action = row.required_action_dict || {};
  const severity = effectiveSeverity(row);
  return (
    <div className={`pod-row-card ${severityClass(severity)}`}>
      <button className="pod-row-main" onClick={onSelect}>
        <StatusDot severity={severity} />
        <div className="pod-identity">
          <strong>{row.pod_id_str}</strong>
          <span>{row.strategy_import_str || "-"}</span>
        </div>
        <Pill>{row.mode_str}</Pill>
        <div className="pod-state">
          <strong>{incubation ? rehearsalLabel(row) : action.label_str || row.next_action_str || "No action"}</strong>
          <span>{shortReason(row)}</span>
        </div>
        <div className="pod-numbers">
          <span>{formatMoney(row.equity_float)}</span>
          <small>{row.position_count_int ?? 0} pos</small>
        </div>
      </button>
      <MiniStageRail steps={row.lifecycle_step_dict_list || []} />
      <div className="row-actions">
        <button className="quiet-button" onClick={onSelect}>
          <Search size={15} /> Inspect
        </button>
      </div>
    </div>
  );
}

function MiniStageRail({ steps }: { steps: LifecycleStep[] }) {
  const executionSteps = steps.filter((step) => step.step_key_str !== "diff");
  if (!executionSteps.length) return null;
  const currentKey = currentStageKey(executionSteps);
  return (
    <div className="mini-stage-rail" aria-label="POD current lifecycle stages">
      {executionSteps.map((step) => (
        <div
          className={`mini-stage ${severityClass(step.severity_str || step.status_str || "gray")} ${step.step_key_str === currentKey ? "current" : ""}`}
          key={step.step_key_str || step.label_str}
          title={`${step.label_str || step.step_key_str}: ${step.status_str || "-"}`}
        >
          <span>{step.label_str || step.step_key_str}</span>
        </div>
      ))}
    </div>
  );
}

function DetailDrawer({
  detail,
  loading,
  tab,
  onTab,
  onClose,
  onAction
}: {
  detail: PodDetail | null;
  loading: boolean;
  tab: DetailTab;
  onTab: (tab: DetailTab) => void;
  onClose: () => void;
  onAction: (row: PodRow, actionName: string) => void;
}) {
  const row = detail?.pod_row_dict;
  const [controlsEnabled, setControlsEnabled] = useState(false);
  const [panelRect, setPanelRect] = useState<PanelRect>(() => initialPanelRect());

  useEffect(() => {
    setControlsEnabled(false);
    setPanelRect(initialPanelRect());
  }, [row?.pod_id_str]);

  function startDrag(event: React.PointerEvent<HTMLElement>) {
    if ((event.target as HTMLElement).closest("button, input")) return;
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startRect = panelRect;
    const onMove = (moveEvent: PointerEvent) => {
      setPanelRect(
        clampPanelRect({
          ...startRect,
          left: startRect.left + moveEvent.clientX - startX,
          top: startRect.top + moveEvent.clientY - startY
        })
      );
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  function startResize(event: React.PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startY = event.clientY;
    const startRect = panelRect;
    const onMove = (moveEvent: PointerEvent) => {
      setPanelRect(
        clampPanelRect({
          ...startRect,
          width: startRect.width + moveEvent.clientX - startX,
          height: startRect.height + moveEvent.clientY - startY
        })
      );
    };
    const onUp = () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  return (
    <div className="drawer-backdrop">
      <aside
        className="detail-drawer"
        aria-label="POD detail"
        style={{
          left: panelRect.left,
          top: panelRect.top,
          width: panelRect.width,
          height: panelRect.height
        }}
      >
        <div className="drawer-header" onPointerDown={startDrag}>
          <div className="drawer-title">
            <Move size={16} />
            <div className="eyebrow">POD Detail</div>
            <h2>{row?.pod_id_str || "Loading"}</h2>
          </div>
          <button className="icon-button" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </div>
        {loading && (
          <div className="drawer-loading">
            <Loader2 className="spin" size={18} /> Loading latest state
          </div>
        )}
        {row && (
          <>
            <div className="drawer-control-gate">
              <label>
                <input
                  type="checkbox"
                  checked={controlsEnabled}
                  onChange={(event) => setControlsEnabled(event.currentTarget.checked)}
                />
                Enable controls
              </label>
              <span>Actions stay disabled until enabled for this POD.</span>
            </div>
            <div className={`drawer-actions ${controlsEnabled ? "" : "disabled"}`}>
              <button disabled={!controlsEnabled} onClick={() => onAction(row, "compare_reference")}>
                Run DIFF
              </button>
              {OPS_ACTION_LIST.map((actionName) => (
                <button key={actionName} disabled={!controlsEnabled} onClick={() => onAction(row, actionName)}>
                  {actionLabel(actionName)}
                </button>
              ))}
            </div>
            <div className="drawer-tabs">
              {(["summary", "logs", "lifecycle", "broker", "freshness", "raw"] as DetailTab[]).map((tabName) => (
                <button key={tabName} className={tab === tabName ? "active" : ""} onClick={() => onTab(tabName)}>
                  {tabName}
                </button>
              ))}
            </div>
            <div className="drawer-body">
              {tab === "summary" && <SummaryTab detail={detail} />}
              {tab === "logs" && <LogsTab events={detail?.event_dict_list || []} />}
              {tab === "lifecycle" && <LifecycleTab steps={detail?.lifecycle_step_dict_list || []} />}
              {tab === "broker" && <BrokerTab detail={detail} />}
              {tab === "freshness" && <FreshnessTab freshness={detail?.data_freshness_dict} />}
              {tab === "raw" && <RawTab detail={detail} />}
            </div>
          </>
        )}
        <div className="drawer-resize-handle" onPointerDown={startResize} title="Resize POD detail" />
      </aside>
    </div>
  );
}

function SummaryTab({ detail }: { detail: PodDetail | null }) {
  const row = detail?.pod_row_dict;
  if (!row) return null;
  const action = detail?.required_action_dict || row.required_action_dict || {};
  return (
    <div className="detail-stack">
      <div className="summary-line">
        <StatusDot severity={effectiveSeverity(row)} />
        <div>
          <strong>{action.label_str || row.next_action_str || "No action"}</strong>
          <span>{action.detail_str || shortReason(row)}</span>
        </div>
      </div>
      <div className="metric-grid">
        <Metric label="Mode" value={row.mode_str} />
        <Metric label="Account" value={row.account_route_str} />
        <Metric label="Equity" value={formatMoney(row.equity_float)} />
        <Metric label="Cash" value={formatMoney(row.cash_float)} />
        <Metric label="Decision" value={row.latest_decision_plan_status_str || "-"} />
        <Metric label="VPlan" value={row.latest_vplan_status_str || "-"} />
        <Metric label="Reconcile" value={row.latest_reconciliation_status_str || "-"} />
        <Metric label="Latest Event" value={formatTime(row.latest_event_timestamp_str)} />
      </div>
      {detail?.latest_diff_dict && (
        <div className="subtle-box">
          <strong>Reference DIFF</strong>
          <span>{String(detail.latest_diff_dict.status_str || "not_run")}</span>
        </div>
      )}
      <StageMap steps={detail?.lifecycle_step_dict_list || row.lifecycle_step_dict_list || []} />
    </div>
  );
}

function LogsTab({ events }: { events: EventRow[] }) {
  if (!events.length) return <EmptyState text="No recent POD events." />;
  return (
    <div className="log-list">
      {events.slice().reverse().map((event, index) => (
        <div className={`log-row ${severityClass(event.level_str || event.severity_str || "gray")}`} key={`${event.event_name_str}-${index}`}>
          <span>{formatTime(event.timestamp_str || event.event_timestamp_str || event.created_timestamp_str)}</span>
          <strong>{event.event_name_str || "event"}</strong>
          <small>{event.reason_str || event.message_str || event.error_str || "-"}</small>
        </div>
      ))}
    </div>
  );
}

function LifecycleTab({ steps }: { steps: LifecycleStep[] }) {
  if (!steps.length) return <EmptyState text="No lifecycle steps were returned." />;
  return (
    <div className="detail-stack">
      <StageMap steps={steps} />
      <div className="lifecycle-list">
        {steps.map((step) => (
          <div className={`lifecycle-row ${severityClass(step.severity_str || step.status_str || "gray")}`} key={step.step_key_str || step.label_str}>
            <StatusDot severity={step.severity_str || step.status_str || "gray"} />
            <div>
              <strong>{step.label_str || step.step_key_str}</strong>
              <span>{step.detail_str || step.evidence_str || step.status_str}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StageMap({ steps }: { steps: LifecycleStep[] }) {
  const executionSteps = steps.filter((step) => step.step_key_str !== "diff");
  const diffStep = steps.find((step) => step.step_key_str === "diff");
  if (!executionSteps.length) return <EmptyState text="No stage map was returned." />;
  const currentKey = currentStageKey(executionSteps);
  return (
    <div className="stage-map" aria-label="POD lifecycle stage map">
      <div className="stage-map-header">
        <strong>Stage Map</strong>
        <span>{stageSummary(executionSteps, currentKey)}</span>
      </div>
      <div className="stage-rail">
        {executionSteps.map((step, index) => {
          const current = step.step_key_str === currentKey;
          return (
            <React.Fragment key={step.step_key_str || step.label_str}>
              <div className={`stage-node ${severityClass(step.severity_str || step.status_str || "gray")} ${current ? "current" : ""}`}>
                <span>{step.label_str || step.step_key_str}</span>
                <strong>{step.status_str || "-"}</strong>
                {current && <small>current</small>}
              </div>
              {index < executionSteps.length - 1 && <ArrowRight className="stage-arrow" size={15} />}
            </React.Fragment>
          );
        })}
      </div>
      {diffStep && (
        <div className={`stage-reference ${severityClass(diffStep.severity_str || diffStep.status_str || "gray")}`}>
          <span>DIFF</span>
          <strong>{diffStep.status_str || "-"}</strong>
          <small>{diffStep.detail_str || diffStep.evidence_str || "-"}</small>
        </div>
      )}
    </div>
  );
}

function BrokerTab({ detail }: { detail: PodDetail | null }) {
  const row = detail?.pod_row_dict;
  return (
    <div className="metric-grid">
      <Metric label="Account route" value={row?.account_route_str || "-"} />
      <Metric label="Positions" value={row?.position_count_int ?? 0} />
      <Metric label="Warnings" value={row?.warning_count_int ?? 0} />
      <Metric label="Latest state" value={formatTime(row?.latest_pod_state_timestamp_str)} />
      <Metric label="EOD status" value={String(row?.eod_snapshot_dict?.status_str || "-")} />
      <Metric label="DB status" value={row?.db_status_str || "-"} />
    </div>
  );
}

function FreshnessTab({ freshness }: { freshness?: DataFreshness }) {
  const itemList = freshness?.item_dict_list || [];
  if (!freshness && !itemList.length) return <EmptyState text="No freshness data was returned." />;
  return (
    <div className="detail-stack">
      <div className="summary-line">
        <StatusDot severity={freshness?.severity_str || freshness?.status_str || "gray"} />
        <div>
          <strong>{freshness?.status_str || "unknown"}</strong>
          <span>Norgate: {freshness?.norgate_status_str || "-"}</span>
        </div>
      </div>
      <div className="freshness-list">
        {itemList.map((item) => (
          <div className={`freshness-row ${severityClass(item.severity_str || item.status_str || "gray")}`} key={item.label_str}>
            <strong>{item.label_str}</strong>
            <span>{item.status_str}</span>
            <small>{item.detail_str || formatTime(item.timestamp_str)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function RawTab({ detail }: { detail: PodDetail | null }) {
  if (!detail) return null;
  return (
    <div className="raw-view">
      {Object.entries(detail).map(([key, value]) => (
        <details key={key}>
          <summary>{key}</summary>
          <code>{compactJson(value)}</code>
        </details>
      ))}
    </div>
  );
}

function JobList({ jobMap }: { jobMap: Record<string, DashboardJob> }) {
  const jobs = Object.values(jobMap).sort((left, right) => right.created_timestamp_str.localeCompare(left.created_timestamp_str));
  if (!jobs.length) return <EmptyState text="No actions started from this browser session." />;
  return (
    <div className="job-list">
      {jobs.slice(0, 8).map((job) => (
        <div className={`job-row ${severityClass(job.status_str === "failed" ? "red" : job.status_str === "succeeded" ? "green" : "yellow")}`} key={job.job_id_str}>
          <TerminalSquare size={16} />
          <div>
            <strong>{job.action_name_str || "job"} / {job.pod_id_str}</strong>
            <span>{job.status_str}{job.error_str ? `: ${job.error_str}` : ""}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function SectionTitle({ icon, title, aside }: { icon: React.ReactNode; title: string; aside?: string }) {
  return (
    <div className="section-title">
      <div>{icon}<h2>{title}</h2></div>
      {aside && <span>{aside}</span>}
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: React.ReactNode; tone?: string }) {
  return (
    <div className={`metric ${tone || ""}`}>
      <span>{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}

function Pill({ children }: { children: React.ReactNode }) {
  return <span className="pill">{children}</span>;
}

function StatusDot({ severity }: { severity?: Severity }) {
  return <span className={`status-dot ${severityClass(severity)}`} />;
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="empty-state">
      <PauseCircle size={17} />
      <span>{text}</span>
    </div>
  );
}

interface AttentionRow {
  row: PodRow;
  group: string;
  severity: Severity;
  state: string;
  reason: string;
}

function buildAttentionRows(rows: PodRow[]): AttentionRow[] {
  return rows
    .map((row) => ({
      row,
      group: queueGroup(row),
      severity: effectiveSeverity(row),
      state: stateSentence(row),
      reason: shortReason(row)
    }))
    .filter((item) => item.group !== "Healthy")
    .sort((left, right) => {
      const groupDelta = groupRank(left.group) - groupRank(right.group);
      if (groupDelta !== 0) return groupDelta;
      return severityRank(left.severity) - severityRank(right.severity);
    });
}

function resolveTopVerdict(rows: PodRow[]) {
  const redCount = rows.filter((row) => severityClass(effectiveSeverity(row)) === "red").length;
  const yellowCount = rows.filter((row) => severityClass(effectiveSeverity(row)) === "yellow").length;
  const grayCount = rows.filter((row) => severityClass(effectiveSeverity(row)) === "gray").length;
  if (!rows.length) return { severity: "gray", title: "No PODs in this view", subtitle: "Nothing is enabled for this page." };
  if (redCount > 0) return { severity: "red", title: `${redCount} POD${redCount === 1 ? "" : "s"} need action`, subtitle: "Open the attention queue and inspect the first red item." };
  if (yellowCount > 0) return { severity: "yellow", title: `${yellowCount} POD${yellowCount === 1 ? "" : "s"} waiting`, subtitle: "No red blocker, but timing or review is pending." };
  if (grayCount > 0) return { severity: "gray", title: `${grayCount} POD${grayCount === 1 ? "" : "s"} missing state`, subtitle: "Setup or stale data needs inspection before trusting the view." };
  return { severity: "green", title: "All clear", subtitle: "Enabled PODs are idle, complete, or healthy." };
}

function queueGroup(row: PodRow) {
  const severity = severityClass(effectiveSeverity(row));
  const label = `${row.debug_summary_dict?.verdict_label_str || ""} ${row.required_action_dict?.label_str || ""}`.toLowerCase();
  if (severity === "red") return "Needs Action";
  if (severity === "green") return "Healthy";
  if (label.includes("waiting") || label.includes("parked") || severity === "yellow") return "Waiting";
  if (severity === "gray") return "Missing / Stale";
  return "Healthy";
}

function countByGroup(rows: PodRow[], group: string) {
  return rows.filter((row) => queueGroup(row) === group).length;
}

function groupRank(group: string) {
  if (group === "Needs Action") return 0;
  if (group === "Waiting") return 1;
  if (group === "Missing / Stale") return 2;
  return 3;
}

function severityRank(severity: Severity) {
  const clean = severityClass(severity);
  if (clean === "red") return 0;
  if (clean === "yellow") return 1;
  if (clean === "gray") return 2;
  return 3;
}

function effectiveSeverity(row: PodRow): Severity {
  return row.debug_summary_dict?.severity_str || row.required_action_dict?.severity_str || row.health_str || "gray";
}

function initialPanelRect(): PanelRect {
  if (typeof window === "undefined") return { left: 24, top: 76, width: 760, height: 720 };
  const maxWidth = Math.max(300, window.innerWidth - 24);
  const maxHeight = Math.max(320, window.innerHeight - 24);
  return clampPanelRect({
    left: Math.max(12, window.innerWidth - Math.min(860, maxWidth) - 24),
    top: 76,
    width: Math.min(860, Math.max(430, window.innerWidth - 48), maxWidth),
    height: Math.min(820, Math.max(420, window.innerHeight - 110), maxHeight)
  });
}

function clampPanelRect(rect: PanelRect): PanelRect {
  if (typeof window === "undefined") return rect;
  const maxWidth = Math.max(300, window.innerWidth - 24);
  const maxHeight = Math.max(320, window.innerHeight - 24);
  const minWidth = Math.min(430, maxWidth);
  const minHeight = Math.min(360, maxHeight);
  const width = Math.min(Math.max(rect.width, minWidth), maxWidth);
  const height = Math.min(Math.max(rect.height, minHeight), maxHeight);
  const left = Math.min(Math.max(rect.left, 12), Math.max(12, window.innerWidth - width - 12));
  const top = Math.min(Math.max(rect.top, 12), Math.max(12, window.innerHeight - height - 12));
  return { left, top, width, height };
}

function currentStageKey(steps: LifecycleStep[]): string | undefined {
  const redStep = steps.find((step) => severityClass(step.severity_str || step.status_str || "gray") === "red");
  if (redStep?.step_key_str) return redStep.step_key_str;
  const yellowStep = steps.find((step) => severityClass(step.severity_str || step.status_str || "gray") === "yellow");
  if (yellowStep?.step_key_str) return yellowStep.step_key_str;
  const waitingStep = steps.find((step) => ["waiting", "blocked_by_execution"].includes(String(step.status_str || "")));
  if (waitingStep?.step_key_str) return waitingStep.step_key_str;
  const activeStep = steps.slice().reverse().find((step) => severityClass(step.severity_str || step.status_str || "gray") === "green");
  return activeStep?.step_key_str || steps[0]?.step_key_str;
}

function stageSummary(steps: LifecycleStep[], currentKey?: string) {
  const currentStep = steps.find((step) => step.step_key_str === currentKey);
  if (!currentStep) return "no current stage";
  return `${currentStep.label_str || currentStep.step_key_str}: ${currentStep.status_str || "-"}`;
}

function severityClass(severity?: Severity) {
  const text = String(severity || "gray").toLowerCase();
  if (text.includes("red") || text.includes("error") || text.includes("fail")) return "red";
  if (text.includes("yellow") || text.includes("warn") || text.includes("waiting") || text.includes("running") || text.includes("queued")) return "yellow";
  if (text.includes("green") || text.includes("success") || text.includes("complete") || text.includes("succeeded")) return "green";
  return "gray";
}

function stateSentence(row: PodRow) {
  return row.debug_summary_dict?.verdict_label_str || row.required_action_dict?.label_str || row.next_action_str || "No action";
}

function shortReason(row: PodRow) {
  return row.debug_summary_dict?.primary_reason_str || row.required_action_dict?.detail_str || row.reason_code_str || "-";
}

function rehearsalLabel(row: PodRow) {
  const rehearsal = row.rehearsal_status_dict || {};
  return rehearsal.promotion_gate_status_str || rehearsal.last_cycle_status_str || row.next_action_str || "not started";
}

function actionLabel(actionName: string) {
  const labelMap: Record<string, string> = {
    tick: "Tick",
    submit_vplan: "Submit",
    post_execution_reconcile: "Reconcile",
    eod_snapshot: "EOD"
  };
  return labelMap[actionName] || actionName;
}

function rowByPodId(rows: PodRow[], podId: string) {
  return rows.find((row) => row.pod_id_str === podId);
}

function formatMoney(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
}

function formatTime(value?: string | null) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function compactJson(value: unknown) {
  return JSON.stringify(value, null, 2).slice(0, 2400);
}

function currentViewName(): ViewName {
  return window.location.pathname.startsWith("/incubation") ? "incubation" : "ops";
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.message_str || payload.error_str || response.statusText);
  }
  return payload as T;
}

async function fetchActionToken(): Promise<string> {
  const payload = await fetchJson<ActionToken>("/api/action-token", { cache: "no-store" });
  if (!payload.action_token_str) {
    throw new Error("Dashboard action token is missing.");
  }
  return payload.action_token_str;
}

const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
}
