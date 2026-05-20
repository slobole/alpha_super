import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Gauge,
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
  BrokerAckRow,
  DashboardJob,
  DashboardSummary,
  DebugTimelineEvent,
  EventRow,
  ExecutionReport,
  ExecutionRow,
  FillRow,
  LifecycleStep,
  PodDetail,
  PodRow,
  RequiredAction,
  Severity,
  VPlanDetail,
  VPlanRow
} from "./types";

const OPS_ACTION_LIST = ["tick", "submit_vplan", "post_execution_reconcile", "eod_snapshot"] as const;
const REFRESH_MS = 4000;

type ViewName = "ops" | "incubation";
type DetailTab = "stage" | "timeline" | "logs" | "raw";

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
  const [detailTab, setDetailTab] = useState<DetailTab>("stage");
  const [selectedStageKey, setSelectedStageKey] = useState<string>("summary");
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

  function selectPod(row: PodRow, stageKey?: string) {
    setSelectedDetail(null);
    setDetailTab("stage");
    setSelectedStageKey(stageKey || currentStageKey(row.lifecycle_step_dict_list || [], row) || "summary");
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

        <ActionStatusStrip
          jobMap={jobMap}
          onSelectPod={(podId) => {
            const row = rowByPodId(podRows, podId);
            if (row) selectPod(row);
          }}
        />

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
            jobMap={jobMap}
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
          selectedStageKey={selectedStageKey}
          onStageSelect={(stageKey) => {
            setSelectedStageKey(stageKey);
            setDetailTab("stage");
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
  onSelect: (row: PodRow, stageKey?: string) => void;
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
            <PodRowCard key={row.pod_id_str} row={row} job={podBadgeJob(jobMap, row.pod_id_str)} onSelect={() => onSelect(row)} onStageSelect={(stageKey) => onSelect(row, stageKey)} />
          ))}
          {!rows.length && <EmptyState text="No live or paper PODs are enabled." />}
        </div>
      </section>
    </div>
  );
}

function IncubationHome({
  rows,
  attentionRows,
  jobMap,
  onSelect
}: {
  rows: PodRow[];
  attentionRows: AttentionRow[];
  jobMap: Record<string, DashboardJob>;
  onSelect: (row: PodRow, stageKey?: string) => void;
}) {
  return (
    <div className="content-grid incubation-grid">
      <section className="surface pods-surface wide">
        <SectionTitle icon={<ClipboardList size={18} />} title="Incubation Rehearsal" aside={`${rows.length} pods`} />
        <div className="pod-table">
          {rows.map((row) => (
            <PodRowCard key={row.pod_id_str} row={row} job={podBadgeJob(jobMap, row.pod_id_str)} onSelect={() => onSelect(row)} onStageSelect={(stageKey) => onSelect(row, stageKey)} incubation />
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
  job,
  onSelect,
  onStageSelect
}: {
  row: PodRow;
  incubation?: boolean;
  job?: DashboardJob;
  onSelect: () => void;
  onStageSelect: (stageKey: string) => void;
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
          {job && <JobBadge job={job} />}
        </div>
      </button>
      <MiniStageRail row={row} steps={row.lifecycle_step_dict_list || []} onStageSelect={onStageSelect} />
      <div className="row-actions">
        <button className="quiet-button" onClick={onSelect}>
          <Search size={15} /> Inspect
        </button>
      </div>
    </div>
  );
}

function MiniStageRail({ row, steps, onStageSelect }: { row: PodRow; steps: LifecycleStep[]; onStageSelect: (stageKey: string) => void }) {
  const executionSteps = compactStageStepList(steps);
  if (!executionSteps.length) return null;
  const currentKey = currentStageKey(executionSteps, row);
  return (
    <div className="mini-stage-rail" aria-label="POD current lifecycle stages">
      {executionSteps.map((step) => (
        <button
          type="button"
          className={`mini-stage ${severityClass(step.severity_str || step.status_str || "gray")} ${step.step_key_str === currentKey ? "current" : ""}`}
          key={step.step_key_str || step.label_str}
          title={`${step.label_str || step.step_key_str}: ${step.status_str || "-"}`}
          onClick={() => step.step_key_str && onStageSelect(step.step_key_str)}
        >
          <span>{step.label_str || step.step_key_str}</span>
          <strong>{stageChipStatus(step, row)}</strong>
        </button>
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
  selectedStageKey,
  onStageSelect,
  onAction
}: {
  detail: PodDetail | null;
  loading: boolean;
  tab: DetailTab;
  onTab: (tab: DetailTab) => void;
  onClose: () => void;
  selectedStageKey: string;
  onStageSelect: (stageKey: string) => void;
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
              {(["stage", "timeline", "logs", "raw"] as DetailTab[]).map((tabName) => (
                <button key={tabName} className={tab === tabName ? "active" : ""} onClick={() => onTab(tabName)}>
                  {tabName === "stage" ? "Stage Inspector" : tabName}
                </button>
              ))}
            </div>
            <div className="drawer-body">
              {tab === "stage" && <StageInspector detail={detail} selectedStageKey={selectedStageKey} onStageSelect={onStageSelect} />}
              {tab === "timeline" && <TimelineTab events={detail?.debug_story_dict?.timeline_event_dict_list || []} />}
              {tab === "logs" && <LogsTab events={detail?.event_dict_list || []} />}
              {tab === "raw" && <RawTab detail={detail} />}
            </div>
          </>
        )}
        <div className="drawer-resize-handle" onPointerDown={startResize} title="Resize POD detail" />
      </aside>
    </div>
  );
}

function StageInspector({
  detail,
  selectedStageKey,
  onStageSelect
}: {
  detail: PodDetail | null;
  selectedStageKey: string;
  onStageSelect: (stageKey: string) => void;
}) {
  const row = detail?.pod_row_dict;
  if (!row) return null;
  const action = detail?.required_action_dict || row.required_action_dict || {};
  const stepList = detail?.lifecycle_step_dict_list || row.lifecycle_step_dict_list || [];
  const currentKey = currentStageKey(compactStageStepList(stepList), row);
  const activeStageKey = normalizeStageKey(selectedStageKey) || currentKey || stepList[0]?.step_key_str || "summary";
  const activeStep = stepList.find((step) => step.step_key_str === activeStageKey);
  return (
    <div className="detail-stack">
      <div className="summary-line">
        <StatusDot severity={effectiveSeverity(row)} />
        <div>
          <strong>{action.label_str || row.next_action_str || "No action"}</strong>
          <span>{action.detail_str || shortReason(row)}</span>
        </div>
      </div>
      <section className="stage-inspector" aria-label="Stage Inspector evidence">
        <div className="stage-inspector-header">
          <div>
            <div className="eyebrow">Stage Inspector</div>
            <h2>{activeStep?.label_str || stageLabel(activeStageKey)}</h2>
          </div>
          <Pill>{activeStep?.status_str || stageChipStatus(activeStep, row)}</Pill>
        </div>
        {renderStageEvidence(activeStageKey, detail)}
      </section>
      <StageMap row={row} steps={stepList} selectedStageKey={activeStageKey} onStageSelect={onStageSelect} />
    </div>
  );
}

function TimelineTab({ events }: { events: DebugTimelineEvent[] }) {
  if (!events.length) return <EmptyState text="No debug timeline was returned." />;
  return (
    <div className="timeline-list">
      {events.map((event, index) => (
        <div className={`timeline-row ${severityClass(event.severity_str || event.status_str || "gray")}`} key={`${event.source_str}-${event.label_str}-${index}`}>
          <StatusDot severity={event.severity_str || event.status_str || "gray"} />
          <span>{formatTime(event.timestamp_str)}</span>
          <strong>{event.source_str || "Event"} / {event.label_str || "-"}</strong>
          <small>{event.status_str || "-"}</small>
          <em>{event.detail_str || "-"}</em>
        </div>
      ))}
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

function StageMap({
  row,
  steps,
  selectedStageKey,
  onStageSelect
}: {
  row: PodRow;
  steps: LifecycleStep[];
  selectedStageKey: string;
  onStageSelect: (stageKey: string) => void;
}) {
  const executionSteps = compactStageStepList(steps);
  const diffStep = steps.find((step) => step.step_key_str === "diff");
  if (!executionSteps.length) return <EmptyState text="No stage map was returned." />;
  const currentKey = currentStageKey(executionSteps, row);
  return (
    <div className="stage-map" aria-label="POD lifecycle stage map">
      <div className="stage-map-header">
        <strong>Stage Map</strong>
        <span>{stageSummary(executionSteps, currentKey, row)}</span>
      </div>
      <div className="stage-rail">
        {executionSteps.map((step, index) => {
          const current = step.step_key_str === currentKey;
          const selected = step.step_key_str === selectedStageKey;
          return (
            <React.Fragment key={step.step_key_str || step.label_str}>
              <button
                type="button"
                className={`stage-node ${severityClass(step.severity_str || step.status_str || "gray")} ${current ? "current" : ""} ${selected ? "selected" : ""}`}
                onClick={() => step.step_key_str && onStageSelect(step.step_key_str)}
              >
                <span>{step.label_str || step.step_key_str}</span>
                <strong>{stageChipStatus(step, row)}</strong>
                {current && <small>current</small>}
              </button>
              {index < executionSteps.length - 1 && <ArrowRight className="stage-arrow" size={15} />}
            </React.Fragment>
          );
        })}
      </div>
      {diffStep && (
        <button
          type="button"
          className={`stage-reference ${severityClass(diffStep.severity_str || diffStep.status_str || "gray")} ${selectedStageKey === "diff" ? "selected" : ""}`}
          onClick={() => onStageSelect("diff")}
        >
          <span>DIFF</span>
          <strong>{stageChipStatus(diffStep, row)}</strong>
          <small>{diffStep.detail_str || diffStep.evidence_str || "-"}</small>
        </button>
      )}
    </div>
  );
}

function renderStageEvidence(stageKey: string, detail: PodDetail | null) {
  const row = detail?.pod_row_dict;
  if (!row) return null;
  const report = detail?.latest_execution_report_dict || {};
  const vplan = detail?.latest_vplan_dict || {};
  const decision = detail?.latest_decision_plan_dict || {};
  const eod = row.eod_snapshot_dict || {};
  const diff = detail?.latest_diff_dict || {};
  const stage = normalizeStageKey(stageKey);
  if (stage === "db") {
    return (
      <div className="stage-evidence-grid">
        <Metric label="DB" value={row.db_status_str || "-"} />
        <Metric label="State Time" value={formatTime(row.latest_pod_state_timestamp_str)} />
        <Metric label="Source" value={row.latest_pod_state_source_str || "-"} />
        <Metric label="Stage" value={row.latest_pod_state_stage_str || "-"} />
        <div className="evidence-box wide"><strong>DB path</strong><span>{row.db_path_str || "-"}</span></div>
      </div>
    );
  }
  if (stage === "decision") {
    return (
      <div className="detail-stack">
        <div className="stage-evidence-grid">
          <Metric label="Decision" value={row.latest_decision_plan_status_str || stringField(decision, "status_str")} />
          <Metric label="Submitted" value={formatTime(row.latest_decision_plan_submission_timestamp_str || stringField(decision, "submission_timestamp_str"))} />
          <Metric label="Target Exec" value={formatTime(row.latest_decision_plan_target_execution_timestamp_str || stringField(decision, "target_execution_timestamp_str"))} />
          <Metric label="Book" value={stringField(decision, "decision_book_type_str") || "-"} />
        </div>
        <KeyValuePanel title="Target weights" value={decision.display_target_weight_map_dict} />
      </div>
    );
  }
  if (stage === "vplan") {
    return (
      <div className="detail-stack">
        <div className="stage-evidence-grid">
          <Metric label="VPlan" value={row.latest_vplan_status_str || vplan.status_str || "-"} />
          <Metric label="VPlan ID" value={row.latest_vplan_id_int ?? vplan.vplan_id_int ?? "-"} />
          <Metric label="Submitted" value={formatTime(row.latest_vplan_submission_timestamp_str || vplan.submission_timestamp_str)} />
          <Metric label="Target Exec" value={formatTime(row.latest_vplan_target_execution_timestamp_str || vplan.target_execution_timestamp_str)} />
        </div>
        <EvidenceTable
          title="Order deltas"
          rows={vplan.vplan_row_dict_list || []}
          emptyText="No VPlan rows were returned."
          columns={[
            ["Asset", (item: VPlanRow) => item.asset_str || "-"],
            ["Current", (item: VPlanRow) => formatNumber(item.current_share_float)],
            ["Target", (item: VPlanRow) => formatNumber(item.target_share_float)],
            ["Delta", (item: VPlanRow) => formatNumber(item.order_delta_share_float)],
            ["Ref", (item: VPlanRow) => formatNumber(item.live_reference_price_float)]
          ]}
        />
      </div>
    );
  }
  if (stage === "ack") {
    return (
      <div className="detail-stack">
        <div className="stage-evidence-grid">
          <Metric label="ACK Status" value={row.latest_submit_ack_status_str || vplan.submit_ack_status_str || "-"} />
          <Metric label="ACK Rows" value={row.broker_ack_count_int ?? report.broker_ack_count_int ?? 0} />
          <Metric label="Missing" value={row.missing_ack_count_int ?? vplan.missing_ack_count_int ?? 0} tone={(row.missing_ack_count_int || vplan.missing_ack_count_int || 0) > 0 ? "red" : undefined} />
          <Metric label="Coverage" value={formatPercent(vplan.ack_coverage_ratio_float)} />
        </div>
        <EvidenceTable
          title="Broker ACK evidence"
          rows={vplan.broker_ack_row_dict_list || []}
          emptyText="No broker ACK rows were returned."
          columns={[
            ["Asset", (item: BrokerAckRow) => item.asset_str || "-"],
            ["Status", (item: BrokerAckRow) => item.ack_status_str || "-"],
            ["Source", (item: BrokerAckRow) => item.ack_source_str || "-"],
            ["Broker", (item: BrokerAckRow) => (item.broker_response_ack_bool ? "acked" : "missing")],
            ["Time", (item: BrokerAckRow) => formatTime(item.response_timestamp_str)]
          ]}
        />
      </div>
    );
  }
  if (stage === "fill") {
    return (
      <div className="detail-stack">
        <div className="stage-evidence-grid">
          <Metric label="Fills" value={report.fill_count_int ?? row.fill_count_int ?? 0} />
          <Metric label="Open Coverage" value={`${report.fill_with_official_open_count_int ?? 0} / ${report.fill_count_int ?? row.fill_count_int ?? 0}`} />
          <Metric label="Open Slippage" value={formatBps(report.official_open_slippage_bps_float)} />
          <Metric label="Ref Slippage" value={formatBps(report.vplan_reference_slippage_bps_float)} />
        </div>
        <EvidenceTable
          title="Fills"
          rows={report.fill_row_dict_list || vplan.fill_row_dict_list || []}
          emptyText="No fills were returned for this VPlan."
          columns={[
            ["Asset", (item: FillRow) => item.asset_str || "-"],
            ["Shares", (item: FillRow) => formatNumber(item.fill_amount_float)],
            ["Fill", (item: FillRow) => formatNumber(item.fill_price_float)],
            ["Open", (item: FillRow) => formatNumber(item.official_open_price_float)],
            ["Time", (item: FillRow) => formatTime(item.fill_timestamp_str)]
          ]}
        />
      </div>
    );
  }
  if (stage === "reconcile") {
    return (
      <div className="detail-stack">
        <div className="stage-evidence-grid">
          <Metric label="Reconcile" value={row.latest_reconciliation_status_str || "-"} tone={severityClass(row.latest_reconciliation_status_str || "gray")} />
          <Metric label="Time" value={formatTime(row.latest_reconciliation_timestamp_str)} />
          <Metric label="Residuals" value={report.residual_count_int ?? 0} tone={(report.residual_count_int || 0) > 0 ? "red" : undefined} />
          <Metric label="Reason" value={row.reason_code_str || "-"} />
        </div>
        <EvidenceTable
          title="Model vs broker"
          rows={report.execution_row_dict_list || []}
          emptyText="No execution rows were returned."
          columns={[
            ["Asset", (item: ExecutionRow) => item.asset_str || "-"],
            ["Target", (item: ExecutionRow) => formatNumber(item.target_share_float)],
            ["Broker", (item: ExecutionRow) => formatNumber(item.broker_share_float)],
            ["Residual", (item: ExecutionRow) => formatNumber(item.residual_share_float)],
            ["Order", (item: ExecutionRow) => item.latest_broker_order_status_str || "-"]
          ]}
        />
      </div>
    );
  }
  if (stage === "eod") {
    return (
      <div className="stage-evidence-grid">
        <Metric label="EOD" value={String(eod.status_str || "-")} tone={severityClass(String(eod.severity_str || eod.status_str || "gray"))} />
        <Metric label="Market Date" value={String(eod.latest_market_date_str || eod.expected_market_date_str || "-")} />
        <Metric label="Equity" value={formatMoney(numberField(eod, "equity_float"))} />
        <Metric label="Cash" value={formatMoney(numberField(eod, "cash_float"))} />
        <div className="evidence-box wide"><strong>Detail</strong><span>{String(eod.detail_str || "-")}</span></div>
      </div>
    );
  }
  if (stage === "diff") {
    const htmlUrl = stringField(diff, "html_url_str") || row.latest_diff_artifact_url_str || "";
    return (
      <div className="stage-evidence-grid">
        <Metric label="DIFF" value={String(diff.status_str || row.latest_diff_status_str || "not_run")} />
        <Metric label="Issues" value={numberField(diff, "open_issue_count_int") ?? row.latest_diff_open_issue_count_int ?? "-"} />
        <Metric label="Tracking Error" value={formatNumber(numberField(diff, "equity_tracking_error_float"))} />
        <Metric label="Timestamp" value={formatTime(stringField(diff, "timestamp_str"))} />
        <div className="evidence-box wide">
          <strong>Artifact</strong>
          {htmlUrl ? <a href={htmlUrl} target="_blank" rel="noreferrer">Open DIFF artifact</a> : <span>No artifact yet.</span>}
        </div>
      </div>
    );
  }
  return <EmptyState text="No evidence renderer exists for this stage." />;
}

function EvidenceTable<T>({
  title,
  rows,
  columns,
  emptyText
}: {
  title: string;
  rows: T[];
  columns: Array<[string, (row: T) => React.ReactNode]>;
  emptyText: string;
}) {
  if (!rows.length) return <EmptyState text={emptyText} />;
  return (
    <div className="evidence-table-wrap">
      <strong>{title}</strong>
      <div className="evidence-table" style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(92px, 1fr))` }}>
        {columns.map(([label]) => <span className="evidence-heading" key={label}>{label}</span>)}
        {rows.map((row, rowIndex) => (
          <React.Fragment key={rowIndex}>
            {columns.map(([label, renderFn]) => <span key={`${rowIndex}-${label}`}>{renderFn(row)}</span>)}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function KeyValuePanel({ title, value }: { title: string; value: unknown }) {
  if (!value || typeof value !== "object") return <EmptyState text={`${title} were not returned.`} />;
  return (
    <div className="key-value-panel">
      <strong>{title}</strong>
      {Object.entries(value as Record<string, unknown>).map(([key, entry]) => (
        <div key={key}><span>{key}</span><strong>{String(entry)}</strong></div>
      ))}
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

function ActionStatusStrip({ jobMap, onSelectPod }: { jobMap: Record<string, DashboardJob>; onSelectPod: (podId: string) => void }) {
  const jobList = visibleJobList(jobMap);
  if (!jobList.length) return null;
  const runningCount = jobList.filter((job) => ["queued", "running"].includes(job.status_str)).length;
  const failedCount = jobList.filter((job) => job.status_str === "failed").length;
  return (
    <section className="action-status-strip" aria-label="Action status">
      <div>
        <TerminalSquare size={16} />
        <strong>Action Status</strong>
        <span>{runningCount} running / {failedCount} failed</span>
      </div>
      <div className="action-status-list">
        {jobList.map((job) => (
          <button key={job.job_id_str} className={`job-chip ${jobTone(job)}`} onClick={() => onSelectPod(job.pod_id_str)}>
            <span>{actionLabel(job.action_name_str || "job")}</span>
            <strong>{job.pod_id_str}</strong>
            <small>{job.status_str}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function JobBadge({ job }: { job: DashboardJob }) {
  return <small className={`job-badge ${jobTone(job)}`}>{job.status_str}</small>;
}

function visibleJobList(jobMap: Record<string, DashboardJob>) {
  const jobs = Object.values(jobMap).sort((left, right) => right.created_timestamp_str.localeCompare(left.created_timestamp_str));
  if (!jobs.length) return [];
  const importantJobList = jobs.filter((job) => ["queued", "running", "failed"].includes(job.status_str));
  const mergedJobList = [...importantJobList];
  if (!mergedJobList.some((job) => job.job_id_str === jobs[0].job_id_str)) mergedJobList.push(jobs[0]);
  return mergedJobList.slice(0, 5);
}

function podBadgeJob(jobMap: Record<string, DashboardJob>, podId: string) {
  return Object.values(jobMap)
    .filter((job) => job.pod_id_str === podId && ["queued", "running", "failed"].includes(job.status_str))
    .sort((left, right) => right.created_timestamp_str.localeCompare(left.created_timestamp_str))[0];
}

function jobTone(job: DashboardJob) {
  if (job.status_str === "failed") return "red";
  if (job.status_str === "succeeded") return "green";
  if (["queued", "running"].includes(job.status_str)) return "yellow";
  return "gray";
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

function compactStageStepList(steps: LifecycleStep[]) {
  return steps.filter((step) => step.step_key_str !== "diff");
}

function normalizeStageKey(stageKey?: string | null) {
  const clean = String(stageKey || "").toLowerCase();
  if (clean === "broker_order") return "vplan";
  if (clean === "broker_ack") return "ack";
  if (clean === "fills") return "fill";
  if (clean === "reconciliation") return "reconcile";
  return clean;
}

function stageLabel(stageKey: string) {
  const labelMap: Record<string, string> = {
    db: "DB",
    decision: "Decision",
    vplan: "VPlan",
    ack: "ACK",
    fill: "Fill",
    reconcile: "Reconcile",
    eod: "EOD",
    diff: "DIFF"
  };
  return labelMap[normalizeStageKey(stageKey)] || stageKey;
}

function stageChipStatus(step: LifecycleStep | undefined, row: PodRow) {
  const key = normalizeStageKey(step?.step_key_str);
  if (key === "ack") {
    const missing = row.missing_ack_count_int || 0;
    if (missing > 0) return `missing ${missing}`;
    if ((row.broker_ack_count_int || 0) > 0) return `${row.broker_ack_count_int} ack`;
  }
  if (key === "fill") {
    return `${row.fill_count_int || 0} fill`;
  }
  if (key === "vplan" && row.latest_vplan_status_str) return row.latest_vplan_status_str;
  if (key === "decision" && row.latest_decision_plan_status_str) return row.latest_decision_plan_status_str;
  if (key === "reconcile") {
    const reconcileStatus = row.latest_reconciliation_status_str || step?.status_str;
    if (!reconcileStatus || reconcileStatus === "none") {
      return row.next_action_str === "post_execution_reconcile" ? "pending" : "none";
    }
    return reconcileStatus;
  }
  if (key === "eod" && row.eod_snapshot_dict?.status_str) return String(row.eod_snapshot_dict.status_str);
  if (key === "diff" && row.latest_diff_status_str) return row.latest_diff_status_str;
  return step?.status_str || "-";
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

function currentStageKey(steps: LifecycleStep[], row?: PodRow): string | undefined {
  const redStep = steps.find((step) => severityClass(step.severity_str || step.status_str || "gray") === "red");
  if (redStep?.step_key_str) return redStep.step_key_str;
  const nextActionStage = currentStageFromNextAction(row?.next_action_str);
  if (nextActionStage && steps.some((step) => normalizeStageKey(step.step_key_str) === nextActionStage)) {
    return nextActionStage;
  }
  const yellowStep = steps.find((step) => severityClass(step.severity_str || step.status_str || "gray") === "yellow");
  if (yellowStep?.step_key_str) return yellowStep.step_key_str;
  const waitingStep = steps.find((step) => ["waiting", "blocked_by_execution"].includes(String(step.status_str || "")));
  if (waitingStep?.step_key_str) return waitingStep.step_key_str;
  const activeStep = steps.slice().reverse().find((step) => severityClass(step.severity_str || step.status_str || "gray") === "green");
  return activeStep?.step_key_str || steps[0]?.step_key_str;
}

function currentStageFromNextAction(nextAction?: string | null) {
  const actionStageMap: Record<string, string> = {
    build_decision_plan: "decision",
    build_vplan: "vplan",
    review_vplan: "vplan",
    submit_vplan: "vplan",
    post_execution_reconcile: "reconcile",
    eod_snapshot: "eod"
  };
  return actionStageMap[String(nextAction || "")];
}

function stageSummary(steps: LifecycleStep[], currentKey?: string, row?: PodRow) {
  const currentStep = steps.find((step) => step.step_key_str === currentKey);
  if (!currentStep) return "no current stage";
  return `${currentStep.label_str || currentStep.step_key_str}: ${row ? stageChipStatus(currentStep, row) : currentStep.status_str || "-"}`;
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
    compare_reference: "DIFF",
    job: "Job",
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

function formatNumber(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(value);
}

function formatBps(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${formatNumber(value)} bps`;
}

function formatPercent(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${formatNumber(value * 100)}%`;
}

function stringField(value: unknown, key: string) {
  const item = value && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined;
  return item == null || item === "" ? undefined : String(item);
}

function numberField(value: unknown, key: string) {
  const item = value && typeof value === "object" ? (value as Record<string, unknown>)[key] : undefined;
  return typeof item === "number" && !Number.isNaN(item) ? item : undefined;
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
