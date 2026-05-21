import React, { useEffect, useMemo, useRef, useState } from "react";
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
  PauseCircle,
  RefreshCcw,
  TerminalSquare,
  X
} from "lucide-react";
import "./styles.css";
import type {
  ActionToken,
  BrokerAckRow,
  CombinedBook,
  CombinedBookEnvironment,
  DashboardJob,
  DashboardSummary,
  DecisionPlanDetail,
  DebugTimelineEvent,
  EquityPoint,
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
type DetailTab = "stage" | "execution" | "equity" | "timeline" | "logs" | "raw";
type KeyValueKind = "plain" | "weight";

export function mergeStablePodDetailPayload(
  incomingDetail: PodDetail,
  existingDetail: PodDetail | null
): PodDetail {
  if (!existingDetail) return incomingDetail;
  const mergedDetail = { ...incomingDetail };
  if (sameDecisionPlanBool(incomingDetail, existingDetail)) {
    const existingDecision = existingDetail.latest_decision_plan_dict || null;
    const incomingDecision = incomingDetail.latest_decision_plan_dict || null;
    if (existingDecision && incomingDecision) {
      mergedDetail.latest_decision_plan_dict = mergeDecisionPlanDetail(
        incomingDecision,
        existingDecision
      );
    } else if (existingDecision && !incomingDecision) {
      mergedDetail.latest_decision_plan_dict = existingDecision;
    }
  }
  if (sameVPlanBool(incomingDetail, existingDetail) && !incomingDetail.latest_vplan_dict) {
    mergedDetail.latest_vplan_dict = existingDetail.latest_vplan_dict;
  }
  if (
    sameExecutionReportBool(incomingDetail, existingDetail)
    && !incomingDetail.latest_execution_report_dict
  ) {
    mergedDetail.latest_execution_report_dict = existingDetail.latest_execution_report_dict;
  }
  return mergedDetail;
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
  const summaryRef = useRef<DashboardSummary | null>(null);
  const detailRequestSeqRef = useRef(0);

  useEffect(() => {
    void refreshSummary();
    const intervalId = window.setInterval(() => void refreshSummary(), REFRESH_MS);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    summaryRef.current = summary;
  }, [summary]);

  useEffect(() => {
    const onPopState = () => setViewName(currentViewName());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    if (!selectedPodId) return;
    let activeBool = true;
    void loadPodDetail(selectedPodId, () => activeBool, true);
    const intervalId = window.setInterval(
      () => void loadPodDetail(selectedPodId, () => activeBool, false),
      REFRESH_MS
    );
    return () => {
      activeBool = false;
      window.clearInterval(intervalId);
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

  async function loadPodDetail(
    podId: string,
    shouldApplyFn: () => boolean = () => true,
    showLoadingBool = true
  ) {
    const requestSeqInt = detailRequestSeqRef.current + 1;
    detailRequestSeqRef.current = requestSeqInt;
    const shouldApplyResponse = () => (
      shouldApplyFn() && requestSeqInt === detailRequestSeqRef.current
    );
    if (showLoadingBool) setDetailLoading(true);
    try {
      const payload = await fetchJson<PodDetail>(`/api/pods/${encodeURIComponent(podId)}`);
      if (shouldApplyResponse()) {
        setSelectedDetail((existingDetail) => mergeStablePodDetailPayload(payload, existingDetail));
      }
    } catch (error) {
      if (shouldApplyResponse()) {
        const fallbackRow = rowByPodId(summaryRef.current?.pod_row_dict_list || [], podId) || {
            pod_id_str: podId,
            mode_str: "unknown",
            account_route_str: "-",
            strategy_import_str: "-"
        };
        setSelectedDetail((existingDetail) => existingDetail || {
          pod_row_dict: fallbackRow,
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
      if (showLoadingBool && shouldApplyResponse()) setDetailLoading(false);
    }
  }

  function selectPod(row: PodRow, stageKey?: string, tabName?: DetailTab) {
    const samePodBool = selectedPodId === row.pod_id_str;
    if (!samePodBool) {
      setSelectedDetail(null);
      detailRequestSeqRef.current += 1;
    } else if (!selectedDetail) {
      void loadPodDetail(row.pod_id_str);
    }
    setDetailTab(tabName || tabForStage(stageKey));
    setSelectedStageKey(stageKey || currentStageKey(currentCycleStepList(row, row.lifecycle_step_dict_list || []), row) || "summary");
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
  const liveRows = useMemo(() => podRows.filter((row) => row.mode_str === "live"), [podRows]);
  const paperRows = useMemo(() => podRows.filter((row) => row.mode_str === "paper"), [podRows]);
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
            liveRows={liveRows}
            paperRows={paperRows}
            attentionRows={attentionRows}
            combinedBook={summary?.combined_book_dict}
            jobMap={jobMap}
            onSelect={selectPod}
            expandedPodId={selectedPodId}
            selectedDetail={selectedDetail}
            detailLoading={detailLoading}
            detailTab={detailTab}
            selectedStageKey={selectedStageKey}
            onTab={setDetailTab}
            onStageSelect={(stageKey) => {
              setSelectedStageKey(stageKey);
              setDetailTab(tabForStage(stageKey));
            }}
            onCloseDetail={() => {
              setSelectedPodId(null);
              setSelectedDetail(null);
            }}
            onAction={startAction}
          />
        ) : (
          <IncubationHome
            rows={incubationRows}
            attentionRows={attentionRows}
            jobMap={jobMap}
            onSelect={selectPod}
            expandedPodId={selectedPodId}
            selectedDetail={selectedDetail}
            detailLoading={detailLoading}
            detailTab={detailTab}
            selectedStageKey={selectedStageKey}
            onTab={setDetailTab}
            onStageSelect={(stageKey) => {
              setSelectedStageKey(stageKey);
              setDetailTab(tabForStage(stageKey));
            }}
            onCloseDetail={() => {
              setSelectedPodId(null);
              setSelectedDetail(null);
            }}
            onAction={startAction}
          />
        )}
      </main>
    </div>
  );
}

function OpsHome({
  liveRows,
  paperRows,
  attentionRows,
  combinedBook,
  jobMap,
  onSelect,
  expandedPodId,
  selectedDetail,
  detailLoading,
  detailTab,
  selectedStageKey,
  onTab,
  onStageSelect,
  onCloseDetail,
  onAction
}: {
  liveRows: PodRow[];
  paperRows: PodRow[];
  attentionRows: AttentionRow[];
  combinedBook?: CombinedBook;
  jobMap: Record<string, DashboardJob>;
  onSelect: (row: PodRow, stageKey?: string, tabName?: DetailTab) => void;
  expandedPodId: string | null;
  selectedDetail: PodDetail | null;
  detailLoading: boolean;
  detailTab: DetailTab;
  selectedStageKey: string;
  onTab: (tab: DetailTab) => void;
  onStageSelect: (stageKey: string) => void;
  onCloseDetail: () => void;
  onAction: (row: PodRow, actionName: string) => void;
}) {
  return (
    <div className="ops-stack">
      <section className="surface attention-surface">
        <SectionTitle icon={<ListChecks size={18} />} title="Attention Queue" aside={`${attentionRows.length} items`} />
        {attentionRows.length === 0 ? (
          <EmptyState text="No live or paper POD needs attention." />
        ) : (
          <div className="attention-list">
            {attentionRows.map((item) => (
              <AttentionQueueItem
                key={item.row.pod_id_str}
                item={item}
                onOpen={() => onSelect(item.row, item.stageKey, tabForStage(item.stageKey))}
              />
            ))}
          </div>
        )}
      </section>

      <PodSection
        title="LIVE PODs"
        rows={liveRows}
        emptyText="No LIVE PODs are enabled."
        modeSummary={modeEnvironment(combinedBook, "live")}
        jobMap={jobMap}
        expandedPodId={expandedPodId}
        selectedDetail={selectedDetail}
        detailLoading={detailLoading}
        detailTab={detailTab}
        selectedStageKey={selectedStageKey}
        onSelect={onSelect}
        onTab={onTab}
        onStageSelect={onStageSelect}
        onCloseDetail={onCloseDetail}
        onAction={onAction}
      />

      <PodSection
        title="PAPER PODs"
        rows={paperRows}
        emptyText="No PAPER PODs are enabled."
        modeSummary={modeEnvironment(combinedBook, "paper")}
        jobMap={jobMap}
        expandedPodId={expandedPodId}
        selectedDetail={selectedDetail}
        detailLoading={detailLoading}
        detailTab={detailTab}
        selectedStageKey={selectedStageKey}
        onSelect={onSelect}
        onTab={onTab}
        onStageSelect={onStageSelect}
        onCloseDetail={onCloseDetail}
        onAction={onAction}
      />
    </div>
  );
}

function IncubationHome({
  rows,
  attentionRows,
  jobMap,
  onSelect,
  expandedPodId,
  selectedDetail,
  detailLoading,
  detailTab,
  selectedStageKey,
  onTab,
  onStageSelect,
  onCloseDetail,
  onAction
}: {
  rows: PodRow[];
  attentionRows: AttentionRow[];
  jobMap: Record<string, DashboardJob>;
  onSelect: (row: PodRow, stageKey?: string, tabName?: DetailTab) => void;
  expandedPodId: string | null;
  selectedDetail: PodDetail | null;
  detailLoading: boolean;
  detailTab: DetailTab;
  selectedStageKey: string;
  onTab: (tab: DetailTab) => void;
  onStageSelect: (stageKey: string) => void;
  onCloseDetail: () => void;
  onAction: (row: PodRow, actionName: string) => void;
}) {
  return (
    <div className="ops-stack">
      <section className="surface">
        <SectionTitle icon={<ListChecks size={18} />} title="Promotion Queue" aside={`${attentionRows.length} items`} />
        <div className="attention-list">
          {attentionRows.map((item) => (
            <AttentionQueueItem key={item.row.pod_id_str} item={item} onOpen={() => onSelect(item.row, item.stageKey)} />
          ))}
          {!attentionRows.length && <EmptyState text="No rehearsal state to review." />}
        </div>
      </section>
      <PodSection
        title="Incubation Rehearsal"
        rows={rows}
        emptyText="No incubation PODs are enabled."
        jobMap={jobMap}
        expandedPodId={expandedPodId}
        selectedDetail={selectedDetail}
        detailLoading={detailLoading}
        detailTab={detailTab}
        selectedStageKey={selectedStageKey}
        onSelect={onSelect}
        onTab={onTab}
        onStageSelect={onStageSelect}
        onCloseDetail={onCloseDetail}
        onAction={onAction}
        incubation
      />
    </div>
  );
}

function AttentionQueueItem({ item, onOpen }: { item: AttentionRow; onOpen: () => void }) {
  return (
    <button className={`attention-item ${severityClass(item.severity)}`} onClick={onOpen}>
      <StatusDot severity={item.severity} />
      <div className="attention-pod">
        <strong>{item.row.pod_id_str}</strong>
        <span>{item.row.mode_str} / {item.stageLabel}</span>
      </div>
      <div className="attention-reason">
        <strong>{item.state}</strong>
        <span>{item.reason}</span>
      </div>
      <div className="attention-evidence">
        <small>{item.evidence}</small>
        <small>{item.timing}</small>
      </div>
      <div className="attention-age">
        <small>{item.freshness}</small>
        <strong>Open</strong>
      </div>
      <ArrowRight size={16} />
    </button>
  );
}

function PodSection({
  title,
  rows,
  emptyText,
  modeSummary,
  jobMap,
  expandedPodId,
  selectedDetail,
  detailLoading,
  detailTab,
  selectedStageKey,
  onSelect,
  onTab,
  onStageSelect,
  onCloseDetail,
  onAction,
  incubation
}: {
  title: string;
  rows: PodRow[];
  emptyText: string;
  modeSummary?: CombinedBookEnvironment;
  jobMap: Record<string, DashboardJob>;
  expandedPodId: string | null;
  selectedDetail: PodDetail | null;
  detailLoading: boolean;
  detailTab: DetailTab;
  selectedStageKey: string;
  onSelect: (row: PodRow, stageKey?: string, tabName?: DetailTab) => void;
  onTab: (tab: DetailTab) => void;
  onStageSelect: (stageKey: string) => void;
  onCloseDetail: () => void;
  onAction: (row: PodRow, actionName: string) => void;
  incubation?: boolean;
}) {
  const selectedRow = rows.find((row) => row.pod_id_str === expandedPodId);
  return (
    <section className="surface mode-section">
      <SectionTitle icon={incubation ? <ClipboardList size={18} /> : <Activity size={18} />} title={title} aside={`${rows.length} pods`} />
      {modeSummary && <ModeEquitySummary environment={modeSummary} />}
      <div className={`pod-workbench ${selectedRow ? "has-detail" : ""}`}>
        <div className="pod-table">
          {rows.map((row) => (
            <PodRowCard
              key={row.pod_id_str}
              row={row}
              job={podBadgeJob(jobMap, row.pod_id_str)}
              expanded={expandedPodId === row.pod_id_str}
              onSelect={() => onSelect(row)}
              onStageSelect={(stageKey) => onSelect(row, stageKey, tabForStage(stageKey))}
              incubation={incubation}
            />
          ))}
          {!rows.length && <EmptyState text={emptyText} />}
        </div>
        {selectedRow && (
          <InlinePodDetail
            detail={selectedDetail}
            fallbackRow={selectedRow}
            loading={detailLoading}
            tab={detailTab}
            onTab={onTab}
            onClose={onCloseDetail}
            selectedStageKey={selectedStageKey}
            onStageSelect={onStageSelect}
            onAction={onAction}
          />
        )}
      </div>
    </section>
  );
}

function PodRowCard({
  row,
  incubation,
  job,
  expanded,
  onSelect,
  onStageSelect
}: {
  row: PodRow;
  incubation?: boolean;
  job?: DashboardJob;
  expanded?: boolean;
  onSelect: () => void;
  onStageSelect: (stageKey: string) => void;
}) {
  const action = row.required_action_dict || {};
  const severity = effectiveSeverity(row);
  return (
    <div className={`pod-row-card ${severityClass(severity)} ${expanded ? "expanded" : ""}`}>
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
    </div>
  );
}

function MiniStageRail({ row, steps, onStageSelect }: { row: PodRow; steps: LifecycleStep[]; onStageSelect: (stageKey: string) => void }) {
  const executionSteps = compactStageStepList(currentCycleStepList(row, steps));
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

function InlinePodDetail({
  detail,
  fallbackRow,
  loading,
  tab,
  onTab,
  onClose,
  selectedStageKey,
  onStageSelect,
  onAction
}: {
  detail: PodDetail | null;
  fallbackRow: PodRow;
  loading: boolean;
  tab: DetailTab;
  onTab: (tab: DetailTab) => void;
  onClose: () => void;
  selectedStageKey: string;
  onStageSelect: (stageKey: string) => void;
  onAction: (row: PodRow, actionName: string) => void;
}) {
  const row = detail?.pod_row_dict || fallbackRow;
  const [controlsEnabled, setControlsEnabled] = useState(false);

  useEffect(() => {
    setControlsEnabled(false);
  }, [row?.pod_id_str]);

  return (
    <section className="pod-inline-detail" aria-label="Inline POD detail">
      <div className="inline-detail-header">
        <div className="inline-detail-title">
          <div className="eyebrow">Current Cycle Workbench</div>
          <h2>{row.pod_id_str}</h2>
          <span>{row.mode_str} / {row.account_route_str || "-"}</span>
        </div>
        <button className="icon-button" onClick={onClose} title="Collapse">
          <X size={18} />
        </button>
      </div>
      {loading && (
        <div className="inline-loading">
          <Loader2 className="spin" size={18} /> Loading latest state
        </div>
      )}
      <div className="inline-control-gate">
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
      <div className={`inline-actions ${controlsEnabled ? "" : "disabled"}`}>
        <button disabled={!controlsEnabled} onClick={() => onAction(row, "compare_reference")}>
          Run DIFF
        </button>
        {OPS_ACTION_LIST.map((actionName) => (
          <button key={actionName} disabled={!controlsEnabled} onClick={() => onAction(row, actionName)}>
            {actionLabel(actionName)}
          </button>
        ))}
      </div>
      <div className="inline-detail-tabs">
        {(["stage", "execution", "equity", "timeline", "logs", "raw"] as DetailTab[]).map((tabName) => (
          <button key={tabName} className={tab === tabName ? "active" : ""} onClick={() => onTab(tabName)}>
            {tabName === "stage" ? "Stage" : tabName === "timeline" ? "Operator Log" : tabName}
          </button>
        ))}
      </div>
      <div className="inline-detail-body">
        {tab === "stage" && <StageInspector detail={detail || { pod_row_dict: row }} selectedStageKey={selectedStageKey} onStageSelect={onStageSelect} />}
        {tab === "execution" && <ExecutionReceipt detail={detail} />}
        {tab === "equity" && <EquityPanel detail={detail} />}
        {tab === "timeline" && <OperatorLogTab events={detail?.debug_story_dict?.timeline_event_dict_list || []} />}
        {tab === "logs" && <LogsTab events={detail?.event_dict_list || []} />}
        {tab === "raw" && <RawTab detail={detail || { pod_row_dict: row }} />}
      </div>
    </section>
  );
}

function ExecutionReceipt({ detail }: { detail: PodDetail | null }) {
  const report = detail?.latest_execution_report_dict;
  const rowList = report?.execution_row_dict_list || [];
  if (!report) return <EmptyState text="No execution report was returned for this POD yet." />;
  return (
    <div className="detail-stack execution-receipt" aria-label="Execution receipt">
      <div className="stage-evidence-grid">
        <Metric label="Fills" value={report.fill_count_int ?? 0} />
        <Metric label="ACK Coverage" value={`${report.broker_ack_count_int ?? 0} / ${report.broker_order_count_int ?? 0}`} />
        <Metric label="Residuals" value={report.residual_count_int ?? 0} tone={(report.residual_count_int || 0) > 0 ? "red" : undefined} />
        <Metric label="Open Coverage" value={`${report.fill_with_official_open_count_int ?? 0} / ${report.fill_count_int ?? 0}`} />
        <Metric label="BPS vs Ref" value={formatBps(report.vplan_reference_slippage_bps_float)} tone={costTone(report.vplan_reference_slippage_bps_float)} />
        <Metric label="BPS vs Open" value={formatBps(report.official_open_slippage_bps_float)} tone={costTone(report.official_open_slippage_bps_float)} />
        <Metric label="$ vs Ref" value={formatMoney(report.vplan_reference_slippage_notional_float)} tone={costTone(report.vplan_reference_slippage_notional_float)} />
        <Metric label="$ vs Open" value={formatMoney(report.official_open_slippage_notional_float)} tone={costTone(report.official_open_slippage_notional_float)} />
      </div>
      <div className="subtle-box">
        <strong>Cost convention</strong>
        <span>Positive BPS means worse execution cost; negative BPS means price improvement.</span>
      </div>
      <EvidenceTable
        title="Asset receipt"
        rows={rowList}
        emptyText="No per-asset execution rows were returned."
        columns={[
          ["Asset", (item: ExecutionRow) => item.asset_str || "-"],
          ["Side", (item: ExecutionRow) => item.side_str || "-"],
          ["Planned", (item: ExecutionRow) => formatNumber(item.planned_order_delta_share_float)],
          ["Filled", (item: ExecutionRow) => formatNumber(item.filled_share_float)],
          ["Fill Px", (item: ExecutionRow) => formatNumber(item.fill_price_float)],
          ["Ref Px", (item: ExecutionRow) => formatNumber(item.vplan_reference_price_float)],
          ["Open Px", (item: ExecutionRow) => formatNumber(item.official_open_price_float)],
          ["BPS Ref", (item: ExecutionRow) => <span className={costTone(item.vplan_reference_slippage_bps_float)}>{formatBps(item.vplan_reference_slippage_bps_float)}</span>],
          ["BPS Open", (item: ExecutionRow) => <span className={costTone(item.official_open_slippage_bps_float)}>{formatBps(item.official_open_slippage_bps_float)}</span>],
          ["$ Ref", (item: ExecutionRow) => <span className={costTone(item.vplan_reference_slippage_notional_float)}>{formatMoney(item.vplan_reference_slippage_notional_float)}</span>],
          ["$ Open", (item: ExecutionRow) => <span className={costTone(item.official_open_slippage_notional_float)}>{formatMoney(item.official_open_slippage_notional_float)}</span>],
          ["Order", (item: ExecutionRow) => item.latest_broker_order_status_str || "-"]
        ]}
      />
    </div>
  );
}

function EquityPanel({ detail }: { detail: PodDetail | null }) {
  const pnl = detail?.pod_pnl_dict;
  const pointList = pnl?.equity_point_dict_list || [];
  const pointCount = pointList.length;
  if (!pnl || pointCount === 0) {
    return (
      <div className="equity-panel" aria-label="Equity panel">
        <EmptyState text="No EOD equity samples yet. The equity curve appears after the first EOD snapshot is written." />
      </div>
    );
  }
  return (
    <div className="detail-stack equity-panel" aria-label="Equity panel">
      <div className="stage-evidence-grid">
        <Metric label="EOD Points" value={pnl.point_count_int ?? pointCount} />
        <Metric label="Latest Equity" value={formatMoney(pnl.latest_equity_float)} />
        <Metric label="Daily PnL" value={formatPnl(pnl.daily_pnl_float, pnl.daily_pnl_pct_float)} tone={pnlTone(pnl.daily_pnl_float)} />
        <Metric label="Since Start" value={formatPnl(pnl.since_start_pnl_float, pnl.since_start_pnl_pct_float)} tone={pnlTone(pnl.since_start_pnl_float)} />
      </div>
      {pointCount === 1 ? (
        <div className="subtle-box">
          <strong>One EOD sample</strong>
          <span>Curve starts after the next EOD sample. Latest date: {pointList[0]?.market_date_str || "-"}</span>
        </div>
      ) : (
        <Sparkline pointList={pointList} />
      )}
      <EvidenceTable
        title="EOD equity samples"
        rows={pointList.slice(-8)}
        emptyText="No EOD points were returned."
        columns={[
          ["Date", (item: EquityPoint) => item.market_date_str || "-"],
          ["Equity", (item: EquityPoint) => formatMoney(item.equity_float)],
          ["Cash", (item: EquityPoint) => formatMoney(item.cash_float)],
          ["Daily", (item: EquityPoint) => formatPnl(item.daily_pnl_float, item.daily_pnl_pct_float)],
          ["Since Start", (item: EquityPoint) => formatPnl(item.since_start_pnl_float, item.since_start_pnl_pct_float)]
        ]}
      />
    </div>
  );
}

function ModeEquitySummary({ environment }: { environment: CombinedBookEnvironment }) {
  const pointCount = environment.carry_forward_point_count_int ?? environment.strict_point_count_int ?? environment.equity_point_dict_list?.length ?? 0;
  return (
    <div className="mode-summary">
      <Metric label="Book Equity" value={formatMoney(environment.latest_equity_float)} />
      <Metric label="Daily PnL" value={formatPnl(environment.daily_pnl_float, environment.daily_pnl_pct_float)} tone={pnlTone(environment.daily_pnl_float)} />
      <Metric label="Since Start" value={formatPnl(environment.since_start_pnl_float, environment.since_start_pnl_pct_float)} tone={pnlTone(environment.since_start_pnl_float)} />
      <Metric label="EOD Points" value={pointCount} />
    </div>
  );
}

function Sparkline({ pointList }: { pointList: EquityPoint[] }) {
  const valueList = pointList
    .map((point) => point.equity_float)
    .filter((value): value is number => typeof value === "number" && !Number.isNaN(value));
  if (valueList.length < 2) return <EmptyState text="Need at least two valid equity points for a curve." />;
  const minValue = Math.min(...valueList);
  const maxValue = Math.max(...valueList);
  const range = maxValue - minValue || 1;
  const pathStr = valueList
    .map((value, index) => {
      const x = (index / Math.max(1, valueList.length - 1)) * 100;
      const y = 46 - ((value - minValue) / range) * 40;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <div className="sparkline-wrap" aria-label="POD equity curve">
      <svg viewBox="0 0 100 52" role="img">
        <path d={pathStr} />
      </svg>
      <div>
        <strong>{formatMoney(valueList[valueList.length - 1])}</strong>
        <span>{pointList[0]?.market_date_str || "-"} to {pointList[pointList.length - 1]?.market_date_str || "-"}</span>
      </div>
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
  const rawStepList = detail?.lifecycle_step_dict_list || row.lifecycle_step_dict_list || [];
  const stepList = currentCycleStepList(row, rawStepList);
  const currentKey = currentStageKey(compactStageStepList(stepList), row);
  const activeStageKey = normalizeStageKey(selectedStageKey) || currentKey || stepList[0]?.step_key_str || "summary";
  const activeStep = stepList.find((step) => step.step_key_str === activeStageKey);
  return (
    <div className="detail-stack">
      <div className="summary-line">
        <StatusDot severity={effectiveSeverity(row)} />
        <div>
          <strong>{action.label_str || row.next_action_str || "No action"}</strong>
          <span>{action.detail_str || action.reason_str || shortReason(row)}</span>
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
      <PreviousCycleReceipt detail={detail} />
    </div>
  );
}

function OperatorLogTab({ events }: { events: DebugTimelineEvent[] }) {
  if (!events.length) return <EmptyState text="No operator log timeline was returned." />;
  return (
    <div className="detail-stack" aria-label="Operator Log">
      <div className="subtle-box">
        <strong>Read-only operator log</strong>
        <span>Timeline is rebuilt from local dashboard evidence and JSONL events. It does not call Norgate or mutate live state.</span>
      </div>
      <EvidenceTable
        title="Operator timeline"
        rows={events}
        emptyText="No operator events were returned."
        columns={[
          ["Time", (event: DebugTimelineEvent) => formatTime(event.timestamp_str)],
          ["Phase", (event: DebugTimelineEvent) => `${event.source_str || "Event"} / ${event.label_str || "-"}`],
          ["Status", (event: DebugTimelineEvent) => <span className={severityClass(event.severity_str || event.status_str || "gray")}>{event.status_str || "-"}</span>],
          ["Cycle", (event: DebugTimelineEvent) => cycleLabel(event)],
          ["Reason", (event: DebugTimelineEvent) => event.detail_str || "-"]
        ]}
      />
    </div>
  );
}

function cycleLabel(event: DebugTimelineEvent) {
  const decision = event.decision_plan_id_int ?? null;
  const vplan = event.vplan_id_int ?? null;
  const partList = [];
  if (decision !== null) partList.push(`plan=${decision}`);
  if (vplan !== null) partList.push(`vplan=${vplan}`);
  if (event.cycle_role_str) partList.push(String(event.cycle_role_str));
  return partList.length ? partList.join(" / ") : "-";
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
        <div>
          <div className="eyebrow">Stage Map</div>
          <strong>Current Cycle</strong>
        </div>
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
  const report: ExecutionReport = detail?.latest_execution_report_dict || {};
  const vplan: VPlanDetail = detail?.latest_vplan_dict || {};
  const decision = detail?.latest_decision_plan_dict || {};
  const eod = row.eod_snapshot_dict || {};
  const diff = detail?.latest_diff_dict || {};
  const stage = normalizeStageKey(stageKey);
  if (previousCycleBool(row) && ["vplan", "ack", "fill", "reconcile"].includes(stage)) {
    return <CurrentCycleEmptyEvidence stageKey={stage} row={row} />;
  }
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
        <div className="subtle-box">
          <strong>Current planned cycle</strong>
          <span>{currentDecisionCycleText(row, decision)}</span>
        </div>
        <div className="stage-evidence-grid">
          <Metric label="Decision ID" value={decision.decision_plan_id_int ?? row.latest_decision_plan_id_int ?? "-"} />
          <Metric label="Decision" value={row.latest_decision_plan_status_str || stringField(decision, "status_str")} />
          <Metric label="Signal" value={formatTime(stringField(decision, "signal_timestamp_str"))} />
          <Metric label="Submitted" value={formatTime(row.latest_decision_plan_submission_timestamp_str || stringField(decision, "submission_timestamp_str"))} />
          <Metric label="Target Exec" value={formatTime(row.latest_decision_plan_target_execution_timestamp_str || stringField(decision, "target_execution_timestamp_str"))} />
          <Metric label="Book" value={stringField(decision, "decision_book_type_str") || "-"} />
          <Metric label="Policy" value={stringField(decision, "execution_policy_str") || "-"} />
          <Metric label="Snapshot" value={snapshotLabel(decision)} />
        </div>
        <DecisionIntentPanel decision={decision} />
        <KeyValuePanel title="Decision base positions" value={decision.decision_base_position_map_dict} />
        <KeyValuePanel title="Snapshot metadata" value={decision.snapshot_metadata_dict} />
      </div>
    );
  }
  if (stage === "vplan") {
    return (
      <div className="detail-stack">
        <div className="stage-evidence-grid">
          <Metric label="VPlan" value={row.latest_vplan_status_str || vplan.status_str || "-"} />
          <Metric label="VPlan ID" value={row.latest_vplan_id_int ?? vplan.vplan_id_int ?? "-"} />
          <Metric label="Decision ID" value={row.latest_vplan_decision_plan_id_int ?? vplan.decision_plan_id_int ?? "-"} />
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
          <Metric label="Fill records" value={report.fill_count_int ?? row.fill_count_int ?? 0} />
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
    const htmlUrl = safeDashboardHref(stringField(diff, "html_url_str") || row.latest_diff_artifact_url_str || "");
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

function currentDecisionCycleText(row: PodRow, decision: DecisionPlanDetail) {
  const planId = decision.decision_plan_id_int ?? row.latest_decision_plan_id_int ?? "-";
  const status = row.latest_decision_plan_status_str || decision.status_str || "-";
  const target = formatTime(row.latest_decision_plan_target_execution_timestamp_str || decision.target_execution_timestamp_str);
  if (previousCycleBool(row)) {
    return `DecisionPlan ${planId} is ${status} for ${target}. No VPlan for this current cycle has been built yet.`;
  }
  return `DecisionPlan ${planId} is ${status} for ${target}.`;
}

function snapshotLabel(decision: DecisionPlanDetail) {
  const metadata = decision.snapshot_metadata_dict || {};
  const date = stringField(metadata, "norgate_snapshot_date_str") || stringField(metadata, "snapshot_date_str");
  const profile = stringField(metadata, "norgate_data_profile_str") || stringField(metadata, "data_profile_str");
  if (date && profile) return `${profile} / ${date}`;
  return date || profile || "-";
}

function CurrentCycleEmptyEvidence({ stageKey, row }: { stageKey: string; row: PodRow }) {
  const planId = row.latest_decision_plan_id_int ?? "-";
  const previousPlanId = row.latest_vplan_decision_plan_id_int ?? "-";
  return (
    <div className="subtle-box current-cycle-empty">
      <strong>No current-cycle {stageLabel(stageKey)} yet</strong>
      <span>Current DecisionPlan is {planId}. The latest VPlan evidence belongs to previous DecisionPlan {previousPlanId}, so it is shown below as audit history.</span>
    </div>
  );
}

function PreviousCycleReceipt({ detail }: { detail: PodDetail | null }) {
  const row = detail?.pod_row_dict;
  if (!row || !previousCycleBool(row)) return null;
  const planId = row.latest_decision_plan_id_int ?? "-";
  const vplanId = row.latest_vplan_id_int ?? "-";
  const previousPlanId = row.latest_vplan_decision_plan_id_int ?? "-";
  const report = detail?.latest_execution_report_dict || {};
  return (
    <section className="previous-cycle-receipt" aria-label="Previous Cycle Receipt">
      <div className="previous-cycle-header">
        <div>
          <div className="eyebrow">Previous Cycle Receipt</div>
          <h2>Historical execution evidence</h2>
        </div>
        <Pill>not current</Pill>
      </div>
      <div className="previous-cycle-note">
        VPlan {vplanId} belongs to DecisionPlan {previousPlanId}. Current DecisionPlan is {planId}, so these fills and ACKs are audit history, not today&apos;s progress.
      </div>
      <div className="stage-evidence-grid previous-cycle-grid">
        <Metric label="Prev Decision" value={previousPlanId} />
        <Metric label="Prev VPlan" value={vplanId} />
        <Metric label="VPlan" value={row.latest_vplan_status_str || "-"} />
        <Metric label="Target Exec" value={formatTime(row.latest_vplan_target_execution_timestamp_str)} />
        <Metric label="ACK" value={`${row.broker_ack_count_int ?? report.broker_ack_count_int ?? 0} / ${row.broker_order_count_int ?? report.broker_order_count_int ?? 0}`} />
        <Metric label="Fills" value={report.fill_count_int ?? row.fill_count_int ?? 0} />
        <Metric label="Reconcile" value={row.latest_reconciliation_status_str || "-"} />
        <Metric label="Residuals" value={report.residual_count_int ?? 0} tone={(report.residual_count_int || 0) > 0 ? "red" : undefined} />
      </div>
    </section>
  );
}

function sameDecisionPlanBool(incomingDetail: PodDetail, existingDetail: PodDetail) {
  const incomingPlanId = incomingDetail.latest_decision_plan_dict?.decision_plan_id_int
    ?? incomingDetail.pod_row_dict.latest_decision_plan_id_int;
  const existingPlanId = existingDetail.latest_decision_plan_dict?.decision_plan_id_int
    ?? existingDetail.pod_row_dict.latest_decision_plan_id_int;
  return sameStableIdBool(incomingPlanId, existingPlanId);
}

function sameVPlanBool(incomingDetail: PodDetail, existingDetail: PodDetail) {
  const incomingVPlanId = incomingDetail.latest_vplan_dict?.vplan_id_int
    ?? incomingDetail.pod_row_dict.latest_vplan_id_int;
  const existingVPlanId = existingDetail.latest_vplan_dict?.vplan_id_int
    ?? existingDetail.pod_row_dict.latest_vplan_id_int;
  return sameStableIdBool(incomingVPlanId, existingVPlanId);
}

function sameExecutionReportBool(incomingDetail: PodDetail, existingDetail: PodDetail) {
  const incomingVPlanId = incomingDetail.latest_execution_report_dict?.latest_vplan_id_int
    ?? incomingDetail.pod_row_dict.latest_vplan_id_int;
  const existingVPlanId = existingDetail.latest_execution_report_dict?.latest_vplan_id_int
    ?? existingDetail.pod_row_dict.latest_vplan_id_int;
  return sameStableIdBool(incomingVPlanId, existingVPlanId);
}

function sameStableIdBool(left: unknown, right: unknown) {
  if (left === null || left === undefined || right === null || right === undefined) return false;
  return String(left) === String(right);
}

function mergeDecisionPlanDetail(
  incomingDecision: DecisionPlanDetail,
  existingDecision: DecisionPlanDetail
): DecisionPlanDetail {
  const mergedDecision = { ...existingDecision, ...incomingDecision };
  if (!hasRecordEntriesBool(incomingDecision.target_weight_map_dict)) {
    mergedDecision.target_weight_map_dict = existingDecision.target_weight_map_dict;
  }
  if (!hasRecordEntriesBool(incomingDecision.entry_target_weight_map_dict)) {
    mergedDecision.entry_target_weight_map_dict = existingDecision.entry_target_weight_map_dict;
  }
  if (!hasRecordEntriesBool(incomingDecision.full_target_weight_map_dict)) {
    mergedDecision.full_target_weight_map_dict = existingDecision.full_target_weight_map_dict;
  }
  if (!hasRecordEntriesBool(incomingDecision.display_target_weight_map_dict)) {
    mergedDecision.display_target_weight_map_dict = existingDecision.display_target_weight_map_dict;
  }
  if (!hasRecordEntriesBool(incomingDecision.decision_base_position_map_dict)) {
    mergedDecision.decision_base_position_map_dict = existingDecision.decision_base_position_map_dict;
  }
  if (!hasRecordEntriesBool(incomingDecision.snapshot_metadata_dict)) {
    mergedDecision.snapshot_metadata_dict = existingDecision.snapshot_metadata_dict;
  }
  if (!hasRecordEntriesBool(incomingDecision.strategy_state_dict)) {
    mergedDecision.strategy_state_dict = existingDecision.strategy_state_dict;
  }
  if (!incomingDecision.exit_asset_list?.length) {
    mergedDecision.exit_asset_list = existingDecision.exit_asset_list;
  }
  if (!incomingDecision.entry_priority_list?.length) {
    mergedDecision.entry_priority_list = existingDecision.entry_priority_list;
  }
  return mergedDecision;
}

function safeDashboardHref(rawHref?: string | null) {
  const href = String(rawHref || "").trim();
  if (!href) return "";
  if (href.startsWith("/") && !href.startsWith("//")) return href;
  try {
    const url = new URL(href, window.location.origin);
    if (url.protocol === "http:" || url.protocol === "https:") return url.href;
  } catch {
    return "";
  }
  return "";
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

function DecisionIntentPanel({ decision }: { decision: DecisionPlanDetail }) {
  const bookType = decision.decision_book_type_str || "-";
  const exitAssetList = decision.exit_asset_list || [];
  const entryTargetEntryList = recordEntryList(preferredEntryTargetMap(decision));
  const fullTargetEntryList = recordEntryList(preferredFullTargetMap(decision));
  const showFullTargetBool = bookType === "full_target_weight_book" || fullTargetEntryList.length > 0;
  const primaryTargetEntryList = bookType === "full_target_weight_book" ? fullTargetEntryList : entryTargetEntryList;
  const primaryTargetTitle = bookType === "full_target_weight_book" ? "Full target book" : "Entry targets";
  const emptyBool = !exitAssetList.length && !primaryTargetEntryList.length && !(showFullTargetBool && fullTargetEntryList.length);
  return (
    <section className="decision-intent-panel" aria-label="Decision Intent">
      <div className="decision-intent-header">
        <div>
          <div className="eyebrow">Decision Intent</div>
          <h2>{bookType}</h2>
        </div>
        <Pill>{exitAssetList.length} exits / {primaryTargetEntryList.length} targets</Pill>
      </div>
      {exitAssetList.length > 0 && (
        <div className="intent-section">
          <strong>Assets to exit</strong>
          <div className="asset-chip-list">
            {exitAssetList.map((asset) => <span className="asset-chip exit" key={asset}>{asset}</span>)}
          </div>
        </div>
      )}
      {primaryTargetEntryList.length > 0 && (
        <IntentTargetTable title={primaryTargetTitle} entryList={primaryTargetEntryList} />
      )}
      {bookType !== "full_target_weight_book" && fullTargetEntryList.length > 0 && (
        <IntentTargetTable title="Full targets" entryList={fullTargetEntryList} />
      )}
      {emptyBool && <EmptyState text="No decision intent assets were returned for this current cycle." />}
    </section>
  );
}

function IntentTargetTable({
  title,
  entryList
}: {
  title: string;
  entryList: Array<[string, unknown]>;
}) {
  return (
    <div className="intent-section">
      <strong>{title}</strong>
      <div className="intent-target-table">
        {entryList.map(([asset, weight]) => (
          <div className="intent-target-row" key={asset}>
            <span>{asset}</span>
            <strong>{formatWeightPercent(numericValueOrNull(weight))}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function preferredEntryTargetMap(decision: DecisionPlanDetail) {
  if (hasRecordEntriesBool(decision.entry_target_weight_map_dict)) {
    return decision.entry_target_weight_map_dict;
  }
  if (decision.decision_book_type_str !== "full_target_weight_book") {
    return decision.display_target_weight_map_dict;
  }
  return undefined;
}

function preferredFullTargetMap(decision: DecisionPlanDetail) {
  if (hasRecordEntriesBool(decision.full_target_weight_map_dict)) {
    return decision.full_target_weight_map_dict;
  }
  if (decision.decision_book_type_str === "full_target_weight_book") {
    return decision.display_target_weight_map_dict;
  }
  return undefined;
}

function KeyValuePanel({
  title,
  value,
  valueKind = "plain"
}: {
  title: string;
  value: unknown;
  valueKind?: KeyValueKind;
}) {
  const entryList = recordEntryList(value);
  if (!entryList.length) return <EmptyState text={`${title} were not returned.`} />;
  return (
    <div className="key-value-panel">
      <strong>{title}</strong>
      {entryList.map(([key, entry]) => (
        <div key={key}><span>{key}</span><strong>{formatKeyValueEntry(entry, valueKind)}</strong></div>
      ))}
    </div>
  );
}

function hasRecordEntriesBool(value: unknown) {
  return recordEntryList(value).length > 0;
}

function recordEntryList(value: unknown): Array<[string, unknown]> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value as Record<string, unknown>);
}

function formatKeyValueEntry(value: unknown, valueKind: KeyValueKind) {
  const numericValue = numericValueOrNull(value);
  if (valueKind === "weight" && numericValue !== null) return formatPercent(numericValue);
  if (typeof value === "number") return formatNumber(value);
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "object") return compactJson(value);
  return String(value);
}

function numericValueOrNull(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const numericValue = Number(value);
    if (Number.isFinite(numericValue)) return numericValue;
  }
  return null;
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
  stageKey: string;
  stageLabel: string;
  evidence: string;
  timing: string;
  freshness: string;
}

function buildAttentionRows(rows: PodRow[]): AttentionRow[] {
  return rows
    .map((row) => {
      const stageKey = currentStageKey(row.lifecycle_step_dict_list || [], row) || "summary";
      return {
        row,
        group: queueGroup(row),
        severity: effectiveSeverity(row),
        state: stateSentence(row),
        reason: shortReason(row),
        stageKey,
        stageLabel: stageLabel(stageKey),
        evidence: attentionEvidence(row),
        timing: attentionTiming(row),
        freshness: attentionFreshness(row)
      };
    })
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

function attentionEvidence(row: PodRow) {
  const ack = row.missing_ack_count_int ? `ACK missing ${row.missing_ack_count_int}` : `ACK ${row.broker_ack_count_int || 0}`;
  const fills = `Fill ${row.fill_count_int || 0}`;
  const issues = row.latest_diff_open_issue_count_int ? `DIFF ${row.latest_diff_open_issue_count_int}` : "DIFF ok";
  return `${ack} / ${fills} / ${issues}`;
}

function attentionTiming(row: PodRow) {
  const target = row.latest_vplan_target_execution_timestamp_str || row.latest_decision_plan_target_execution_timestamp_str;
  if (target) return `Target ${formatTime(target)}`;
  if (row.latest_event_timestamp_str) return `Event ${formatTime(row.latest_event_timestamp_str)}`;
  return "Target -";
}

function attentionFreshness(row: PodRow) {
  const eod = row.eod_snapshot_dict || {};
  const freshness = row.data_freshness_dict || {};
  const eodStatus = stringField(eod, "status_str");
  const podTime = freshness.pod_state_updated_timestamp_str || row.latest_pod_state_timestamp_str || row.latest_event_timestamp_str;
  return eodStatus ? `EOD ${eodStatus}` : `State ${formatTime(podTime)}`;
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

function currentCycleStepList(row: PodRow, steps: LifecycleStep[]) {
  if (!previousCycleBool(row)) return steps;
  return steps.map((step) => {
    const key = normalizeStageKey(step.step_key_str);
    if (key === "vplan") {
      return {
        ...step,
        status_str: "not_built",
        severity_str: "gray",
        detail_str: "No VPlan has been built for the current DecisionPlan yet."
      };
    }
    if (["ack", "fill", "reconcile"].includes(key)) {
      return {
        ...step,
        status_str: "none",
        severity_str: "gray",
        detail_str: "Latest evidence belongs to a previous cycle."
      };
    }
    return step;
  });
}

function previousCycleBool(row: PodRow) {
  return row.latest_vplan_cycle_role_str === "previous";
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

function tabForStage(stageKey?: string): DetailTab {
  const stage = normalizeStageKey(stageKey);
  if (stage === "ack" || stage === "fill") return "execution";
  return "stage";
}

function stageChipStatus(step: LifecycleStep | undefined, row: PodRow) {
  const key = normalizeStageKey(step?.step_key_str);
  if (
    previousCycleBool(row)
    && ["vplan", "ack", "fill", "reconcile"].includes(key)
    && ["not_built", "none"].includes(String(step?.status_str || ""))
  ) {
    return step?.status_str === "not_built" ? "not built" : "none";
  }
  const previousPrefix = row.latest_vplan_cycle_role_str === "previous" ? "prev " : "";
  if (key === "ack") {
    const missing = row.missing_ack_count_int || 0;
    if (missing > 0) return `${previousPrefix}missing ${missing}`;
    if ((row.broker_ack_count_int || 0) > 0) return `${previousPrefix}${row.broker_ack_count_int} ack`;
  }
  if (key === "fill") {
    return `${previousPrefix}${row.fill_count_int || 0} fill records`;
  }
  if (key === "vplan" && row.latest_vplan_status_str) return `${previousPrefix}${row.latest_vplan_status_str}`;
  if (key === "decision" && row.latest_decision_plan_status_str) return row.latest_decision_plan_status_str;
  if (key === "reconcile") {
    const reconcileStatus = row.latest_reconciliation_status_str || step?.status_str;
    if (!reconcileStatus || reconcileStatus === "none") {
      return row.next_action_str === "post_execution_reconcile" ? "pending" : "none";
    }
    return `${previousPrefix}${reconcileStatus}`;
  }
  if (key === "eod" && row.eod_snapshot_dict?.status_str) return String(row.eod_snapshot_dict.status_str);
  if (key === "diff" && row.latest_diff_status_str) return row.latest_diff_status_str;
  return step?.status_str || "-";
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
  return row.debug_summary_dict?.primary_reason_str || row.required_action_dict?.detail_str || row.required_action_dict?.reason_str || row.reason_code_str || "-";
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

function modeEnvironment(combinedBook: CombinedBook | undefined, mode: string) {
  return combinedBook?.environment_dict_list?.find((environment) => environment.mode_str === mode);
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

function formatWeightPercent(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value * 100) + "%";
}

function formatPnl(value?: number | null, pct?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const pctText = typeof pct === "number" && !Number.isNaN(pct) ? ` (${formatPercent(pct)})` : "";
  return `${formatMoney(value)}${pctText}`;
}

function costTone(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value) || value === 0) return undefined;
  return value > 0 ? "red" : "green";
}

function pnlTone(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value) || value === 0) return undefined;
  return value > 0 ? "green" : "red";
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
