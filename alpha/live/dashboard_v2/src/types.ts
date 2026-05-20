export type Severity = "green" | "yellow" | "red" | "gray" | string;

export interface DashboardSummary {
  as_of_timestamp_str: string;
  pod_row_dict_list: PodRow[];
  alert_dict_list: DashboardAlert[];
  alert_summary_dict: Record<string, number>;
  mode_list: string[];
  combined_book_dict?: Record<string, unknown>;
}

export interface ActionToken {
  action_token_str: string;
}

export interface DashboardAlert {
  pod_id_str: string;
  mode_str: string;
  account_route_str: string;
  alert_type_str: string;
  severity_str: Severity;
  label_str: string;
  reason_str: string;
}

export interface DebugSummary {
  severity_str?: Severity;
  verdict_label_str?: string;
  primary_reason_str?: string;
  primary_evidence_str?: string;
  latest_evidence_timestamp_str?: string | null;
  next_inspect_command_name_str?: string;
}

export interface RequiredAction {
  label_str?: string;
  severity_str?: Severity;
  detail_str?: string;
  inspect_command_name_str?: string;
  context_item_dict_list?: Array<Record<string, string>>;
}

export interface LifecycleStep {
  step_key_str?: string;
  label_str?: string;
  status_str?: string;
  severity_str?: Severity;
  detail_str?: string;
  evidence_str?: string;
}

export interface DebugTimelineEvent {
  source_str?: string;
  label_str?: string;
  status_str?: string;
  severity_str?: Severity;
  timestamp_str?: string | null;
  detail_str?: string;
}

export interface DebugStory {
  verdict_dict?: Record<string, unknown>;
  blocker_dict_list?: Array<Record<string, unknown>>;
  timeline_event_dict_list?: DebugTimelineEvent[];
  evidence_item_dict_list?: Array<Record<string, unknown>>;
  recommended_command_dict_list?: Array<Record<string, unknown>>;
}

export interface VPlanRow {
  asset_str?: string;
  current_share_float?: number | null;
  target_share_float?: number | null;
  order_delta_share_float?: number | null;
  live_reference_price_float?: number | null;
  reason_str?: string | null;
}

export interface BrokerOrderRow {
  asset_str?: string;
  status_str?: string;
  broker_order_type_str?: string;
  amount_float?: number | null;
  filled_amount_float?: number | null;
  avg_fill_price_float?: number | null;
  submitted_timestamp_str?: string | null;
  last_status_timestamp_str?: string | null;
}

export interface BrokerAckRow {
  asset_str?: string;
  order_request_key_str?: string;
  broker_order_type_str?: string;
  broker_response_ack_bool?: boolean;
  ack_status_str?: string;
  ack_source_str?: string;
  response_timestamp_str?: string | null;
}

export interface FillRow {
  asset_str?: string;
  broker_order_id_str?: string;
  fill_amount_float?: number | null;
  fill_price_float?: number | null;
  fill_timestamp_str?: string | null;
  official_open_price_float?: number | null;
  open_price_source_str?: string | null;
  slippage_bps_float?: number | null;
  slippage_notional_float?: number | null;
}

export interface ExecutionRow {
  asset_str?: string;
  side_str?: string;
  planned_order_delta_share_float?: number | null;
  filled_share_float?: number | null;
  current_share_float?: number | null;
  target_share_float?: number | null;
  broker_share_float?: number | null;
  residual_share_float?: number | null;
  unresolved_bool?: boolean;
  fill_price_float?: number | null;
  official_open_price_float?: number | null;
  vplan_reference_price_float?: number | null;
  official_open_slippage_bps_float?: number | null;
  vplan_reference_slippage_bps_float?: number | null;
  latest_broker_order_status_str?: string | null;
}

export interface ExecutionReport {
  latest_vplan_id_int?: number | null;
  fill_count_int?: number;
  fill_with_official_open_count_int?: number;
  broker_order_count_int?: number;
  broker_ack_count_int?: number;
  residual_count_int?: number;
  official_open_slippage_bps_float?: number | null;
  official_open_slippage_notional_float?: number | null;
  vplan_reference_slippage_bps_float?: number | null;
  vplan_reference_slippage_notional_float?: number | null;
  fill_row_dict_list?: FillRow[];
  execution_row_dict_list?: ExecutionRow[];
}

export interface VPlanDetail {
  vplan_id_int?: number | null;
  status_str?: string;
  pod_id_str?: string;
  account_route_str?: string;
  submission_timestamp_str?: string | null;
  target_execution_timestamp_str?: string | null;
  submit_ack_status_str?: string | null;
  ack_coverage_ratio_float?: number | null;
  missing_ack_count_int?: number;
  vplan_row_dict_list?: VPlanRow[];
  broker_order_row_dict_list?: BrokerOrderRow[];
  broker_ack_row_dict_list?: BrokerAckRow[];
  fill_row_dict_list?: FillRow[];
}

export interface FreshnessItem {
  label_str?: string;
  status_str?: string;
  severity_str?: Severity;
  detail_str?: string;
  timestamp_str?: string | null;
}

export interface DataFreshness {
  status_str?: string;
  severity_str?: Severity;
  item_dict_list?: FreshnessItem[];
  pod_state_updated_timestamp_str?: string | null;
  norgate_status_str?: string;
}

export interface RehearsalStatus {
  status_str?: string;
  promotion_gate_status_str?: string;
  last_cycle_status_str?: string;
  completed_cycle_count_int?: number;
  sim_ledger_status_str?: string;
  ibkr_reference_status_str?: string;
  ibkr_open_price_status_str?: string;
  detail_str?: string;
}

export interface PodRow {
  pod_id_str: string;
  mode_str: string;
  account_route_str: string;
  strategy_import_str: string;
  db_status_str?: string;
  health_str?: Severity;
  next_action_str?: string;
  reason_code_str?: string;
  equity_float?: number | null;
  cash_float?: number | null;
  position_count_int?: number | null;
  warning_count_int?: number | null;
  latest_event_timestamp_str?: string | null;
  latest_pod_state_timestamp_str?: string | null;
  latest_pod_state_source_str?: string | null;
  latest_pod_state_stage_str?: string | null;
  latest_broker_snapshot_timestamp_str?: string | null;
  latest_decision_plan_status_str?: string | null;
  latest_decision_plan_submission_timestamp_str?: string | null;
  latest_decision_plan_target_execution_timestamp_str?: string | null;
  latest_vplan_status_str?: string | null;
  latest_vplan_id_int?: number | null;
  latest_vplan_submission_timestamp_str?: string | null;
  latest_vplan_target_execution_timestamp_str?: string | null;
  latest_submit_ack_status_str?: string | null;
  latest_diff_status_str?: string | null;
  latest_diff_open_issue_count_int?: number | null;
  latest_diff_artifact_url_str?: string | null;
  latest_reconciliation_status_str?: string | null;
  latest_reconciliation_timestamp_str?: string | null;
  broker_ack_count_int?: number;
  fill_count_int?: number;
  broker_order_count_int?: number;
  missing_ack_count_int?: number;
  exception_count_int?: number;
  db_path_str?: string;
  eod_snapshot_dict?: Record<string, unknown>;
  data_freshness_dict?: DataFreshness;
  rehearsal_status_dict?: RehearsalStatus;
  required_action_dict?: RequiredAction;
  lifecycle_step_dict_list?: LifecycleStep[];
  debug_summary_dict?: DebugSummary;
}

export interface PodDetail {
  pod_row_dict: PodRow;
  required_action_dict?: RequiredAction;
  lifecycle_step_dict_list?: LifecycleStep[];
  data_freshness_dict?: DataFreshness;
  eod_snapshot_dict?: Record<string, unknown>;
  rehearsal_status_dict?: RehearsalStatus;
  debug_story_dict?: DebugStory;
  pod_pnl_dict?: Record<string, unknown>;
  latest_decision_plan_dict?: Record<string, unknown> | null;
  latest_vplan_dict?: VPlanDetail | null;
  latest_execution_report_dict?: ExecutionReport | null;
  event_dict_list?: EventRow[];
  latest_diff_dict?: Record<string, unknown>;
}

export interface EventRow {
  timestamp_str?: string;
  event_timestamp_str?: string;
  created_timestamp_str?: string;
  level_str?: string;
  severity_str?: string;
  event_name_str?: string;
  reason_str?: string;
  message_str?: string;
  error_str?: string;
}

export interface DashboardJob {
  job_id_str: string;
  pod_id_str: string;
  mode_str: string;
  action_name_str?: string | null;
  status_str: "queued" | "running" | "succeeded" | "failed" | string;
  created_timestamp_str: string;
  started_timestamp_str?: string | null;
  completed_timestamp_str?: string | null;
  result_dict?: Record<string, unknown> | null;
  error_str?: string | null;
  traceback_str?: string | null;
}
