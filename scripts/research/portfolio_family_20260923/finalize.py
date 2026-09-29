"""Close the research artifacts and hash their complete reviewable evidence."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from scripts.research.portfolio_family_20260923.freeze import ROOT_PATH, STUDY_PATH, sha256_str


def main() -> None:
    now_str=datetime.now(timezone.utc).isoformat()
    state_path=STUDY_PATH/"research_state.json"
    state_dict=json.loads(state_path.read_text(encoding="utf-8"))
    if state_dict["adaptive_search"]["actual_total_variants"] != 318:
        raise ValueError("Final adaptive round must be reconciled before closeout")
    for relative_str in ("REPORT.html","REPORT_FULL.html","decision_notebook.ipynb","verification/report_review.md",
                         "verification/analysis_review.md","verification/core_review.md","verification/html_render_check.json"):
        if not (STUDY_PATH/relative_str).is_file():
            raise FileNotFoundError(relative_str)
    render_dict=json.loads((STUDY_PATH/"verification/html_render_check.json").read_text(encoding="utf-8"))
    if not render_dict["pass"]:
        raise ValueError("HTML rendering checks must pass")
    minutes_int=int((datetime.now(timezone.utc)-datetime.fromisoformat(state_dict["created_at"])).total_seconds()/60)
    state_dict["phase"]="complete"
    state_dict["updated_at"]=now_str
    state_dict["runtime_budget"]["active_minutes_used"]=minutes_int
    state_dict["final_decision"]={"disposition":"promising_component","research_status":"forward_hypothesis",
        "verdict":"Use the preselected coarse CORE5 risk family as explanatory research models; local gates and costs are descriptive on seen history, with no deployment authority.",
        "next_gate":"Combined-book physical execution at client capital and common data vintage; liquidity/borrow checks and genuinely prospective frozen evidence."}
    state_dict["adaptive_search"]["stop_reason"]="Original300 and final18 adaptive cells complete; no further search or fine tuning; historical evidence cannot provide an untouched holdout."
    state_path.write_text(json.dumps(state_dict,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    knowledge_path=STUDY_PATH/"knowledge_record.json"
    knowledge_dict=json.loads(knowledge_path.read_text(encoding="utf-8"))
    knowledge_dict["adaptive_lineage"].update({"rounds_completed":1,"actual_total_variants":318,"declared_total_variants":318,
        "active_minutes_used":minutes_int,"stop_reason":state_dict["adaptive_search"]["stop_reason"]})
    knowledge_dict["last_reviewed_at"]=now_str
    knowledge_dict["research_status"]=state_dict["final_decision"]["research_status"]
    knowledge_dict["disposition"]=state_dict["final_decision"]["disposition"]
    knowledge_path.write_text(json.dumps(knowledge_dict,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    decision_path=STUDY_PATH/"decision_log.jsonl"
    existing_list=[json.loads(line_str) for line_str in decision_path.read_text(encoding="utf-8").splitlines() if line_str.strip()]
    if not any(record_dict.get("event_id")=="D_final_closeout" for record_dict in existing_list):
        event_dict={"schema_version":"quant-research-decision-event-v1","study_id":state_dict["study_id"],"event_id":"D_final_closeout",
            "phase":"complete","recorded_at":now_str,"decision":"finish_frozen_research_and_publish_local_report",
            "reason":"All318 evaluation cells completed; no additional weight search; transparent mechanisms and residual implementation gaps delivered.",
            "evidence_paths":["REPORT.html","REPORT_FULL.html","verification/report_review.md","tables/frozen_gate_results.csv"],
            "holdout_consequence":"None remains untouched; all findings at most forward hypotheses.","active_minutes_used":minutes_int}
        with decision_path.open("a",encoding="utf-8",newline="\n") as stream_obj:
            stream_obj.write(json.dumps(event_dict,ensure_ascii=False,separators=(",",":"))+"\n")
    def record_dict(file_path: Path, relative_bool: bool) -> dict:
        return {"path":file_path.relative_to(STUDY_PATH).as_posix() if relative_bool else str(file_path.resolve()),
                "bytes":file_path.stat().st_size,"sha256":sha256_str(file_path)}
    file_list=[record_dict(file_path,True) for file_path in sorted(STUDY_PATH.rglob("*"))
        if file_path.is_file() and file_path.name not in ("run_manifest.json","run_stdout.log","run_stderr.log")]
    external_list=[record_dict(file_path,False) for file_path in sorted((ROOT_PATH/"scripts/research/portfolio_family_20260923").glob("*")) if file_path.is_file()]
    external_list.extend(record_dict(file_path,False) for file_path in sorted((ROOT_PATH/"tests").glob("test_portfolio_family*.py")))
    manifest_dict={"schema_version":"quant-research-manifest-v1","study_id":state_dict["study_id"],"created_at":now_str,
        "scientific_spec_sha256":sha256_str(STUDY_PATH/"research_spec_frozen.json"),"research_only":True,
        "files":file_list,"external_deliverables_and_sources":external_list,
        "source_lineage":"Nested source_manifest, benchmark_manifest and replay_manifest retain original input paths/hashes; no raw source overwritten."}
    (STUDY_PATH/"run_manifest.json").write_text(json.dumps(manifest_dict,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"closed_local_research","files":len(file_list),"external":len(external_list),"minutes":minutes_int,"actual_cells":318}))


if __name__=="__main__":
    main()
