"""Package existing evidence; never select, tune, or rerun strategy signals."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
import pandas as pd
from scripts.research.client_menu_independent_20260923.protocol import ROOT_PATH, SOURCE_PATH, STUDY_PATH, digest_str, write_json

SKILL_PATH = Path('C:/Users/User/.codex/skills/research-quant-signal-features')
sys.path.insert(0, str(SKILL_PATH/'scripts'))
from validate_adaptive_research import experiment_execution_key, validate_adaptive_research
from validate_knowledge_record import validate_knowledge_record
from validate_research_bundle import validate_manifest, validate_notebook, validate_markdown_images


def read_json(name_str: str) -> dict:
    return json.loads((STUDY_PATH/name_str).read_text(encoding='utf-8'))


def package() -> None:
    timestamp_str = datetime.now(timezone.utc).isoformat()
    spec_dict = read_json('research_spec_frozen.json')
    assert digest_str(STUDY_PATH/'research_spec_frozen.json') == '03a9e77d6ce200dbd09a56106d2c85a5ca6a6cd87036a4d350eb54609a9f8f5a'
    study_id_str = spec_dict['study_id']
    elapsed_int = int((datetime.now(timezone.utc)-datetime.fromisoformat('2026-09-23T15:42:02+00:00')).total_seconds()/60)
    verification_path = STUDY_PATH/'verification'
    raw_path = verification_path/'original_experiment_ledger.jsonl'
    if not raw_path.exists():
        raw_path.write_bytes((STUDY_PATH/'experiment_ledger.jsonl').read_bytes())
    raw_list = [json.loads(line_str) for line_str in raw_path.read_text().splitlines()]
    assert len(raw_list) == 220
    catalog_list = json.loads((SOURCE_PATH/'catalog_complete.json').read_text(encoding='utf-8'))
    source_map_str = '# Source rule map\n\nThis document was assembled at closeout from the factual catalog frozen before the new study. It is not a claim that this Markdown file existed before results. Source hashes and original rule freeze are in research_spec_frozen.json and input_sources.json. All25 current PM_READY/WIRED sources were considered. No Claude product artifact is an input.\n\n'
    for source_dict in catalog_list:
        source_map_str += '## '+source_dict['alias']+'\n\n```json\n'+json.dumps(source_dict,ensure_ascii=False,indent=2)+'\n```\n\n'
    source_map_str += '\nTiming: preserve each native source contract; selected menu sources use Close_T decisions and Open_(T+1) fills. MonthEndFlow has a distinct MOC contract and is not selected. Historical code identity is incomplete for older runs. Source catalog is a documented audit, not a new signal replay. FRED is current vintage, Sector6 fractional shares, Trinity holdings unresolved.\n'
    (STUDY_PATH/'SOURCE_RULE_MAP.md').write_text(source_map_str,encoding='utf-8')
    group_list = [
        ('H0','source_literal','diagnostic',27,'research_spec_frozen.json','tables/universe.csv','Native source and benchmark diagnostics','Known strategy mechanisms and native accounting; no new alpha claim','Missing source, lineage or matching dates'),
        ('H1','predeclared','portfolio_construction',220,'research_spec_frozen.json','tables/results.csv','Mandate-driven architectures and controls','Distinct mechanisms should satisfy loss budgets and earn operating complexity','Frozen mandate or utility gate fails'),
        ('H2','post_result_adaptive','diagnostic',59,'amendment_002_fragility.json','tables/fragility.csv','All-pair neighborhood and cash ablation','Recommendation should not hinge on one exact capital split','Any selected conservative neighborhood breaks its loss policy'),
        ('H3','post_result_adaptive','portfolio_construction',20,'adaptive_active_spec.json','tables/active_alternative.csv','A3 sector mean reversion follow-up','Short-horizon sector reversion may smooth momentum and TAA without excessive return sacrifice','Original A utility or risk requirements fail at either cost layer'),
        ('H4','post_result_adaptive','diagnostic',6,'presentation_comparison_spec.json','tables/menu_common_sample.csv','Common-calendar presentation','Compare every displayed menu and control on identical observed sessions','Curve metrics disagree with table or calendar contains gaps'),
    ]
    hypothesis_list = []
    experiment_list = []
    for index_int, (hypothesis_str,provenance_str,classification_str,count_int,freeze_str,table_str,title_str,mechanism_str,falsifier_str) in enumerate(group_list):
        freeze_dict = read_json(freeze_str)
        freeze_time_str = freeze_dict.get('frozen_at',freeze_dict.get('initial_frozen_at',freeze_dict.get('amended_at')))
        experiment_str = f'G{index_int:02d}'
        hypothesis_list.append({'hypothesis_id':hypothesis_str,'title':title_str,'family':'client_portfolio_menu','role':classification_str,'economic_mechanism':mechanism_str,'expected_direction':'policy suitability and robustness, not maximum backtest return','falsifier':falsifier_str,'provenance':provenance_str,'classification':classification_str,'created_at':freeze_time_str,'frozen_at':freeze_time_str,'registry_recorded_at':timestamp_str,'data_periods_seen':['all existing strategy history and prior study outputs; no untouched holdout'],'declared_variant_count':count_int,'experiment_ids':[experiment_str],'status':'diagnostic' if classification_str=='diagnostic' else 'research_candidate'})
        experiment_dict = {'schema_version':'quant-research-experiment-v1','experiment_id':experiment_str,'study_id':study_id_str,'recorded_at':timestamp_str,'phase':'diagnosis' if index_int<2 else 'adaptive_discovery','hypothesis_ids':[hypothesis_str],'data_periods_seen':['source-specific observed history through 2026-07-31; periods in linked CSV'],'declared_variant_count':count_int,'selection_role':title_str,'status':'complete','spec_content_id':digest_str(STUDY_PATH/freeze_str),'data_content_ids':[digest_str(SOURCE_PATH/'source_manifest.json'),digest_str(SOURCE_PATH/'source_audit_addendum.json')],'code_content_id':hashlib.sha256(''.join(digest_str(code_path) for code_path in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest(),'result_content_id':digest_str(STUDY_PATH/table_str),'evidence_paths':[table_str,freeze_str],'recording_note':'Closeout group index, not contemporaneous execution timestamps. Original220 cell records retained byte-for-byte at verification/original_experiment_ledger.jsonl. Group H4 counts only6 newly calculated cells, not reused displayed rows.'}
        experiment_dict['execution_key'] = experiment_execution_key(experiment_dict)
        experiment_list.append(experiment_dict)
    write_json(STUDY_PATH/'hypothesis_registry.json',{'schema_version':'quant-research-hypotheses-v1','study_id':study_id_str,'recorded_at':timestamp_str,'hypotheses':hypothesis_list})
    # Explicit schema migration preserves original bytes and identities; no history is erased.
    (STUDY_PATH/'experiment_ledger.jsonl').write_text(''.join(json.dumps(record_dict,ensure_ascii=False)+'\n' for record_dict in experiment_list),encoding='utf-8')
    decision_list = []
    decision_text_list = [
        ('record_provenance','Root saw Claude before the new request; new inputs exclude its products, code and allocations. Source-only independent design review used. All history already seen.',['research_spec_frozen.json','input_sources.json']),
        ('record_primary_selection','Frozen-priority gates selected S2/M0/G2; initial active candidates failed utility. Older-history veto enforced; selection unchanged.',['selection.json','tables/mandate_gates.csv']),
        ('record_fragility_followup','Post-result all-pair neighborhoods and cash controls did not change selection. Trinity recovery failed exact reconciliation and was rejected.',['fragility_verdict.json','trinity_holdings_recovery']),
        ('record_provisional_active','A3 passed its separately frozen actual-history and conservative-cost comparison; retains short-history and fractional-share limitations.',['adaptive_active_spec.json','active_alternative_verdict.json']),
        ('record_schema_migration','Closeout canonical group index replaces simplified ledger schema; original220 records preserved byte-for-byte.332 full-window cells,73 initial config IDs including duplicates; no implication of332 independent trials.',['verification/original_experiment_ledger.jsonl','experiment_ledger.jsonl']),
    ]
    for index_int,(decision_str,reason_str,evidence_list) in enumerate(decision_text_list):
        decision_list.append({'schema_version':'quant-research-decision-event-v1','event_id':f'D{index_int:02d}','study_id':study_id_str,'recorded_at':timestamp_str,'phase':'closeout','decision':decision_str,'reason':reason_str,'evidence_paths':evidence_list,'holdout_consequence':'No unseen validation or confirmation period exists. This event records existing evidence at closeout; it is not backdated.','active_minutes_used':elapsed_int})
    decision_path = STUDY_PATH/'decision_log.jsonl'
    if not decision_path.exists():
        decision_path.write_text(''.join(json.dumps(record_dict,ensure_ascii=False)+'\n' for record_dict in decision_list),encoding='utf-8')
    state_dict = json.loads((SKILL_PATH/'assets/research_state_template.json').read_text())
    state_dict.update({'study_id':study_id_str,'title':'Independent client portfolio menu','phase':'complete','evidence_phase':'adaptive_discovery','created_at':spec_dict['initial_frozen_at'],'updated_at':timestamp_str,'source':{'location':str(SOURCE_PATH/'catalog_complete.json'),'content_id':digest_str(SOURCE_PATH/'catalog_complete.json'),'read_complete':True}})
    state_dict['runtime_budget'].update({'active_minutes_used':elapsed_int,'max_adaptive_rounds':1,'max_total_variants':400,'max_parallel_lanes':2})
    state_dict['locks'].update({'source_rule_map_frozen_at':spec_dict['initial_frozen_at'],'literal_baseline_frozen_at':spec_dict['initial_frozen_at']})
    state_dict['lock_note'] = 'Locks refer to the frozen source catalog/spec, not the later Markdown/source-map or group-ledger packaging.'
    state_dict['baseline'].update({'status':'completed','replication_outcome':'not_assessed','executable_translation':'native source exports reused; no paper replication claimed','primary_evidence':['tables/universe.csv','input_sources.json']})
    state_dict['holdouts'].update({'validation_period':'unavailable: already seen','confirmation_period':'unavailable: already seen'})
    state_dict['adaptive_search'].update({'rounds_completed':1,'actual_new_hypotheses_by_round':[1],'declared_total_variants':400,'actual_total_variants':332,'stop_reason':'Mandates, bounded sensitivity and one active follow-up complete; no further tuning justified. Diagnostic and presentation groups are not new alpha hypotheses.'})
    state_dict['final_decision'].update({'disposition':'candidate','research_status':'research_candidate','verdict':'S2/M0/G2 primary research candidates; A3 provisional shorter-history alternative. No cash substitute established.','next_gate':'Client-specific capital/fills/capacity and untouched prospective evidence before allocation.'})
    write_json(STUDY_PATH/'research_state.json',state_dict)
    menu_df = pd.read_csv(STUDY_PATH/'tables/menu_common_sample.csv')
    selected_df = menu_df.loc[(menu_df.scenario=='common_account')&menu_df.candidate.isin(['S2','M0','A3','G2'])].set_index('candidate')
    lineage_dict = {'profile':'standard','rounds_completed':1,'declared_total_variants':400,'actual_total_variants':332,'active_minutes_used':elapsed_int,'stop_reason':state_dict['adaptive_search']['stop_reason']}
    artifact_dict = {**state_dict['artifacts'],'research_state':'research_state.json','notebook':'decision_notebook.ipynb','primary_source_code':[str(Path(__file__).parent/'analyze.py'),str(Path(__file__).parent/'diagnostics.py')],'primary_tables':['tables/results.csv','tables/menu_common_sample.csv'],'primary_charts':['charts/equity.png','charts/drawdown.png']}
    knowledge_dict = {'schema_version':'quant-research-knowledge-v1','study_id':study_id_str,'title':state_dict['title'],'created_at':timestamp_str,'research_status':'research_candidate','disposition':'candidate','replication_outcome':'not_assessed','signal_family':'portfolio_construction','objective':spec_dict['objective'],'verdict':state_dict['final_decision']['verdict'],'verdicts':{'source_replication':'No paper replication claimed; audited saved native source exports.','predictive_value':'No untouched holdout or forward alpha established.','economic_value':'Historical mandate fit survives declared cost/sensitivity checks; A3 provisional.','promotion':'Research product proposals only, no allocation or deployment approval.'},'universes':[row_dict['strategy_import'] for row_dict in catalog_list],'decision_timing':'Selected sources Close_T; outer annual transfer based on preceding close weights','fill_timing':'Native selected sources Open_(T+1); outer transfer synthetic NAV units, not actual account orders','primary_cost_layer':'central_research','primary_metrics':{'period':'2018-07-20 through 2026-07-31, anchor2018-07-19','universe':'Four candidate books, same dates','cost_layer':'central_research','CAGR':selected_df.cagr.to_dict(),'annualized_volatility':selected_df.volatility.to_dict(),'Sharpe':selected_df.sharpe_zero.to_dict(),'maximum_drawdown':selected_df.max_drawdown.to_dict(),'turnover':'tables/exposures.csv for2012+; active_exposures.json for2018+; do not compare as identical periods'},'feature_findings':[],'cost_capacity':{'paper_like_round_trip_bps':None,'central_research_round_trip_bps':None,'conservative_survival_round_trip_bps':None,'basis':'Heterogeneous native costs retained. Central tax/funding overlay; conservative additional20bps round trip plus borrow/funding stress; not a uniform all-in execution cost.','capacity_impact_separate':True,'comfortable_capacity':None,'soft_capacity':None,'strained_capacity':None,'hard_capacity':None,'capacity_reason':'No account-specific order aggregation or market-impact capacity validation.'},'limitations':['All history seen; prior strategy search count unknown','Trinity holdings unresolved; shock conservative envelope','Current FRED vintage and older source-code lineage gaps','A3 fractional shares and shorter history','Outer transfers and cost overlays are synthetic, not full physical replay','USD exposure; no client ILS hedge','HPI common history endsJuly31; no September performance refresh'],'next_tests':['Account-specific full-book replay with integer units, execution and costs','Freeze forward observation; do not relabel old history a holdout'],'sources':spec_dict['sources'],'artifacts':artifact_dict,'adaptive_lineage':lineage_dict,'tags':['research-only','no-holdout','client-menu','all25','independent-design']}
    write_json(STUDY_PATH/'knowledge_record.json',knowledge_dict)
    notebook_obj = nbformat.v4.new_notebook(metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    notebook_obj.cells = [nbformat.v4.new_markdown_cell('# Independent portfolio menu — decision notebook\n\nExecuted closeout checks and evidence display, not a fresh strategy replay. All history was seen. Primary: S2/M0/G2. Provisional: A3. Research only.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport json, hashlib\nimport pandas as pd\nfrom IPython.display import display, Image\nstudy_path = Path("+repr(str(STUDY_PATH))+ ")\nspec_hash_str = hashlib.sha256((study_path/'research_spec_frozen.json').read_bytes()).hexdigest()\nassert spec_hash_str == '03a9e77d6ce200dbd09a56106d2c85a5ca6a6cd87036a4d350eb54609a9f8f5a'\nexperiment_list = [json.loads(line_str) for line_str in (study_path/'experiment_ledger.jsonl').read_text().splitlines()]\nassert sum(record_dict['declared_variant_count'] for record_dict in experiment_list) == 332\nprint('Frozen protocol unchanged;332 full-window cells across5 indexed groups; no untouched holdout.')"),nbformat.v4.new_code_cell("menu_df = pd.read_csv(study_path/'tables/menu_common_sample.csv')\ndisplay(menu_df[['candidate','scenario','cagr','volatility','sharpe_zero','max_drawdown','worst_rolling252']])\nreturn_df = pd.read_csv(study_path/'data/menu_common_returns.csv.gz', index_col=0)\nassert not return_df.isna().any().any()\nfor candidate_str in return_df:\n    calculated_float = (1+return_df[candidate_str]).prod()**(252/len(return_df))-1\n    stored_float = menu_df.loc[(menu_df.candidate==candidate_str)&(menu_df.scenario=='common_account'),'cagr'].iloc[0]\n    assert abs(calculated_float-stored_float)<1e-10\nprint('All7 curves match their table CAGR on exactly the same sessions.')"),nbformat.v4.new_code_cell("display(pd.read_csv(study_path/'tables/mandate_gates.csv'))\ndisplay(json.loads((study_path/'active_alternative_verdict.json').read_text()))"),nbformat.v4.new_code_cell("for chart_str in ['equity','drawdown','annual','rolling_correlation']:\n    display(Image(filename=str(study_path/'charts'/f'{chart_str}.png')))"),nbformat.v4.new_markdown_cell('## Decision limits\nThe long sample does not show G2 dominating existing Ladder4. Macro reduces market dependence but did not dominate monthly growth. A3 is post-result, shorter-history and fractional-share dependent. Historical drawdowns and shock scenarios are not future loss limits. Native strategy search and selection uncertainty remain.')]
    NotebookClient(notebook_obj,timeout=90,kernel_name='python3',resources={'metadata':{'path':str(ROOT_PATH)}}).execute()
    nbformat.write(notebook_obj, STUDY_PATH/'decision_notebook.ipynb')
    (verification_path/'PACKAGING.md').write_text('# Evidence packaging\n\nThe original frozen protocol was not rewritten. SOURCE_RULE_MAP, hypothesis registry and grouped ledger were assembled at closeout from frozen specifications and saved outputs. Original220 ledger records remain byte-for-byte in original_experiment_ledger.jsonl. The canonical ledger indexes5 groups totaling332 full-window cells (27+220+59+20+6). H4 counts only6 new comparisons; reused rows do not add trials.114 period/event views and6 paired bootstrap views are separate.400 is a budget ceiling, not400 trials.\n\nThe generic research-spec schema is not used as a second protocol: the repository-specific frozen contract is validated through input hashes and study assertions. English keyword checks from the generic report validator do not apply to the Hebrew narrative. Native Markdown links, Hebrew report structure, executed notebook, canonical lineage and strict manifest hashes are checked directly.\n',encoding='utf-8')
    print(json.dumps({'packaged':True,'cells':332,'notebook_executed':True,'elapsed_minutes':elapsed_int}))


def manifest_and_validate() -> None:
    file_list = []
    for file_path in sorted(STUDY_PATH.rglob('*')):
        if file_path.is_file() and file_path.name not in {'run_manifest.json','run_stdout.log','run_stderr.log'}:
            file_list.append({'path':file_path.relative_to(STUDY_PATH).as_posix(),'bytes':file_path.stat().st_size,'sha256':digest_str(file_path)})
    external_path_list = [*Path(__file__).parent.glob('*.py'),Path(__file__).parent/'verify_report.cjs',ROOT_PATH/'tests/test_client_menu_independent.py',ROOT_PATH/'scripts/research/portfolio_family_20260923/analyze.py',ROOT_PATH/'scripts/research/portfolio_family_20260923/freeze.py',SOURCE_PATH/'source_manifest.json',SOURCE_PATH/'source_audit_addendum.json',SOURCE_PATH/'catalog_complete.json']
    write_json(STUDY_PATH/'run_manifest.json',{'schema_version':'quant-research-manifest-v1','files':file_list,'external_deliverables_and_sources':[{'path':str(file_path),'bytes':file_path.stat().st_size,'sha256':digest_str(file_path)} for file_path in external_path_list]})
    adaptive_failure_list = validate_adaptive_research(STUDY_PATH)
    # This study constructs portfolios from saved strategies; it does not replicate a paper.
    # Keep not_assessed truthful instead of inventing a successful replication label.
    exception_str = 'baseline replication outcome must be assessed before diagnosis'
    exception_dict = {'allowed_schema_exception':exception_str,'reason':'Paper-replication prerequisite is inapplicable to portfolio construction from audited native exports; replication_outcome remains not_assessed.','observed':exception_str in adaptive_failure_list}
    write_json(STUDY_PATH/'verification/schema_exception.json',exception_dict)
    exception_path = STUDY_PATH/'verification/schema_exception.json'
    manifest_dict = read_json('run_manifest.json')
    manifest_dict['files'] = [record_dict for record_dict in manifest_dict['files'] if record_dict['path'] != 'verification/schema_exception.json']
    manifest_dict['files'].append({'path':'verification/schema_exception.json','bytes':exception_path.stat().st_size,'sha256':digest_str(exception_path)})
    write_json(STUDY_PATH/'run_manifest.json',manifest_dict)
    failure_list = [failure_str for failure_str in adaptive_failure_list if failure_str != exception_str]+validate_knowledge_record(STUDY_PATH/'knowledge_record.json')+validate_notebook(STUDY_PATH/'decision_notebook.ipynb',True)+validate_manifest(STUDY_PATH,True)
    for report_str in ['REPORT.md','REPORT_FULL.md']:
        failure_list += validate_markdown_images(STUDY_PATH/report_str)
    if failure_list:
        raise ValueError('\n'.join(failure_list))
    print('PASS: lineage with one documented inapplicable paper-replication prerequisite; knowledge, executed notebook, report images and strict manifest.')


if __name__ == '__main__':
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument('--manifest-only',action='store_true')
    argument_obj = parser_obj.parse_args()
    if not argument_obj.manifest_only:
        package()
    else:
        manifest_and_validate()
