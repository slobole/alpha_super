"""Portfolio maturity stays explicit and matches the Bench presentation."""

from __future__ import annotations

from pathlib import Path

import yaml

from alpha import portfolio_registry, strategy_registry
from alpha.bench import catalog
from alpha.bench.app import create_app
from alpha.strategy_registry import MaturityTier


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]


class _ReadOnlyJobManager:
    def active_count(self) -> int:
        return 0

    def list_jobs(self) -> list:
        return []


def test_loren_is_explicitly_wired():
    assert portfolio_registry.tier_for("loren") is MaturityTier.WIRED
    assert portfolio_registry.tier_label_for("loren") == "wired"


def test_unregistered_portfolio_defaults_to_research():
    assert portfolio_registry.tier_for("not_registered") is MaturityTier.RESEARCH


def test_registered_portfolios_exist_and_parse():
    portfolio_by_name_dict = {
        portfolio_obj.name_str: portfolio_obj for portfolio_obj in catalog.list_portfolios()
    }
    for portfolio_name_str, tier_obj in portfolio_registry.PORTFOLIO_TIER_DICT.items():
        assert portfolio_name_str in portfolio_by_name_dict
        portfolio_obj = portfolio_by_name_dict[portfolio_name_str]
        assert portfolio_obj.error_str is None
        assert portfolio_obj.tier_int == int(tier_obj)
        assert portfolio_obj.tier_label_str == strategy_registry.TIER_LABEL_DICT[tier_obj]


def test_registered_portfolio_pods_reach_the_claimed_tier():
    """A portfolio cannot claim more maturity than any strategy inside it."""
    for portfolio_name_str, tier_obj in portfolio_registry.PORTFOLIO_TIER_DICT.items():
        config_path = REPO_ROOT_PATH / "portfolios" / f"{portfolio_name_str}.yaml"
        config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for pod_dict in config_dict["pods"]:
            strategy_import_str = pod_dict["strategy_import_str"]
            assert strategy_registry.tier_for(strategy_import_str) >= tier_obj


def test_portfolios_page_renders_maturity_column():
    flask_app_obj = create_app(job_manager_obj=_ReadOnlyJobManager())
    response_obj = flask_app_obj.test_client().get("/portfolios")
    html_str = response_obj.get_data(as_text=True)

    assert response_obj.status_code == 200
    assert "<th>Maturity</th>" in html_str
    assert '<span class="toolbar-group-label">Maturity</span>' in html_str
    assert 'data-filter="maturity:wired"' in html_str
    assert 'data-filter="maturity:pm_ready"' in html_str
    assert 'data-filter="maturity:research"' in html_str
    assert 'data-filter="measured"' not in html_str
    assert 'data-filter="norun"' not in html_str
    assert 'data-filter="stale"' not in html_str
    assert 'data-filter="combine"' not in html_str
    assert 'data-filter="fresh"' not in html_str
    assert 'data-maturity="wired"' in html_str
    assert '<span class="maturity maturity-wired">WIRED</span>' in html_str
