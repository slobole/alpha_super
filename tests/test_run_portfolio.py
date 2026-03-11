import contextlib
import importlib.util
import json
import pickle
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd
import yaml

from alpha.engine.strategy import Strategy


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_PORTFOLIO_PATH = ROOT_DIR / 'strategies' / 'run_portfolio.py'
TEST_TMP_ROOT = ROOT_DIR / '.tmp_test_runs'


def load_run_portfolio_module():
    spec = importlib.util.spec_from_file_location('test_run_portfolio_module', RUN_PORTFOLIO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def temporary_test_dir():
    # `tempfile.TemporaryDirectory()` creates ACL-restricted folders under this
    # Windows sandbox. Build test scratch directories explicitly in-workspace.
    test_dir = TEST_TMP_ROOT / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield test_dir
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


class DummyLoadedStrategy(Strategy):
    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def make_strategy(name: str, capital_base: float = 100.0):
    dates_index = pd.to_datetime(['2024-01-30', '2024-01-31', '2024-02-03'])
    daily_returns_ser = pd.Series([0.0, 0.01, -0.01], index=dates_index, dtype=float)
    total_value_ser = capital_base * (1 + daily_returns_ser).cumprod()

    strategy = DummyLoadedStrategy(
        name=name,
        benchmarks=[],
        capital_base=capital_base,
        slippage=0.0,
        commission_per_share=0.0,
        commission_minimum=0.0,
    )
    strategy.results = pd.DataFrame({
        'daily_returns': daily_returns_ser,
        'total_value': total_value_ser,
        'portfolio_value': total_value_ser,
    }, index=dates_index)
    strategy.summary = pd.DataFrame()
    strategy.summary_trades = pd.DataFrame()
    return strategy


def write_strategy_result(base_dir: Path, strategy) -> Path:
    run_dir = base_dir / strategy.name / '2026-03-09_000000'
    run_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = run_dir / f'{strategy.name}.pkl'
    with pickle_path.open('wb') as file_obj:
        pickle.dump(strategy, file_obj)

    metadata_dict = {
        'artifact_type': 'strategy',
        'strategy_name': strategy.name,
        'class_name': strategy.__class__.__name__,
        'class_module': strategy.__class__.__module__,
        'class_file': str(Path(__file__).resolve()),
        'capital_base': float(strategy._capital_base),
        'pickle_path': str(pickle_path.resolve()),
        'saved_at': '2026-03-09T00:00:00',
    }
    (run_dir / 'metadata.json').write_text(json.dumps(metadata_dict, indent=2), encoding='utf-8')
    return pickle_path


class RunPortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_portfolio = load_run_portfolio_module()

    def test_validate_portfolio_config_rejects_mismatched_pickle_name(self):
        with temporary_test_dir() as temp_dir:
            config_path = temp_dir / 'portfolio.yaml'
            wrong_pickle_path = temp_dir / 'WrongStrategy.pkl'
            wrong_pickle_path.write_bytes(b'placeholder')

            config_dict = {
                'name': 'TestPortfolio',
                'capital': 100000,
                'pods': [
                    {
                        'strategy': 'RightStrategy',
                        'weight': 1.0,
                        'pkl': str(wrong_pickle_path),
                    }
                ],
            }

            with self.assertRaisesRegex(ValueError, 'points to pickle'):
                self.run_portfolio.validate_portfolio_config(config_dict, config_path)

    def test_validate_portfolio_config_rejects_metadata_strategy_mismatch(self):
        with temporary_test_dir() as temp_dir:
            strategy = make_strategy('QPIStrategy')
            pickle_path = write_strategy_result(temp_dir, strategy)
            metadata_path = pickle_path.parent / 'metadata.json'
            metadata_dict = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata_dict['strategy_name'] = 'DifferentStrategy'
            metadata_path.write_text(json.dumps(metadata_dict, indent=2), encoding='utf-8')

            config_path = temp_dir / 'portfolio.yaml'
            config_dict = {
                'name': 'TestPortfolio',
                'capital': 100000,
                'pods': [
                    {
                        'strategy': 'QPIStrategy',
                        'weight': 1.0,
                        'pkl': str(pickle_path),
                    }
                ],
            }

            with self.assertRaisesRegex(ValueError, 'does not match metadata strategy'):
                self.run_portfolio.validate_portfolio_config(config_dict, config_path)

    def test_build_portfolio_reads_rebalance_and_pod_provenance(self):
        with temporary_test_dir() as temp_dir:
            strategy_a = make_strategy('StrategyA')
            strategy_b = make_strategy('StrategyB')
            pickle_a = write_strategy_result(temp_dir, strategy_a)
            pickle_b = write_strategy_result(temp_dir, strategy_b)

            config_path = temp_dir / 'portfolio.yaml'
            config_dict = {
                'name': 'TestPortfolio',
                'capital': 100000,
                'rebalance': 'monthly',
                'pods': [
                    {'strategy': 'StrategyA', 'weight': 0.4, 'pkl': str(pickle_a)},
                    {'strategy': 'StrategyB', 'weight': 0.6, 'pkl': str(pickle_b)},
                ],
            }
            config_path.write_text(yaml.safe_dump(config_dict), encoding='utf-8')

            portfolio = self.run_portfolio.build_portfolio(config_path)

            self.assertEqual(portfolio._rebalance, 'monthly')
            self.assertEqual(portfolio.source_config_path, str(config_path.resolve()))
            self.assertEqual(len(portfolio.pod_info_list), 2)
            self.assertEqual(portfolio.pod_info_list[0]['source_pkl'], str(pickle_a))
            self.assertEqual(portfolio.pod_info_list[1]['source_pkl'], str(pickle_b))
            self.assertAlmostEqual(portfolio.pod_info_list[0]['allocated_capital'], 40000.0)
            self.assertAlmostEqual(portfolio.pod_info_list[1]['allocated_capital'], 60000.0)

