"""
run_portfolio.py
----------------
Combine pre-computed strategy pickle files into a Portfolio for unified analysis.

Usage:
    uv run python strategies/run_portfolio.py portfolios/multipod.yaml
    uv run python strategies/run_portfolio.py portfolios/multipod.yaml --name MyTest
    uv run python strategies/run_portfolio.py portfolios/multipod.yaml --capital 200000
"""

import argparse
import importlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import display

# Executed by file path (Bench, terminal), the script dir — not the repo root —
# lands on sys.path, so make the repo root importable before touching alpha.
REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.portfolio import Portfolio
from alpha.engine.report import save_portfolio_results
from alpha.engine.strategy import Strategy


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / 'results'
STRATEGIES_DIR = ROOT_DIR / 'strategies'
VALID_REBALANCE_SET = {None, 'monthly', 'quarterly', 'annually'}
METADATA_FILENAME = 'metadata.json'
LEGACY_STRATEGY_PACKAGE_MAP = (
    ('strategy_taa_traditional_sma8', STRATEGIES_DIR / 'taa_traditional'),
    ('strategy_taa_beyond_6040', STRATEGIES_DIR / 'taa_beyond_6040'),
    ('strategy_taa_df', STRATEGIES_DIR / 'taa_df'),
    ('strategy_mr_alpha19', STRATEGIES_DIR / 'alpha19'),
    ('strategy_mr_dv2', STRATEGIES_DIR / 'dv2'),
    ('strategy_mr_qpi', STRATEGIES_DIR / 'qpi'),
    ('strategy_mr_spx_rsi2', STRATEGIES_DIR / 'vix_stuff'),
    ('strategy_mr_vix', STRATEGIES_DIR / 'vix_stuff'),
    ('strategy_mr_vxx', STRATEGIES_DIR / 'vix_stuff'),
    ('strategy_bom_', STRATEGIES_DIR / 'bom_tlt'),
    ('strategy_eom_', STRATEGIES_DIR / 'eom_tlt_vs_spy'),
    ('strategy_mo_', STRATEGIES_DIR / 'momentum'),
    ('strategy_seasonality', STRATEGIES_DIR / 'seasonality'),
)

_strategy_classes = {}
_strategy_import_errors = {}
_legacy_main_symbol_map = {}


def _resolve_path(path_like, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _metadata_path(pkl_path: Path) -> Path:
    return pkl_path.parent / METADATA_FILENAME


def _legacy_strategy_package_dir(module_stem_str: str) -> Path | None:
    for legacy_prefix_str, package_dir_path in LEGACY_STRATEGY_PACKAGE_MAP:
        if module_stem_str.startswith(legacy_prefix_str):
            return package_dir_path
    return None


def _module_path_to_import_str(module_path: Path) -> str:
    relative_module_path = module_path.resolve().relative_to(ROOT_DIR)
    return '.'.join(relative_module_path.with_suffix('').parts)


def _remap_legacy_strategy_module_name(class_module_str: str) -> str:
    if not class_module_str.startswith('strategies.strategy_'):
        return class_module_str

    module_stem_str = class_module_str.rsplit('.', 1)[-1]
    package_dir_path = _legacy_strategy_package_dir(module_stem_str)
    if package_dir_path is None:
        return class_module_str

    remapped_module_path = package_dir_path / f'{module_stem_str}.py'
    if not remapped_module_path.exists():
        return class_module_str

    return _module_path_to_import_str(remapped_module_path)


def _remap_legacy_strategy_file_path(module_path: Path) -> Path:
    if module_path.exists():
        return module_path

    if module_path.parent != STRATEGIES_DIR:
        return module_path

    package_dir_path = _legacy_strategy_package_dir(module_path.stem)
    if package_dir_path is None:
        return module_path

    remapped_module_path = package_dir_path / module_path.name
    return remapped_module_path if remapped_module_path.exists() else module_path


def read_result_metadata(pkl_path: Path) -> dict | None:
    """Read optional result metadata stored alongside a pickle."""
    metadata_path = _metadata_path(pkl_path)
    if not metadata_path.exists():
        return None
    with metadata_path.open(encoding='utf-8') as file_obj:
        return json.load(file_obj)


def _import_strategy_module(module_path: Path):
    module_path = module_path.resolve()
    module_name = f'portfolio_loader_{module_path.stem}_{abs(hash(module_path.as_posix()))}'
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        return existing_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _register_strategy_classes_from_module(module) -> list[str]:
    class_names_list = []
    for attr_name in dir(module):
        if attr_name.startswith('__'):
            continue
        _legacy_main_symbol_map[attr_name] = getattr(module, attr_name)

    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            _strategy_classes[attr_name] = obj
            class_names_list.append(attr_name)
    return class_names_list


def _register_strategy_classes_from_metadata(metadata: dict, pkl_path: Path):
    class_file = metadata.get('class_file')
    class_module = metadata.get('class_module')

    if class_file:
        module = _import_strategy_module(_remap_legacy_strategy_file_path(Path(class_file)))
        _register_strategy_classes_from_module(module)
        return

    if class_module and class_module != '__main__':
        remapped_class_module_str = _remap_legacy_strategy_module_name(class_module)
        module = importlib.import_module(remapped_class_module_str)
        _register_strategy_classes_from_module(module)
        return

    raise ImportError(
        f"Metadata for '{pkl_path}' is missing a usable class import path. "
        "Expected 'class_file' or a non-__main__ 'class_module'."
    )


def discover_strategy_classes(search_dir: Path = STRATEGIES_DIR):
    """Import strategy files dynamically so legacy __main__ pickles can be loaded."""
    for module_path in sorted(search_dir.rglob('strategy_*.py')):
        if module_path.name == 'run_portfolio.py':
            continue
        if module_path in _strategy_import_errors:
            continue
        try:
            module = _import_strategy_module(module_path)
        except Exception as exc:
            _strategy_import_errors[module_path] = exc
            continue
        _register_strategy_classes_from_module(module)


def _strategy_import_error_text() -> str:
    if not _strategy_import_errors:
        return 'No strategy import errors were recorded.'
    parts = []
    for module_path, exc in sorted(_strategy_import_errors.items(), key=lambda item: str(item[0])):
        parts.append(f'  - {module_path}: {exc}')
    return 'Strategy import errors:\n' + '\n'.join(parts)


class _StrategyUnpickler(pickle.Unpickler):
    """Unpickler that remaps legacy __main__.* strategy classes."""

    def find_class(self, module, name):
        if module == '__main__':
            if name in _strategy_classes:
                return _strategy_classes[name]
            if name in _legacy_main_symbol_map:
                return _legacy_main_symbol_map[name]
        return super().find_class(module, name)


def load_strategy_pickle(pkl_path: Path, expected_strategy_name: str | None = None):
    """Load a strategy pickle using metadata when available, fallback discovery otherwise."""
    pkl_path = pkl_path.resolve()
    metadata = read_result_metadata(pkl_path)

    if metadata is not None:
        _register_strategy_classes_from_metadata(metadata, pkl_path)
    else:
        discover_strategy_classes()

    with pkl_path.open('rb') as file_obj:
        strategy = _StrategyUnpickler(file_obj).load()

    if not isinstance(strategy, Strategy):
        raise TypeError(f"Pickle '{pkl_path}' did not contain a Strategy instance.")

    if expected_strategy_name is not None and strategy.name != expected_strategy_name:
        raise ValueError(
            f"Configured strategy '{expected_strategy_name}' does not match loaded "
            f"strategy name '{strategy.name}' from '{pkl_path}'."
        )

    if metadata is not None:
        metadata_strategy_name = metadata.get('strategy_name')
        if metadata_strategy_name and metadata_strategy_name != strategy.name:
            raise ValueError(
                f"Metadata strategy name '{metadata_strategy_name}' does not match loaded "
                f"strategy name '{strategy.name}' from '{pkl_path}'."
            )

    if strategy.__class__.__name__ not in _strategy_classes and metadata is None:
        raise ImportError(
            f"Could not register the strategy class needed to load '{pkl_path}'.\n"
            f"{_strategy_import_error_text()}"
        )

    return strategy, metadata


def find_latest_pkl(strategy_name: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Find the most recent result pickle for a strategy by timestamped folder name.

    Searches the research tree first (``results/research/strategy/<name>/
    vanilla_backtest/<ts>/``) — vanilla_backtest only, because other analyzers
    may pickle modified configurations — then falls back to the legacy layout
    ``results/<name>/<ts>/``. Both are ordered newest-first by folder name.
    """
    candidate_run_dir_list: list[Path] = []
    research_vanilla_dir = results_dir / 'research' / 'strategy' / strategy_name / 'vanilla_backtest'
    if research_vanilla_dir.exists():
        candidate_run_dir_list.extend(
            sorted(
                [path for path in research_vanilla_dir.iterdir() if path.is_dir()],
                reverse=True,
            )
        )
    legacy_strategy_dir = results_dir / strategy_name
    if legacy_strategy_dir.exists():
        candidate_run_dir_list.extend(
            sorted(
                [path for path in legacy_strategy_dir.iterdir() if path.is_dir()],
                reverse=True,
            )
        )
    if not candidate_run_dir_list:
        raise FileNotFoundError(
            f"No results found for '{strategy_name}' in '{research_vanilla_dir}' "
            f"or '{legacy_strategy_dir}'."
        )

    for run_dir in candidate_run_dir_list:
        pkl_path = run_dir / f'{strategy_name}.pkl'
        if pkl_path.exists():
            print(f"  Using latest result for {strategy_name}: {run_dir.parent.name}/{run_dir.name}")
            return pkl_path.resolve()

    raise FileNotFoundError(
        f"No pickle file found for '{strategy_name}' under "
        f"'{research_vanilla_dir}' or '{legacy_strategy_dir}'."
    )


def _normalize_rebalance(rebalance):
    if rebalance is None:
        return None
    if not isinstance(rebalance, str):
        raise ValueError(f"rebalance must be one of {VALID_REBALANCE_SET}, got {rebalance!r}")
    rebalance = rebalance.strip().lower()
    if rebalance not in VALID_REBALANCE_SET:
        raise ValueError(f"rebalance must be one of {VALID_REBALANCE_SET}, got '{rebalance}'")
    return rebalance


def load_portfolio_config(config_path: Path) -> dict:
    """Load a portfolio YAML configuration file."""
    with config_path.open(encoding='utf-8') as file_obj:
        config_dict = yaml.safe_load(file_obj)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Portfolio config '{config_path}' must contain a mapping at the top level.")

    return config_dict


def validate_portfolio_config(config_dict: dict, config_path: Path) -> dict:
    """Validate and normalize the portfolio configuration before loading any pickles."""
    pods_list = config_dict.get('pods')
    if not isinstance(pods_list, list) or len(pods_list) == 0:
        raise ValueError(f"Portfolio config '{config_path}' must define a non-empty 'pods' list.")

    capital = config_dict.get('capital')
    if capital is not None and float(capital) <= 0:
        raise ValueError(f"Portfolio capital must be positive, got {capital}.")

    benchmark_symbol = config_dict.get('benchmark')
    if benchmark_symbol is not None and (
        not isinstance(benchmark_symbol, str) or not benchmark_symbol.strip()
    ):
        raise ValueError(f"Portfolio 'benchmark' must be a non-empty string, got {benchmark_symbol!r}.")

    normalized_config = {
        'name': config_dict.get('name', 'Portfolio'),
        'capital': float(capital) if capital is not None else None,
        'rebalance': _normalize_rebalance(config_dict.get('rebalance')),
        'benchmark': benchmark_symbol.strip() if benchmark_symbol is not None else None,
        'pods': [],
    }

    config_dir = config_path.parent
    strategy_name_set = set()
    total_weight = 0.0

    for idx, pod_dict in enumerate(pods_list, start=1):
        if not isinstance(pod_dict, dict):
            raise ValueError(f"Pod #{idx} in '{config_path}' must be a mapping.")

        strategy_name = pod_dict.get('strategy')
        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError(f"Pod #{idx} in '{config_path}' is missing a valid 'strategy' value.")
        if strategy_name in strategy_name_set:
            raise ValueError(
                f"Duplicate strategy '{strategy_name}' in '{config_path}'. "
                "Portfolio pod names must be unique."
            )
        strategy_name_set.add(strategy_name)

        if 'weight' not in pod_dict:
            raise ValueError(f"Pod '{strategy_name}' is missing 'weight'.")
        weight = float(pod_dict['weight'])
        if weight <= 0:
            raise ValueError(f"Pod '{strategy_name}' must have a positive weight, got {weight}.")
        total_weight += weight

        if pod_dict.get('pkl'):
            pkl_path = _resolve_path(pod_dict['pkl'], config_dir)
            if not pkl_path.exists():
                raise FileNotFoundError(f"Configured pickle for '{strategy_name}' does not exist: '{pkl_path}'.")
            if pkl_path.suffix.lower() != '.pkl':
                raise ValueError(f"Configured pickle for '{strategy_name}' must end with '.pkl': '{pkl_path}'.")
            if pkl_path.stem != strategy_name:
                raise ValueError(
                    f"Configured strategy '{strategy_name}' points to pickle '{pkl_path.name}', "
                    "which suggests a different strategy result."
                )

            metadata = read_result_metadata(pkl_path)
            if metadata is not None:
                metadata_strategy_name = metadata.get('strategy_name')
                if metadata_strategy_name and metadata_strategy_name != strategy_name:
                    raise ValueError(
                        f"Configured strategy '{strategy_name}' does not match metadata strategy "
                        f"'{metadata_strategy_name}' in '{_metadata_path(pkl_path)}'."
                    )
        else:
            pkl_path = find_latest_pkl(strategy_name)

        normalized_config['pods'].append({
            'strategy': strategy_name,
            'weight': weight,
            'pkl_path': pkl_path,
        })

    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Portfolio weights must sum to 1.0, got {total_weight:.6f}.")

    return normalized_config


def load_pod_strategy(pod_dict: dict) -> tuple[Strategy, dict]:
    """Load one pod strategy and return its portfolio provenance metadata."""
    strategy_name = pod_dict['strategy']
    pkl_path = pod_dict['pkl_path']
    weight = pod_dict['weight']

    print(f"  Loading {strategy_name} (weight={weight:.2%}) from {pkl_path}")
    strategy, result_metadata = load_strategy_pickle(pkl_path, expected_strategy_name=strategy_name)
    pod_info_dict = {
        'strategy_name': strategy_name,
        'weight': weight,
        'source_pkl': str(pkl_path),
        'result_metadata': result_metadata,
    }
    return strategy, pod_info_dict


def derive_pm_benchmark(
    strategies_list,
    preferred_symbol_str: str | None = None,
) -> tuple[pd.Series | None, str | None, str | None]:
    """Pick the PM benchmark from benchmark value columns stored in pod results.

    Uses only data already inside the pickles (no fresh data load): each saved
    strategy run stores its benchmark level series (e.g. ``$SPX``, TOTALRETURN)
    as a column of ``results``.

    Selection is explicit, never a silent vote:
      * ``preferred_symbol_str`` set (YAML ``benchmark:`` key) — use that symbol;
        raise if no pod stores it, so a configured benchmark cannot silently
        disappear from the report.
      * auto mode — all pods that store a benchmark must agree on symbol and
        adjustment, otherwise attach nothing and say so loudly.

    Returns ``(benchmark_value_ser, benchmark_label_str, benchmark_adjustment_str)``
    or ``(None, None, None)`` when no consistent benchmark exists.
    """
    candidate_list: list[tuple[str, str, pd.Series, str]] = []
    for strategy_obj in strategies_list:
        symbol_str = getattr(strategy_obj, '_performance_benchmark_symbol_str', None)
        if symbol_str is None:
            benchmark_list = list(getattr(strategy_obj, '_benchmarks', []) or [])
            symbol_str = str(benchmark_list[0]) if len(benchmark_list) > 0 else None
        if symbol_str is None or symbol_str not in strategy_obj.results.columns:
            continue
        adjustment_str = getattr(
            strategy_obj, '_performance_benchmark_adjustment_str', 'not_declared'
        )
        benchmark_value_ser = strategy_obj.results[symbol_str].astype(float).dropna()
        if len(benchmark_value_ser) < 2:
            continue
        candidate_list.append(
            (symbol_str, str(adjustment_str), benchmark_value_ser, strategy_obj.name)
        )

    if preferred_symbol_str is not None:
        candidate_list = [
            candidate for candidate in candidate_list if candidate[0] == preferred_symbol_str
        ]
        if not candidate_list:
            raise ValueError(
                f"Configured benchmark '{preferred_symbol_str}' is not stored in any pod's "
                'results; remove the benchmark key or re-run a pod with that benchmark.'
            )
    else:
        if not candidate_list:
            print(
                'PM benchmark: no pod stores a benchmark value column; '
                'benchmark-relative report sections will be omitted.'
            )
            return None, None, None
        symbol_set = {symbol_str for symbol_str, _, _, _ in candidate_list}
        adjustment_set = {adjustment_str for _, adjustment_str, _, _ in candidate_list}
        if len(symbol_set) > 1 or len(adjustment_set) > 1:
            print(
                'PM benchmark WARNING: pods disagree on benchmark '
                f'(symbols={sorted(symbol_set)}, adjustments={sorted(adjustment_set)}); '
                'attaching none instead of silently choosing one. '
                "Set an explicit 'benchmark: <symbol>' key in the portfolio YAML to resolve."
            )
            return None, None, None

    declared_adjustment_set = {
        adjustment_str
        for _, adjustment_str, _, _ in candidate_list
        if adjustment_str != 'not_declared'
    }
    if len(declared_adjustment_set) > 1:
        raise ValueError(
            f"Pods store benchmark '{candidate_list[0][0]}' under conflicting declared "
            f'adjustments {sorted(declared_adjustment_set)}; refusing to mix them.'
        )
    adjustment_str = (
        next(iter(declared_adjustment_set)) if declared_adjustment_set else 'not_declared'
    )

    symbol_str = candidate_list[0][0]

    # *** CRITICAL*** same-symbol stored series must actually agree. Some runs
    # have stored a price-adjusted benchmark while declaring TOTALRETURN; when a
    # genuinely total-return series is also present, the two drift apart by
    # roughly the dividend yield. Compare candidates over their common dates
    # and, on material divergence, say so and prefer the fastest-growing series
    # — for one symbol, total return >= price by construction.
    divergent_growth_pod_name_str = None
    if len(candidate_list) > 1:
        common_index = candidate_list[0][2].index
        for _, _, value_ser, _ in candidate_list[1:]:
            common_index = common_index.intersection(value_ser.index)
        if len(common_index) >= 252:
            year_count_float = (common_index[-1] - common_index[0]).days / 365.25
            cagr_by_pod_dict = {
                pod_name_str: (
                    (value_ser.loc[common_index].iloc[-1] / value_ser.loc[common_index].iloc[0])
                    ** (1.0 / year_count_float)
                    - 1.0
                )
                for _, _, value_ser, pod_name_str in candidate_list
            }
            if max(cagr_by_pod_dict.values()) - min(cagr_by_pod_dict.values()) > 0.005:
                print(
                    f"PM benchmark *** CRITICAL***: pods store '{symbol_str}' series that "
                    'disagree materially over their common window '
                    f'({common_index[0].date()} -> {common_index[-1].date()}):'
                )
                for pod_name_str, cagr_float in sorted(
                    cagr_by_pod_dict.items(), key=lambda item: item[1], reverse=True
                ):
                    print(f'  {pod_name_str}: {cagr_float:.2%} CAGR')
                divergent_growth_pod_name_str = max(cagr_by_pod_dict, key=cagr_by_pod_dict.get)
                print(
                    f"  Using the fastest-growing series ({divergent_growth_pod_name_str}) as "
                    f"'{adjustment_str}': for one symbol total return >= price. At least one "
                    'other pod mislabels its stored benchmark adjustment — fix it at the source.'
                )

    if divergent_growth_pod_name_str is not None:
        benchmark_value_ser = next(
            value_ser
            for _, _, value_ser, pod_name_str in candidate_list
            if pod_name_str == divergent_growth_pod_name_str
        )
    else:
        # Prefer a pod that declares its adjustment, then the longest stored
        # series; every candidate covers the PM common window, and the Portfolio
        # validates completeness after reindexing to that window.
        benchmark_value_ser = max(
            candidate_list,
            key=lambda candidate: (candidate[1] != 'not_declared', len(candidate[2])),
        )[2]
    print(
        f'PM benchmark attached from pod results: {symbol_str} ({adjustment_str}), '
        f'{len(benchmark_value_ser)} bars '
        f'({benchmark_value_ser.index[0].date()} -> {benchmark_value_ser.index[-1].date()}).'
    )
    return benchmark_value_ser, symbol_str, adjustment_str


def build_portfolio(config_path: Path, name_override=None, capital_override=None) -> Portfolio:
    """Build a portfolio object from a validated YAML configuration."""
    config_path = config_path.resolve()
    config_dict = load_portfolio_config(config_path)
    config_dict = validate_portfolio_config(config_dict, config_path)

    portfolio_name = name_override or config_dict['name']
    capital = capital_override if capital_override is not None else config_dict['capital']

    strategies_list = []
    weights_list = []
    pod_info_list = []
    for pod_dict in config_dict['pods']:
        strategy, pod_info_dict = load_pod_strategy(pod_dict)
        strategies_list.append(strategy)
        weights_list.append(pod_dict['weight'])
        pod_info_list.append(pod_info_dict)

    print(f'\nLoaded {len(strategies_list)} strategies: {[strategy.name for strategy in strategies_list]}')
    (
        benchmark_value_ser,
        benchmark_label_str,
        benchmark_adjustment_str,
    ) = derive_pm_benchmark(strategies_list, preferred_symbol_str=config_dict['benchmark'])
    portfolio = Portfolio(
        strategies=strategies_list,
        weights=weights_list,
        name=portfolio_name,
        capital_base=capital,
        rebalance=config_dict['rebalance'],
        pod_info_list=pod_info_list,
        regression_benchmark_value_ser=benchmark_value_ser,
        regression_benchmark_label_str=benchmark_label_str,
        regression_benchmark_adjustment_str=benchmark_adjustment_str,
    )
    portfolio.source_config_path = str(config_path)
    return portfolio


def main():
    parser = argparse.ArgumentParser(description='Combine strategy pickle files into a portfolio')
    parser.add_argument('config', help='Path to portfolio YAML config')
    parser.add_argument('--name', default=None, help='Override portfolio name')
    parser.add_argument('--capital', type=float, default=None, help='Override capital')
    args = parser.parse_args()

    portfolio = build_portfolio(Path(args.config), name_override=args.name, capital_override=args.capital)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print('\n--- Portfolio Summary ---')
    display(portfolio.summary)

    print('\n--- Trade Statistics ---')
    display(portfolio.summary_trades)

    print('\n--- Monthly Returns ---')
    display(portfolio.monthly_returns)

    save_portfolio_results(portfolio)


if __name__ == '__main__':
    main()
