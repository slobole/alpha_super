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

from alpha.engine.portfolio import Portfolio
from alpha.engine.report import save_portfolio_results
from alpha.engine.strategy import Strategy


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / 'results'
STRATEGIES_DIR = ROOT_DIR / 'strategies'
VALID_REBALANCE_SET = {None, 'monthly', 'quarterly', 'annually'}
METADATA_FILENAME = 'metadata.json'

_strategy_classes = {}
_strategy_import_errors = {}


def _resolve_path(path_like, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _metadata_path(pkl_path: Path) -> Path:
    return pkl_path.parent / METADATA_FILENAME


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
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            _strategy_classes[attr_name] = obj
            class_names_list.append(attr_name)
    return class_names_list


def _register_strategy_classes_from_metadata(metadata: dict, pkl_path: Path):
    class_file = metadata.get('class_file')
    class_module = metadata.get('class_module')

    if class_file:
        module = _import_strategy_module(Path(class_file))
        _register_strategy_classes_from_module(module)
        return

    if class_module and class_module != '__main__':
        module = importlib.import_module(class_module)
        _register_strategy_classes_from_module(module)
        return

    raise ImportError(
        f"Metadata for '{pkl_path}' is missing a usable class import path. "
        "Expected 'class_file' or a non-__main__ 'class_module'."
    )


def discover_strategy_classes(search_dir: Path = STRATEGIES_DIR):
    """Import strategy files dynamically so legacy __main__ pickles can be loaded."""
    for module_path in sorted(search_dir.glob('strategy_*.py')):
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
        if module == '__main__' and name in _strategy_classes:
            return _strategy_classes[name]
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
    """Find the most recent result pickle for a strategy by timestamped folder name."""
    strategy_dir = results_dir / strategy_name
    if not strategy_dir.exists():
        raise FileNotFoundError(f"No results found for '{strategy_name}' in '{results_dir}'.")

    run_dirs_list = sorted(
        [path for path in strategy_dir.iterdir() if path.is_dir()],
        reverse=True,
    )
    for run_dir in run_dirs_list:
        pkl_path = run_dir / f'{strategy_name}.pkl'
        if pkl_path.exists():
            print(f"  Falling back to latest result for {strategy_name}: {run_dir.name}")
            return pkl_path.resolve()

    raise FileNotFoundError(f"No pickle file found in '{strategy_dir}'.")


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

    normalized_config = {
        'name': config_dict.get('name', 'Portfolio'),
        'capital': float(capital) if capital is not None else None,
        'rebalance': _normalize_rebalance(config_dict.get('rebalance')),
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
    portfolio = Portfolio(
        strategies=strategies_list,
        weights=weights_list,
        name=portfolio_name,
        capital_base=capital,
        rebalance=config_dict['rebalance'],
        pod_info_list=pod_info_list,
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
