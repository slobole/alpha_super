"""Point-in-time equity panels and the vault seal (design section 4).

A panel is a set of (date x symbol) frames for one index universe: CAPITALSPECIAL Open/High/Low/Close, Volume,
Turnover, Unadjusted Close, Dividend, and the exact point-in-time membership mask (1 on sessions the symbol was an
index member, 0 otherwise; readiness-audit fix #7, no tail trim). It is built from Norgate through the engine's own
loaders and identified by a content hash (`snapshot_id_str`: every field's values, dates, symbols and field names,
and the membership mask) that goes into every ledger row that uses it. Each snapshot is cached in its own folder,
results/scout/panels/<name>/<snapshot_id>/, and never overwritten: a rebuild after a data refresh (new dividends
revise adjusted history) is a new snapshot, and every earlier result can be re-run on the snapshot it used.

The vault: every panel handed to research code is cut at the seal (2023-01-01) unless the id of a vault-opening row
for the family is supplied and that row is in the verified ledger chain. Forward labels computed on a sealed panel
cannot read vault bars, because those bars do not exist in it (purge by construction). Research code reads panels
only through load_panel(); the cache files themselves hold vault bars.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

VAULT_SEAL_STR = "2023-01-01"
FIELD_TUPLE = ("Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend")
PANEL_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "panels"


@dataclass(frozen=True)
class Panel:
    name_str: str
    field_dict: dict  # field name -> DataFrame (date x symbol)
    member_df: pd.DataFrame  # 1/0, same index/columns as the fields
    snapshot_id_str: str
    sealed_bool: bool

    @property
    def date_index(self) -> pd.DatetimeIndex:
        return self.member_df.index

    @property
    def symbol_list(self) -> list[str]:
        return list(self.member_df.columns)

    def field(self, field_str: str) -> pd.DataFrame:
        return self.field_dict[field_str]

    def subset(self, symbol_list: list[str]) -> Panel:
        """The same panel restricted to some symbols (used to keep S1 tests fast)."""
        return replace(
            self,
            field_dict={name: frame[symbol_list] for name, frame in self.field_dict.items()},
            member_df=self.member_df[symbol_list],
        )

    def truncated(self, end_ts) -> Panel:
        """The panel as it was known at the close of end_ts (used by the S1 prefix test)."""
        end_ts = pd.Timestamp(end_ts)
        return replace(
            self,
            field_dict={name: frame.loc[:end_ts] for name, frame in self.field_dict.items()},
            member_df=self.member_df.loc[:end_ts],
        )


def _panel_dir(name_str: str) -> Path:
    return PANEL_ROOT_PATH / name_str.replace(" ", "_").replace("&", "and")


def _file_str(field_str: str) -> str:
    return f"{field_str.replace(' ', '_')}.parquet"


def build_panel(index_name_str: str, start_date_str: str = "1998-01-01", name_str: str | None = None) -> Path:
    """Build the full (unsealed) panel from Norgate and cache it. Returns the cache folder."""
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices

    name_str = name_str or index_name_str
    _, universe_df = build_index_constituent_matrix(index_name_str)
    universe_df = universe_df.loc[universe_df.index >= pd.Timestamp(start_date_str)]
    symbol_list = [symbol for symbol in universe_df.columns if universe_df[symbol].any()]
    price_df = load_raw_prices(symbol_list, [], start_date=start_date_str)
    loaded_symbol_list = sorted({symbol for symbol, _ in price_df.columns})
    date_index = price_df.index

    digest_obj = hashlib.sha256("\n".join(FIELD_TUPLE + ("|",) + tuple(loaded_symbol_list)).encode("utf-8"))
    frame_dict = {}
    for field_str in FIELD_TUPLE:
        frame = pd.DataFrame(
            {symbol: price_df[(symbol, field_str)] for symbol in loaded_symbol_list if (symbol, field_str) in price_df.columns},
            index=date_index,
        ).reindex(columns=loaded_symbol_list)
        frame_dict[field_str] = frame
        digest_obj.update(np.nan_to_num(frame.to_numpy(dtype="float64"), nan=-1.0).tobytes())
        digest_obj.update(frame.index.asi8.tobytes())
    # *** CRITICAL*** membership on a session is exactly Norgate's constituent flag on that session (no forward fill
    # across gaps inside the index calendar, no tail trim).
    member_df = universe_df.reindex(index=date_index, columns=loaded_symbol_list).fillna(0).astype(np.int8)
    digest_obj.update(pd.util.hash_pandas_object(member_df, index=True).to_numpy().tobytes())
    snapshot_id_str = digest_obj.hexdigest()[:16]

    folder_path = _panel_dir(name_str) / snapshot_id_str
    folder_path.mkdir(parents=True, exist_ok=True)
    for field_str, frame in frame_dict.items():
        frame.to_parquet(folder_path / _file_str(field_str))
    member_df.to_parquet(folder_path / "member.parquet")
    meta_dict = {
        "index_name_str": index_name_str,
        "start_date_str": start_date_str,
        "first_date_str": str(date_index[0].date()),
        "last_date_str": str(date_index[-1].date()),
        "symbol_count_int": len(loaded_symbol_list),
        "snapshot_id_str": snapshot_id_str,
        "built_utc_str": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    (folder_path / "meta.json").write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
    (_panel_dir(name_str) / "latest.json").write_text(json.dumps({"snapshot_id_str": snapshot_id_str}), encoding="utf-8")
    return folder_path


def _vault_opening_verified(ledger, vault_opening_row_id_int: int, family_id_str: str) -> bool:
    """The row exists in the verified chain, is a vault opening, and names this family."""
    ledger.verify()
    for row_dict in ledger.rows("vault_opening"):
        if row_dict.get("row_id_int") == vault_opening_row_id_int:
            return row_dict.get("family_id_str") == family_id_str
    return False


def load_panel(
    name_str: str,
    snapshot_id_str: str | None = None,
    vault_opening_row_id_int: int | None = None,
    family_id_str: str | None = None,
    ledger=None,
) -> Panel:
    """Load a cached panel snapshot (the latest by default), sealed at the vault unless the id of a vault-opening
    row for the family is given and the row is in the verified ledger chain."""
    base_path = _panel_dir(name_str)
    if snapshot_id_str is None:
        latest_path = base_path / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"No cached panel {name_str!r}; build it with build_panel().")
        snapshot_id_str = json.loads(latest_path.read_text(encoding="utf-8"))["snapshot_id_str"]
    folder_path = base_path / snapshot_id_str
    meta_path = folder_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No cached snapshot {snapshot_id_str!r} of panel {name_str!r}.")
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    field_dict = {field_str: pd.read_parquet(folder_path / _file_str(field_str)) for field_str in FIELD_TUPLE}
    member_df = pd.read_parquet(folder_path / "member.parquet")

    open_vault_bool = False
    if vault_opening_row_id_int is not None and family_id_str is not None:
        if ledger is None:
            from alpha.scout.ledger import Ledger

            ledger = Ledger()
        open_vault_bool = _vault_opening_verified(ledger, vault_opening_row_id_int, family_id_str)
    panel = Panel(
        name_str=name_str, field_dict=field_dict, member_df=member_df,
        snapshot_id_str=meta_dict["snapshot_id_str"], sealed_bool=not open_vault_bool,
    )
    if open_vault_bool:
        return panel
    # *** CRITICAL*** the seal: research code never receives a bar dated on or after the vault start.
    return panel.truncated(pd.Timestamp(VAULT_SEAL_STR) - pd.Timedelta(days=1))
