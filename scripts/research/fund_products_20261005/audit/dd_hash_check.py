"""defensive_delta audit, step 1: stored sleeve files vs their recorded hashes, and module hashes vs HEAD (read-only)."""
import hashlib, json, os, datetime as dt
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
SRC = MAIN / "results/research/portfolio/shelf_rebuild_20260929/sources"
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/defensive_delta"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_lf(p):  # sha of the file with CRLF normalised to LF (git autocrlf check)
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
def mt(p): return dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")

rows = []
for mp in sorted(SRC.glob("*__metadata.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    a = m["alias_str"]
    path_f, tx_f = SRC / f"{a}__path.csv.gz", SRC / f"{a}__transactions.csv.gz"
    mod_wt, mod_main = WT / m["module_path_str"], MAIN / m["module_path_str"]
    rows.append({
        "alias": a, "tier": m["tier_str"], "module": m["module_path_str"],
        "path_ok": sha(path_f) == m["path_sha256_str"], "tx_ok": sha(tx_f) == m["transaction_sha256_str"],
        "path_mtime": mt(path_f),
        "module_head_same": mod_wt.exists() and (sha(mod_wt) == m["module_sha256_str"] or sha_lf(mod_wt) == m["module_sha256_str"]),
        "module_main_worktree_same": mod_main.exists() and (sha(mod_main) == m["module_sha256_str"] or sha_lf(mod_main) == m["module_sha256_str"]),
        "module_exists_head": mod_wt.exists(),
    })
for r in rows:
    print(r)
(OUT / "hash_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
