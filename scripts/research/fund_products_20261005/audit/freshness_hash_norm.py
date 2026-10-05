"""Recorded module / dependency hashes of the shelf rebuild vs the git blobs at f9ad358 and HEAD (LF or CRLF bytes)."""
import hashlib, json, subprocess
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
SR = Path(r"C:\Users\User\Documents\workspace\alpha_super\results\research\portfolio\shelf_rebuild_20260929")
RUN, HEAD = "f9ad358", "5c0d48d"


def blob(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=WT, capture_output=True, check=True).stdout


def variants(b):
    lf = b.replace(b"\r\n", b"\n")
    return {hashlib.sha256(lf).hexdigest(): "LF", hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(): "CRLF"}


def blob_id(rev, path):
    return subprocess.run(["git", "rev-parse", f"{rev}:{path}"], cwd=WT, capture_output=True, text=True, check=True).stdout.strip()


rows = []
for meta_path in sorted((SR / "sources").glob("*__metadata.json")):
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    p = m["module_path_str"]
    v = variants(blob(RUN, p))
    rows.append((m["alias_str"], p, v.get(m["module_sha256_str"], "NO MATCH"), blob_id(RUN, p) == blob_id(HEAD, p)))
rec = None
for line in (SR / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    if d.get("event_str") == "sleeve_runs_started" and d.get("only_list") is None:
        rec = d["shared_execution_dependency_hash_dict"]
for key, old in rec.items():
    if key.startswith("python_tree::"):
        continue
    v = variants(blob(RUN, key))
    rows.append(("(dependency)", key, v.get(old, "NO MATCH"), blob_id(RUN, key) == blob_id(HEAD, key)))
print(f"{'alias':<20}{'recorded hash == f9ad358 blob as':<34}{'blob f9ad358 == HEAD':<22}path")
for a, p, how, same in rows:
    print(f"{a:<20}{how:<34}{str(same):<22}{p}")
out = WT / "results/research/portfolio/fund_products_20261005/audit/freshness/hash_norm.json"
out.write_text(json.dumps([dict(alias=a, path=p, recorded_matches_run_commit_blob=how, blob_unchanged_to_head=same) for a, p, how, same in rows], indent=1))
