"""Build and resolve the read-only Knowledge Base served by BENCH."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
MKDOCS_CONFIG_PATH = REPO_ROOT_PATH / "mkdocs.yml"
KNOWLEDGE_SITE_DIR_PATH = REPO_ROOT_PATH / ".codex_tmp" / "knowledge-site"


def build_knowledge_site() -> Path:
    """Run the canonical strict MkDocs build and return its site directory."""

    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    mkdocs_config_obj = load_config(
        config_file=str(MKDOCS_CONFIG_PATH),
        site_dir=str(KNOWLEDGE_SITE_DIR_PATH),
        strict=True,
    )
    build(mkdocs_config_obj, dirty=False)

    index_path_obj = KNOWLEDGE_SITE_DIR_PATH / "index.html"
    if not index_path_obj.is_file():
        raise RuntimeError("MkDocs completed without producing index.html.")
    return KNOWLEDGE_SITE_DIR_PATH.resolve()


def resolve_knowledge_file(
    knowledge_site_root_path_obj: Path,
    rel_path_str: str,
) -> Path | None:
    """Resolve one built page or asset without allowing path traversal."""

    knowledge_site_root_path_obj = knowledge_site_root_path_obj.resolve()
    requested_path_obj = knowledge_site_root_path_obj / (rel_path_str or "index.html")
    if requested_path_obj.is_dir():
        requested_path_obj = requested_path_obj / "index.html"
    resolved_path_obj = requested_path_obj.resolve()
    if not resolved_path_obj.is_relative_to(knowledge_site_root_path_obj):
        return None
    if not resolved_path_obj.is_file():
        return None
    return resolved_path_obj
