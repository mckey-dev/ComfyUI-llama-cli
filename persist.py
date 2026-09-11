from __future__ import annotations

from pathlib import Path


PACKAGE_NAME = "ComfyUI-llama-cli"
_PERSIST_ROOT_OVERRIDE: Path | None = None


def _candidate_starts() -> list[Path]:
    starts: list[Path] = []
    cwd = Path.cwd().resolve()
    starts.append(cwd)
    try:
        import folder_paths
        starts.append(Path(folder_paths.base_path).resolve())
    except Exception:
        pass
    starts.append(Path(__file__).resolve().parent)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in starts:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def persist_root() -> Path:
    if _PERSIST_ROOT_OVERRIDE is not None:
        return _PERSIST_ROOT_OVERRIDE

    seen: set[Path] = set()
    for start in _candidate_starts():
        for path in (start, *start.parents):
            if path in seen:
                continue
            seen.add(path)
            if path.name.lower() == "comfyui" and path.parent.name.lower() == "notebooks":
                return path.parent.parent
            if (path / "tmp").is_dir() and (path / "storage").is_dir():
                return path

    try:
        import folder_paths
        return Path(folder_paths.base_path).resolve().parent
    except Exception:
        return Path.cwd().resolve()


def tmp_dir() -> Path:
    return persist_root() / "tmp" / PACKAGE_NAME


def storage_dir() -> Path:
    return persist_root() / "storage" / PACKAGE_NAME
