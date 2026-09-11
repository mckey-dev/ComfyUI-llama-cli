from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


def _load_persist():
    persist_path = Path(__file__).resolve().parent / "persist.py"
    spec = importlib.util.spec_from_file_location("comfyui_llama_cli_persist", persist_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load persist helper from {persist_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_persist = _load_persist()
storage_dir = _persist.storage_dir
tmp_dir = _persist.tmp_dir


LLAMA_CPP_RELEASE_TAG = "b10472"
RELEASE_API_URL = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_CPP_RELEASE_TAG}"
SOURCE_ARCHIVE_URL = (
    f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_RELEASE_TAG}.tar.gz"
)
INSTALL_RELEASE_ZIP = "release_zip"
INSTALL_SOURCE_CUDA = "source_cuda"
LOG_PREFIX = "[ComfyUI-llama-cli]"
USER_AGENT = "ComfyUI-llama-cli"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    cli_executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]
    install_mode: str = INSTALL_RELEASE_ZIP


@dataclass(frozen=True)
class LlamaCliPaths:
    cli: Path


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    cli_executable="llama-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-cli.exe",
        "ggml-cuda.dll",
        "cudart64_13.dll",
    ),
)

LINUX_X64_CUDA = PlatformSpec(
    key="linux-x64-cuda",
    cli_executable="llama-cli",
    asset_patterns=(),
    required_files=("llama-cli",),
    install_mode=INSTALL_SOURCE_CUDA,
)


def storage_root() -> Path:
    return storage_dir()


def _platform_spec() -> PlatformSpec:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return WINDOWS_CUDA_13
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return LINUX_X64_CUDA
    raise RuntimeError(
        "Automatic llama.cpp setup currently supports Windows x64 CUDA 13 and Linux x64 CUDA. "
        "Other platforms are intentionally isolated behind the platform mapping for future support."
    )


def _json_get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_size(num_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size is not None else None
        downloaded = 0
        chunk_size = 1024 * 256
        started_at = time.monotonic()
        last_reported_at = started_at

        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_reported_at < 1.0:
                    continue

                elapsed = max(now - started_at, 0.001)
                speed = downloaded / elapsed
                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(
                        f"{LOG_PREFIX} "
                        f"Downloaded {_format_size(downloaded)} / {_format_size(total_size)} "
                        f"({percent:.1f}%) at {_format_size(speed)}/s"
                    )
                else:
                    print(
                        f"{LOG_PREFIX} "
                        f"Downloaded {_format_size(downloaded)} at {_format_size(speed)}/s"
                    )
                last_reported_at = now

        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = downloaded / elapsed
        if total_size:
            print(
                f"{LOG_PREFIX} "
                f"Finished download: {_format_size(downloaded)} / {_format_size(total_size)} "
                f"(100.0%) at {_format_size(speed)}/s"
            )
        else:
            print(
                f"{LOG_PREFIX} "
                f"Finished download: {_format_size(downloaded)} at {_format_size(speed)}/s"
            )


def _select_assets(release: dict, spec: PlatformSpec) -> list[dict]:
    assets = release.get("assets", [])
    selected = []
    used_names = set()

    for pattern in spec.asset_patterns:
        matches = [
            asset for asset in assets
            if fnmatch.fnmatch(asset.get("name", "").lower(), pattern.lower())
        ]
        if not matches:
            raise RuntimeError(f"Could not find llama.cpp release asset matching: {pattern}")
        asset = sorted(matches, key=lambda item: item.get("name", ""))[0]
        if asset["name"] not in used_names:
            selected.append(asset)
            used_names.add(asset["name"])
    return selected


def _find_file(install_dir: Path, name: str) -> Path | None:
    for path in install_dir.rglob(name):
        if path.is_file():
            return path
    return None


def _find_cli_paths(install_dir: Path, spec: PlatformSpec) -> LlamaCliPaths | None:
    cli = _find_file(install_dir, spec.cli_executable)
    if cli is None:
        return None
    return LlamaCliPaths(cli=cli)


def _has_required_files(install_dir: Path, spec: PlatformSpec) -> bool:
    for name in spec.required_files:
        if not any(path.is_file() for path in install_dir.rglob(name)):
            return False
    return True


def _is_complete_install(install_dir: Path, spec: PlatformSpec) -> bool:
    return _find_cli_paths(install_dir, spec) is not None and _has_required_files(install_dir, spec)


def _install_dir(spec: PlatformSpec, tag: str = LLAMA_CPP_RELEASE_TAG) -> Path:
    return storage_root() / tag / spec.key


def _existing_install(spec: PlatformSpec) -> LlamaCliPaths | None:
    install_dir = _install_dir(spec)
    return _find_cli_paths(install_dir, spec) if _is_complete_install(install_dir, spec) else None


def _reset_tmp() -> Path:
    path = tmp_dir()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _clear_tmp() -> None:
    path = tmp_dir()
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _copy_tree_files(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        target = dest / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _extract_assets(assets: list[dict], install_dir: Path, temp_dir: Path) -> None:
    extract_dir = temp_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        archive_path = temp_dir / asset["name"]
        print(f"{LOG_PREFIX} Downloading {asset['name']}...")
        _download(asset["browser_download_url"], archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
    if install_dir.exists():
        shutil.rmtree(install_dir)
    _copy_tree_files(extract_dir, install_dir)


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Linux CUDA llama.cpp setup requires `{name}` on PATH. "
            "Install cmake and the CUDA toolkit, then run the node again."
        )
    return path


def _run(command: list[str]) -> None:
    print(f"{LOG_PREFIX} {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    output_lines = []
    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                print(f"{LOG_PREFIX} {line}")
                output_lines.append(line)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        returncode = process.wait()
    if returncode != 0:
        tail = "\n".join(output_lines[-40:])
        raise RuntimeError(
            f"llama.cpp build command failed with exit code {returncode}: {' '.join(command)}\n{tail}"
        )


def _source_root(extract_dir: Path) -> Path:
    if (extract_dir / "CMakeLists.txt").is_file():
        return extract_dir
    children = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(children) == 1 and (children[0] / "CMakeLists.txt").is_file():
        return children[0]
    raise RuntimeError(f"Could not find llama.cpp CMakeLists.txt in {extract_dir}")


def _extract_tar_gz(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def _is_shared_lib(path: Path) -> bool:
    name = path.name
    return path.suffix == ".so" or ".so." in name or path.suffix == ".dylib"


def _copy_runtime_artifacts(build_dir: Path, install_dir: Path, spec: PlatformSpec) -> None:
    cli_src = None
    libs = []
    for path in build_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == spec.cli_executable:
            cli_src = path
        elif _is_shared_lib(path):
            libs.append(path)
    if cli_src is None:
        raise RuntimeError(f"llama.cpp build did not produce {spec.cli_executable} in {build_dir}")

    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True)
    shutil.copy2(cli_src, install_dir / spec.cli_executable)
    for src in libs:
        shutil.copy2(src, install_dir / src.name)


def _cmake_configure_args(cmake: str, source_root: Path, build_dir: Path) -> list[str]:
    return [
        cmake,
        "-S", str(source_root),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=native",
    ]


def _build_cuda_from_source(install_dir: Path, spec: PlatformSpec, temp_dir: Path) -> None:
    cmake = _require_tool("cmake")
    _require_tool("nvcc")
    jobs = os.cpu_count() or 1

    archive_path = temp_dir / f"llama.cpp-{LLAMA_CPP_RELEASE_TAG}.tar.gz"
    print(f"{LOG_PREFIX} Downloading llama.cpp {LLAMA_CPP_RELEASE_TAG} source...")
    _download(SOURCE_ARCHIVE_URL, archive_path)

    source_dir = temp_dir / "src"
    source_dir.mkdir()
    _extract_tar_gz(archive_path, source_dir)
    source_root = _source_root(source_dir)
    build_dir = temp_dir / "build"

    print(f"{LOG_PREFIX} Configuring llama.cpp CUDA build...")
    _run(_cmake_configure_args(cmake, source_root, build_dir))
    print(f"{LOG_PREFIX} Building llama-cli. This can take a while...")
    _run([
        cmake,
        "--build", str(build_dir),
        "--config", "Release",
        "--parallel", str(jobs),
        "--target", spec.cli_executable,
    ])
    print(f"{LOG_PREFIX} Installing llama.cpp runtime files...")
    _copy_runtime_artifacts(build_dir, install_dir, spec)


def _complete_paths(install_dir: Path, spec: PlatformSpec, action: str) -> LlamaCliPaths:
    paths = _find_cli_paths(install_dir, spec)
    if paths is None:
        raise RuntimeError(f"{action} llama.cpp but could not find CLI executables in {install_dir}")
    if not _has_required_files(install_dir, spec):
        missing = [
            name for name in spec.required_files
            if not any(path.is_file() for path in install_dir.rglob(name))
        ]
        raise RuntimeError(f"{action} llama.cpp install is incomplete; missing: {', '.join(missing)}")
    if spec.install_mode == INSTALL_SOURCE_CUDA:
        os.chmod(paths.cli, 0o755)
    return paths


def _install_from_source(spec: PlatformSpec) -> LlamaCliPaths:
    install_dir = _install_dir(spec)
    if _is_complete_install(install_dir, spec):
        paths = _find_cli_paths(install_dir, spec)
        if paths is None:
            raise RuntimeError(f"Completed install has incomplete CLI executables: {install_dir}")
        return paths

    temp_dir = _reset_tmp()
    try:
        _build_cuda_from_source(install_dir, spec, temp_dir)
        return _complete_paths(install_dir, spec, "Built")
    except BaseException:
        if install_dir.exists():
            shutil.rmtree(install_dir, ignore_errors=True)
        raise
    finally:
        _clear_tmp()


def ensure_llama_cli_paths() -> LlamaCliPaths:
    spec = _platform_spec()
    existing = _existing_install(spec)
    if existing is not None:
        return existing

    if spec.install_mode == INSTALL_SOURCE_CUDA:
        return _install_from_source(spec)

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or LLAMA_CPP_RELEASE_TAG
    install_dir = _install_dir(spec, tag)

    if _is_complete_install(install_dir, spec):
        paths = _find_cli_paths(install_dir, spec)
        if paths is None:
            raise RuntimeError(f"Completed install has incomplete CLI executables: {install_dir}")
        return paths

    temp_dir = _reset_tmp()
    try:
        assets = _select_assets(release, spec)
        _extract_assets(assets, install_dir, temp_dir)
        return _complete_paths(install_dir, spec, "Downloaded")
    except BaseException:
        if install_dir.exists() and not _is_complete_install(install_dir, spec):
            shutil.rmtree(install_dir, ignore_errors=True)
        raise
    finally:
        _clear_tmp()
