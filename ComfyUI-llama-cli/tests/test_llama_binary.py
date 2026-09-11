from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import llama_binary
import persist


def _write_required_files(install_dir: Path, spec: llama_binary.PlatformSpec) -> None:
    install_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.required_files:
        (install_dir / name).touch()


class PersistRootTests(unittest.TestCase):
    def test_notebooks_comfyui_layout_uses_parent_of_notebooks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            comfy = root / "notebooks" / "comfyui"
            comfy.mkdir(parents=True)
            persist._PERSIST_ROOT_OVERRIDE = None
            with mock.patch.object(persist, "_candidate_starts", return_value=[comfy]):
                self.assertEqual(persist.persist_root(), root)

    def test_existing_tmp_and_storage_win(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tmp").mkdir()
            (root / "storage").mkdir()
            nested = root / "other"
            nested.mkdir()
            persist._PERSIST_ROOT_OVERRIDE = None
            with mock.patch.object(persist, "_candidate_starts", return_value=[nested]):
                self.assertEqual(persist.persist_root(), root)

    def tearDown(self) -> None:
        persist._PERSIST_ROOT_OVERRIDE = None
        llama_binary._persist._PERSIST_ROOT_OVERRIDE = None


class ExistingInstallTests(unittest.TestCase):
    def test_only_current_release_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            llama_binary._persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            vendor_root = persist.storage_dir()
            for tag in ("old-release", llama_binary.LLAMA_CPP_RELEASE_TAG):
                install_dir = vendor_root / tag / llama_binary.WINDOWS_CUDA_13.key
                _write_required_files(install_dir, llama_binary.WINDOWS_CUDA_13)

            paths = llama_binary._existing_install(llama_binary.WINDOWS_CUDA_13)
            self.assertEqual(
                paths.cli,
                vendor_root
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.WINDOWS_CUDA_13.key
                / llama_binary.WINDOWS_CUDA_13.cli_executable,
            )

    def test_linux_current_release_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            llama_binary._persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            vendor_root = persist.storage_dir()
            for tag in ("old-release", llama_binary.LLAMA_CPP_RELEASE_TAG):
                install_dir = vendor_root / tag / llama_binary.LINUX_X64_CUDA.key
                _write_required_files(install_dir, llama_binary.LINUX_X64_CUDA)

            paths = llama_binary._existing_install(llama_binary.LINUX_X64_CUDA)
            self.assertEqual(
                paths.cli,
                vendor_root
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.LINUX_X64_CUDA.key
                / llama_binary.LINUX_X64_CUDA.cli_executable,
            )

    def tearDown(self) -> None:
        persist._PERSIST_ROOT_OVERRIDE = None
        llama_binary._persist._PERSIST_ROOT_OVERRIDE = None


class PlatformSpecTests(unittest.TestCase):
    def test_linux_x64_uses_cuda_source_spec(self) -> None:
        with mock.patch.object(llama_binary.platform, "system", return_value="Linux"):
            with mock.patch.object(llama_binary.platform, "machine", return_value="x86_64"):
                spec = llama_binary._platform_spec()
        self.assertEqual(spec, llama_binary.LINUX_X64_CUDA)
        self.assertEqual(spec.install_mode, llama_binary.INSTALL_SOURCE_CUDA)

    def test_windows_x64_keeps_release_zip_spec(self) -> None:
        with mock.patch.object(llama_binary.platform, "system", return_value="Windows"):
            with mock.patch.object(llama_binary.platform, "machine", return_value="AMD64"):
                spec = llama_binary._platform_spec()
        self.assertEqual(spec, llama_binary.WINDOWS_CUDA_13)
        self.assertEqual(spec.install_mode, llama_binary.INSTALL_RELEASE_ZIP)

    def test_unsupported_platform_still_raises(self) -> None:
        with mock.patch.object(llama_binary.platform, "system", return_value="Darwin"):
            with mock.patch.object(llama_binary.platform, "machine", return_value="arm64"):
                with self.assertRaisesRegex(RuntimeError, "Windows x64 CUDA 13 and Linux x64 CUDA"):
                    llama_binary._platform_spec()


class SourceInstallTests(unittest.TestCase):
    def test_cached_linux_install_skips_build_and_release_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            llama_binary._persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            install_dir = (
                persist.storage_dir()
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.LINUX_X64_CUDA.key
            )
            _write_required_files(install_dir, llama_binary.LINUX_X64_CUDA)

            with mock.patch.object(llama_binary, "_platform_spec", return_value=llama_binary.LINUX_X64_CUDA):
                with mock.patch.object(llama_binary, "_json_get") as json_get:
                    with mock.patch.object(llama_binary, "_build_cuda_from_source") as build:
                        paths = llama_binary.ensure_llama_cli_paths()

            self.assertEqual(paths.cli, install_dir / llama_binary.LINUX_X64_CUDA.cli_executable)
            json_get.assert_not_called()
            build.assert_not_called()

    def test_linux_source_install_does_not_query_release_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            llama_binary._persist._PERSIST_ROOT_OVERRIDE = Path(temp)

            def fake_build(install_dir: Path, spec: llama_binary.PlatformSpec, temp_dir: Path) -> None:
                _write_required_files(install_dir, spec)

            with mock.patch.object(llama_binary, "_platform_spec", return_value=llama_binary.LINUX_X64_CUDA):
                with mock.patch.object(llama_binary, "_json_get") as json_get:
                    with mock.patch.object(llama_binary, "_build_cuda_from_source", side_effect=fake_build):
                        with mock.patch.object(llama_binary.os, "chmod"):
                            paths = llama_binary.ensure_llama_cli_paths()

            json_get.assert_not_called()
            self.assertEqual(
                paths.cli,
                persist.storage_dir()
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.LINUX_X64_CUDA.key
                / llama_binary.LINUX_X64_CUDA.cli_executable,
            )
            self.assertFalse(llama_binary._persist.tmp_dir().exists())

    def test_failed_source_build_does_not_leave_install_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            llama_binary._persist._PERSIST_ROOT_OVERRIDE = Path(temp)
            install_dir = (
                persist.storage_dir()
                / llama_binary.LLAMA_CPP_RELEASE_TAG
                / llama_binary.LINUX_X64_CUDA.key
            )

            def fake_build(destination: Path, spec: llama_binary.PlatformSpec, temp_dir: Path) -> None:
                destination.mkdir(parents=True)
                (destination / spec.cli_executable).write_text("partial", encoding="utf-8")
                raise RuntimeError("cmake failed")

            with mock.patch.object(llama_binary, "_platform_spec", return_value=llama_binary.LINUX_X64_CUDA):
                with mock.patch.object(llama_binary, "_build_cuda_from_source", side_effect=fake_build):
                    with self.assertRaisesRegex(RuntimeError, "cmake failed"):
                        llama_binary.ensure_llama_cli_paths()

            self.assertFalse(install_dir.exists())
            self.assertFalse(llama_binary._persist.tmp_dir().exists())

    def tearDown(self) -> None:
        persist._PERSIST_ROOT_OVERRIDE = None
        llama_binary._persist._PERSIST_ROOT_OVERRIDE = None


class SourceRootTests(unittest.TestCase):
    def test_nested_github_archive_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            extract_dir = Path(temp)
            source_root = extract_dir / "llama.cpp-b10472"
            source_root.mkdir()
            (source_root / "CMakeLists.txt").touch()
            self.assertEqual(llama_binary._source_root(extract_dir), source_root)


class CopyRuntimeArtifactTests(unittest.TestCase):
    def test_copies_cli_and_shared_libs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            build_dir = Path(temp) / "build"
            bin_dir = build_dir / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "llama-cli").write_text("cli", encoding="utf-8")
            (bin_dir / "libggml.so").write_text("lib", encoding="utf-8")
            (bin_dir / "libllama.so.0").write_text("lib", encoding="utf-8")
            (bin_dir / "llama-server").write_text("skip", encoding="utf-8")
            (bin_dir / "notes.txt").write_text("skip", encoding="utf-8")
            install_dir = Path(temp) / "install"

            llama_binary._copy_runtime_artifacts(build_dir, install_dir, llama_binary.LINUX_X64_CUDA)

            names = {path.name for path in install_dir.iterdir()}
            self.assertEqual(names, {"llama-cli", "libggml.so", "libllama.so.0"})


class CMakeConfigureArgsTests(unittest.TestCase):
    def test_configure_args_do_not_disable_server(self) -> None:
        args = llama_binary._cmake_configure_args(
            "cmake",
            Path("/tmp/src"),
            Path("/tmp/build"),
        )
        self.assertNotIn("-DLLAMA_BUILD_SERVER=OFF", args)
        self.assertIn("-DGGML_CUDA=ON", args)
        self.assertIn("-DLLAMA_BUILD_TESTS=OFF", args)
        self.assertIn("-DLLAMA_BUILD_EXAMPLES=OFF", args)


class RunCommandTests(unittest.TestCase):
    def test_failed_command_includes_output_in_exception(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            llama_binary._run([
                sys.executable,
                "-c",
                "import sys; print('visible-build-error'); sys.stderr.write('stderr-line\\n'); sys.exit(2)",
            ])
        message = str(ctx.exception)
        self.assertIn("exit code 2", message)
        self.assertIn("visible-build-error", message)
        self.assertIn("stderr-line", message)

    def test_failed_command_keeps_only_the_last_output_lines(self) -> None:
        script = (
            "import sys\n"
            "for i in range(50):\n"
            "    print(f'line-{i}')\n"
            "sys.exit(1)\n"
        )
        with self.assertRaises(RuntimeError) as ctx:
            llama_binary._run([sys.executable, "-c", script])
        message = str(ctx.exception)
        self.assertNotIn("line-0", message)
        self.assertIn("line-49", message)


if __name__ == "__main__":
    unittest.main()
