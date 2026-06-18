import tempfile
import unittest
import importlib
import sys
import types
from pathlib import Path
from unittest import mock

from utils.download import default_models_dir, read_manifest, verify_manifest, write_manifest


class TorqExamplesIntegrationTest(unittest.TestCase):
    def test_torq_examples_import_points_at_submodule(self):
        torq_examples = importlib.import_module("third_party.torq_examples")
        self.assertEqual(
            Path(torq_examples.__path__[0]).name,
            "torq_examples",
        )
        self.assertTrue((Path(torq_examples.__path__[0]) / "README.md").is_file())

    def test_default_models_dir_is_sl2610_rooted(self):
        self.assertEqual(
            default_models_dir(),
            Path(__file__).resolve().parents[1] / "models",
        )

    def test_manifest_helpers_are_revision_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            (model_dir / "a.vmfb").parent.mkdir(parents=True)
            (model_dir / "a.vmfb").write_text("model")

            write_manifest(model_dir, "org/repo", ["a.vmfb"], revision="abc123")
            manifest = read_manifest(model_dir)
            is_valid = verify_manifest(model_dir)

        self.assertEqual(manifest["revision"], "abc123")
        self.assertTrue(is_valid)

    def test_public_gemma_and_moonshine_imports_stay_stable(self):
        from utils.gemma import download_gemma3, load_gemma, local_gemma3_model_dir
        from utils.moonshine import download_moonshine, load_moonshine, local_moonshine_model_dir

        self.assertTrue(callable(download_gemma3))
        self.assertTrue(callable(load_gemma))
        self.assertTrue(callable(local_gemma3_model_dir))
        self.assertTrue(callable(download_moonshine))
        self.assertTrue(callable(load_moonshine))
        self.assertTrue(callable(local_moonshine_model_dir))

    def test_gemma_download_delegates_to_torq_refresh(self):
        from utils.gemma import download as gemma_download

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with (
                mock.patch.object(gemma_download, "default_models_dir", return_value=base_dir),
                mock.patch.object(gemma_download._torq_gemma_setup, "_refresh_gemma3") as refresh,
            ):
                result = gemma_download.download_gemma3(["instruct"])

        repo_id = gemma_download.GEMMA3_HF_REPO_MAP["instruct"]
        self.assertEqual(result["instruct"], base_dir / repo_id)
        refresh.assert_called_once_with(repo_id, base_dir / repo_id, base_dir)

    def test_moonshine_download_delegates_to_torq_refresh(self):
        from utils.moonshine import download as moonshine_download

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with (
                mock.patch.object(moonshine_download, "default_models_dir", return_value=base_dir),
                mock.patch.object(
                    moonshine_download._torq_moonshine_setup,
                    "_refresh_moonshine",
                ) as refresh,
            ):
                result = moonshine_download.download_moonshine(["tiny-en"])

        repo_id = moonshine_download.MOONSHINE_HF_REPO_MAP["tiny-en"]
        self.assertEqual(result["tiny-en"], base_dir / repo_id)
        refresh.assert_called_once_with(repo_id, base_dir / repo_id, base_dir)

    def test_gemma_torq_stream_accumulates_chunks_and_stats(self):
        from utils.gemma import runner as gemma_runner

        class FakeGemma3Static:
            max_seq_len = 16
            last_infer_time = 12.5
            time_to_first_token = 3.5
            generated_tokens = 2

            def run_stream(self, query):
                self.query = query
                yield "hel"
                yield "lo"

            def _build_prompt_tokens(self, query):
                return [1, 2, 3]

            def _apply_prompt_limit(self, tokens):
                return tokens + [0]

        fake_runner = FakeGemma3Static()

        fake_module = types.ModuleType("third_party.torq_examples.gemma3.src.runner")
        fake_module.Gemma3Static = mock.Mock(return_value=fake_runner)

        with mock.patch.dict(
            sys.modules,
            {"third_party.torq_examples.gemma3.src.runner": fake_module},
        ):
            backend = gemma_runner.GemmaTorq("model.vmfb")
            partials = list(backend.stream_response("translate me"))

        self.assertEqual(partials, ["hel", "hello"])
        self.assertEqual(backend.last_n_input_tokens, 3)
        self.assertEqual(backend.last_n_prefill_tokens, 4)
        self.assertEqual(backend.last_n_output_tokens, 2)
        self.assertEqual(backend.last_infer_time_ms, 12.5)
        self.assertEqual(backend.time_to_first_token_ms, 3.5)


if __name__ == "__main__":
    unittest.main()
