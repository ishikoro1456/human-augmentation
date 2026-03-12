import tempfile
import unittest
from pathlib import Path

from app.demo.script import load_demo_script


class DemoScriptTests(unittest.TestCase):
    def test_cue_is_excluded_from_spoken_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.json"
            path.write_text(
                """
{
  "id": "demo",
  "title": "demo",
  "language": "en",
  "audio_dir": "../audio",
  "segments": [
    {"kind": "speech", "text": "Line 1", "audio": "001.mp3"},
    {"kind": "cue", "text": "operator cue"},
    {"kind": "speech", "text": "Line 2", "audio": "002.mp3"}
  ]
}
""".strip(),
                encoding="utf-8",
            )
            script = load_demo_script(path)

        self.assertEqual(script.spoken_context(upto_index=1), "Line 1")
        self.assertEqual(script.spoken_context(upto_index=2), "Line 1\nLine 2")


if __name__ == "__main__":
    unittest.main()
