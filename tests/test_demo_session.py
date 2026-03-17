import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.types import BackchannelItem
from app.demo.session import (
    MeasurementControl,
    _manual_key_guide,
    _manual_key_map,
    _paused_signal,
    _pick_demo_fallback_item,
)


class DemoSessionTests(unittest.TestCase):
    def test_pick_demo_fallback_item_for_nod(self) -> None:
        items = [
            BackchannelItem(id="01", directory="positive", strength=1, nod=1, text="Yes."),
            BackchannelItem(id="03", directory="positive", strength=3, nod=3, text="I see."),
            BackchannelItem(id="08", directory="negative", strength=1, nod=1, text="Hmm?"),
        ]

        item = _pick_demo_fallback_item(
            items=items,
            hint="nod",
            gesture_intensity=None,
            avoid_ids=[],
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.id, "01")

    def test_pick_demo_fallback_item_respects_avoid_ids(self) -> None:
        items = [
            BackchannelItem(id="08", directory="negative", strength=1, nod=1, text="Hmm?"),
            BackchannelItem(id="09", directory="negative", strength=1, nod=1, text="Really?"),
        ]

        item = _pick_demo_fallback_item(
            items=items,
            hint="shake",
            gesture_intensity=None,
            avoid_ids=["08"],
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.id, "09")

    def test_measurement_control_toggles_state(self) -> None:
        control = MeasurementControl(enabled=False)

        self.assertFalse(control.is_enabled())
        self.assertTrue(control.toggle())
        self.assertTrue(control.is_enabled())
        self.assertFalse(control.set_enabled(False))

    def test_paused_signal_is_non_present(self) -> None:
        signal = _paused_signal()

        self.assertFalse(signal["present"])
        self.assertEqual(signal["gesture_hint"], "other")
        self.assertEqual(signal["reason"], "paused")

    def test_manual_key_map_uses_expected_keys(self) -> None:
        items = [
            BackchannelItem(id="01", directory="positive", strength=1, nod=1, text="Yes."),
            BackchannelItem(id="08", directory="negative", strength=1, nod=1, text="Hmm?"),
        ]
        with TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            positive = audio_dir / "positive"
            negative = audio_dir / "negative"
            positive.mkdir()
            negative.mkdir()
            (positive / "01_s1_n1_yes.mp3").write_bytes(b"yes")
            (negative / "08_s1_n1_hmm.mp3").write_bytes(b"hmm")

            mapping = _manual_key_map(items, audio_dir)

            self.assertEqual(mapping["1"][0].id, "01")
            self.assertEqual(mapping["1"][1], positive / "01_s1_n1_yes.mp3")
            self.assertEqual(mapping["a"][0].id, "08")
            self.assertEqual(mapping["a"][1], negative / "08_s1_n1_hmm.mp3")

    def test_manual_key_guide_lists_controls(self) -> None:
        mapping = {
            "1": (
                BackchannelItem(id="01", directory="positive", strength=1, nod=1, text="Yes."),
                Path("/tmp/positive/01_s1_n1_yes.mp3"),
            ),
            "a": (
                BackchannelItem(id="08", directory="negative", strength=1, nod=1, text="Hmm?"),
                Path("/tmp/negative/08_s1_n1_hmm.mp3"),
            ),
        }

        guide = _manual_key_guide(mapping)

        self.assertIn("1:Yes.", guide)
        self.assertIn("a:Hmm?", guide)
        self.assertIn("Enter: ON/OFF", guide)
        self.assertIn("q: 終了", guide)


if __name__ == "__main__":
    unittest.main()
