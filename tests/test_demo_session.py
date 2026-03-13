import unittest

from app.core.types import BackchannelItem
from app.demo.session import MeasurementControl, _paused_signal, _pick_demo_fallback_item


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


if __name__ == "__main__":
    unittest.main()
