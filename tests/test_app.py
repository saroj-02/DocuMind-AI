import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_app_renders_without_retrieval_count_control(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.slider), 0)
        self.assertEqual(len(app.get("file_uploader")), 1)


if __name__ == "__main__":
    unittest.main()
