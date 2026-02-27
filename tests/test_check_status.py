import unittest
import json
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing useapi_nodes
sys.modules["torch"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()

# Now import the target function
from useapi_nodes import _check_status

class TestCheckStatus(unittest.TestCase):
    def test_status_200_valid_json(self):
        body = b'{"key": "value"}'
        result = _check_status(200, body, "http://test.url")
        self.assertEqual(result, {"key": "value"})

    def test_status_200_empty_body(self):
        result = _check_status(200, b"", "http://test.url")
        self.assertEqual(result, {})

    def test_status_200_invalid_json(self):
        result = _check_status(200, b"invalid json", "http://test.url")
        self.assertEqual(result, {})

    def test_status_429(self):
        with self.assertRaisesRegex(RuntimeError, r"Rate Limited \(429\)"):
            _check_status(429, b"{}", "http://test.url")

    def test_status_503(self):
        with self.assertRaisesRegex(RuntimeError, r"Server Error \(503\)"):
            _check_status(503, b"{}", "http://test.url")

    def test_status_408(self):
        with self.assertRaisesRegex(RuntimeError, r"Request Timeout \(408\)"):
            _check_status(408, b"{}", "http://test.url")

    def test_status_401(self):
        with self.assertRaisesRegex(RuntimeError, r"Unauthorized \(401\)"):
            _check_status(401, b"{}", "http://test.url")

    def test_status_403_recaptcha_msg(self):
        body = json.dumps({"message": "Please solve the reCAPTCHA"}).encode()
        with self.assertRaisesRegex(RuntimeError, r"Transient reCAPTCHA error \(403\)"):
            _check_status(403, body, "http://test.url")

    def test_status_403_recaptcha_body(self):
        # Even if message key is missing, raw body search should find it
        body = b"Block due to captcha detection"
        with self.assertRaisesRegex(RuntimeError, r"Transient reCAPTCHA error \(403\)"):
            _check_status(403, body, "http://test.url")

    def test_status_403_api_error(self):
        body = json.dumps({"error": "API error: 403"}).encode()
        with self.assertRaisesRegex(RuntimeError, r'Google returned 403 to Useapi\.net \("API error: 403"\)'):
            _check_status(403, body, "http://test.url")

    def test_status_403_forbidden(self):
        body = json.dumps({"error": "Access denied"}).encode()
        with self.assertRaisesRegex(RuntimeError, r"Forbidden \(403\)"):
            _check_status(403, body, "http://test.url")

    def test_status_generic_error(self):
        with self.assertRaisesRegex(RuntimeError, r"Server Error \(500\)"):
            _check_status(500, b"Internal Server Error", "http://test.url")

    def test_context_logging(self):
        # Verify context is prepended
        with self.assertRaisesRegex(RuntimeError, r"\[MyContext\] Rate Limited"):
            _check_status(429, b"{}", "http://test.url", context="MyContext")
