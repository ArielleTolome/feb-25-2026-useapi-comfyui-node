"""Unit tests for API token precedence logic in Useapi.net nodes."""
import sys
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock heavy deps if not in ComfyUI env
try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    from unittest.mock import MagicMock
    torch = MagicMock()
    np = MagicMock()
    sys.modules["torch"] = torch
    sys.modules["numpy"] = np

try:
    import cv2
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["cv2"] = MagicMock()

from useapi_nodes import _get_token


class TestTokenPrecedence(unittest.TestCase):
    """Test the priority logic of _get_token: direct arg > env var > error."""

    def test_priority_direct_over_env(self):
        """Verify that a direct argument takes precedence over the environment variable."""
        with patch.dict(os.environ, {"USEAPI_TOKEN": "env_token_value"}):
            token = _get_token("direct_token_value")
            self.assertEqual(token, "direct_token_value")

    def test_fallback_to_env_if_arg_empty(self):
        """Verify fallback to environment variable if argument is empty string."""
        with patch.dict(os.environ, {"USEAPI_TOKEN": "env_token_value"}):
            token = _get_token("")
            self.assertEqual(token, "env_token_value")

    def test_fallback_to_env_if_arg_none(self):
        """Verify fallback to environment variable if argument is None."""
        with patch.dict(os.environ, {"USEAPI_TOKEN": "env_token_value"}):
            token = _get_token(None)
            self.assertEqual(token, "env_token_value")

    def test_fallback_to_env_if_arg_whitespace(self):
        """Verify fallback to environment variable if argument is only whitespace."""
        with patch.dict(os.environ, {"USEAPI_TOKEN": "env_token_value"}):
            token = _get_token("   ")
            self.assertEqual(token, "env_token_value")

    def test_direct_only_no_env(self):
        """Verify that direct argument works when environment variable is unset."""
        with patch.dict(os.environ, {}, clear=True):
            token = _get_token("direct_token_value")
            self.assertEqual(token, "direct_token_value")

    def test_no_token_raises_error(self):
        """Verify that ValueError is raised if neither argument nor env var is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as cm:
                _get_token("")
            self.assertIn("API token is missing", str(cm.exception))

    def test_whitespace_stripping(self):
        """Verify that whitespace is stripped from both direct argument and env var."""
        # Case 1: Direct argument with whitespace
        with patch.dict(os.environ, {}, clear=True):
            token = _get_token("  token_with_spaces  ")
            self.assertEqual(token, "token_with_spaces")

        # Case 2: Env var with whitespace (arg empty)
        with patch.dict(os.environ, {"USEAPI_TOKEN": "  env_token_spaces  "}):
            token = _get_token("")
            self.assertEqual(token, "env_token_spaces")


if __name__ == "__main__":
    unittest.main(verbosity=2)
