"""Structural validation for ComfyUI-UseapiNet nodes.
Runs without network access or API tokens.
"""
import sys
import os
import unittest

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

from useapi_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, _get_token, _extract_runway_task_id

EXPECTED_NODES = [
    "UseapiTokenFromEnv",
    "UseapiVeoGenerate",
    "UseapiVeoUpscale",
    "UseapiVeoExtend",
    "UseapiGoogleFlowGenerateImage",
    "UseapiGoogleFlowUploadAsset",
    "UseapiGoogleFlowImageUpscale",
    "UseapiRunwayUploadAsset",
    "UseapiRunwayGenerate",
    "UseapiRunwayVideoToVideo",
    "UseapiRunwayFramesGenerate",
    "UseapiRunwayImageUpscaler",
    "UseapiLoadVideoFrame",
    "UseapiPreviewVideo",
    "UseapiRunwayAleph",
    "UseapiRunwayGen3TurboExpand",
    "UseapiRunwayGen3TurboActOne",
]


class TestRegistrationMaps(unittest.TestCase):
    def test_all_nodes_in_class_mappings(self):
        for name in EXPECTED_NODES:
            self.assertIn(name, NODE_CLASS_MAPPINGS, f"{name} missing from NODE_CLASS_MAPPINGS")

    def test_all_nodes_in_display_name_mappings(self):
        for name in EXPECTED_NODES:
            self.assertIn(name, NODE_DISPLAY_NAME_MAPPINGS, f"{name} missing from NODE_DISPLAY_NAME_MAPPINGS")

    def test_class_mapping_values_are_classes(self):
        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertIsInstance(cls, type, f"{name} value is not a class")


class TestComfyUIContract(unittest.TestCase):
    def _check_node(self, name):
        cls = NODE_CLASS_MAPPINGS[name]
        self.assertTrue(hasattr(cls, "CATEGORY"), f"{name} missing CATEGORY")
        self.assertTrue(hasattr(cls, "FUNCTION"), f"{name} missing FUNCTION")
        self.assertTrue(hasattr(cls, "RETURN_TYPES"), f"{name} missing RETURN_TYPES")
        self.assertTrue(hasattr(cls, "RETURN_NAMES"), f"{name} missing RETURN_NAMES")
        func_name = cls.FUNCTION
        self.assertTrue(hasattr(cls, func_name), f"{name}.FUNCTION='{func_name}' but method not found")
        self.assertTrue(hasattr(cls, "INPUT_TYPES"), f"{name} missing INPUT_TYPES")
        input_types = cls.INPUT_TYPES()
        self.assertIsInstance(input_types, dict)
        self.assertTrue("required" in input_types or "optional" in input_types)
        self.assertIsInstance(cls.RETURN_TYPES, tuple)
        self.assertIsInstance(cls.RETURN_NAMES, tuple)
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES),
                         f"{name} RETURN_TYPES/RETURN_NAMES length mismatch")

    def test_all_nodes_satisfy_contract(self):
        for name in EXPECTED_NODES:
            if name not in NODE_CLASS_MAPPINGS:
                continue
            with self.subTest(node=name):
                self._check_node(name)


class TestCategories(unittest.TestCase):
    UTILS = ["UseapiTokenFromEnv", "UseapiLoadVideoFrame", "UseapiPreviewVideo"]
    GOOGLE_FLOW = [
        "UseapiVeoGenerate", "UseapiVeoUpscale", "UseapiVeoExtend",
        "UseapiGoogleFlowGenerateImage", "UseapiGoogleFlowUploadAsset",
        "UseapiGoogleFlowImageUpscale",
    ]
    RUNWAY = [
        "UseapiRunwayUploadAsset", "UseapiRunwayGenerate", "UseapiRunwayVideoToVideo",
        "UseapiRunwayFramesGenerate", "UseapiRunwayImageUpscaler",
        "UseapiRunwayAleph", "UseapiRunwayGen3TurboExpand",
        "UseapiRunwayGen3TurboActOne",
    ]

    def _check_category(self, names, expected_cat):
        for name in names:
            if name not in NODE_CLASS_MAPPINGS:
                continue
            with self.subTest(node=name):
                self.assertEqual(NODE_CLASS_MAPPINGS[name].CATEGORY, expected_cat)

    def test_utils_category(self):
        self._check_category(self.UTILS, "Useapi.net/Utils")

    def test_google_flow_category(self):
        self._check_category(self.GOOGLE_FLOW, "Useapi.net/Google Flow")

    def test_runway_category(self):
        self._check_category(self.RUNWAY, "Useapi.net/Runway")


class TestTokenUtility(unittest.TestCase):
    def setUp(self):
        os.environ.pop("USEAPI_TOKEN", None)

    def tearDown(self):
        os.environ.pop("USEAPI_TOKEN", None)

    def test_returns_direct_value(self):
        self.assertEqual(_get_token("abc123"), "abc123")

    def test_strips_whitespace(self):
        self.assertEqual(_get_token("  abc123  "), "abc123")

    def test_falls_back_to_env(self):
        os.environ["USEAPI_TOKEN"] = "env-token"
        self.assertEqual(_get_token(""), "env-token")

    def test_raises_when_no_token(self):
        with self.assertRaises(ValueError):
            _get_token("")

    def test_raises_when_whitespace_only(self):
        with self.assertRaises(ValueError):
            _get_token("   ")


class TestRunwayTaskIdExtraction(unittest.TestCase):
    def test_extracts_nested_task_id(self):
        data = {"task": {"taskId": "nested-123"}}
        self.assertEqual(_extract_runway_task_id(data), "nested-123")

    def test_extracts_top_level_task_id(self):
        data = {"taskId": "top-456"}
        self.assertEqual(_extract_runway_task_id(data), "top-456")

    def test_nested_takes_precedence(self):
        data = {"task": {"taskId": "nested-789"}, "taskId": "top-000"}
        self.assertEqual(_extract_runway_task_id(data), "nested-789")

    def test_returns_empty_string_if_missing(self):
        self.assertEqual(_extract_runway_task_id({}), "")
        self.assertEqual(_extract_runway_task_id({"other": "value"}), "")
        self.assertEqual(_extract_runway_task_id({"task": {}}), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
