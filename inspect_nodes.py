import sys
from unittest.mock import MagicMock

sys.modules["torch"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["cv2"] = MagicMock()

import useapi_nodes

nodes_to_inspect = [
    "UseapiTokenFromEnv",
    "UseapiVeoGenerate",
    "UseapiGoogleFlowGenerateImage",
    "UseapiRunwayGenerate",
    "UseapiRunwayFramesGenerate",
    "UseapiGoogleFlowImageUpscale",
    "UseapiPreviewVideo",
    "UseapiRunwayImages",
]

for node_name in nodes_to_inspect:
    cls = getattr(useapi_nodes, node_name)
    input_types = cls.INPUT_TYPES()
    widgets = []
    # Order: required, then optional
    for key in input_types.get("required", {}):
        widgets.append(key)
    for key in input_types.get("optional", {}):
        widgets.append(key)
    print(f"Node: {node_name}")
    print(f"Widgets: {widgets}")
    print("-" * 20)
