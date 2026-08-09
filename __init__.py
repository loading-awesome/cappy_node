"""Cappy H3 Cache — audio-aware residual reuse for ComfyUI's MiniMax H3."""

from .nodes import CappyMiniMaxH3AudioAwareCache

NODE_CLASS_MAPPINGS = {
    "CappyMiniMaxH3AudioAwareCache": CappyMiniMaxH3AudioAwareCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CappyMiniMaxH3AudioAwareCache": "Cappy MiniMax H3 Audio-Aware Cache",
}

