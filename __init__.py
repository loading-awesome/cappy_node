"""Cappy H3 — measured acceleration for ComfyUI's MiniMax H3."""

from .nodes import CappyMiniMaxH3AudioAwareCache, CappyMiniMaxH3FastPath

NODE_CLASS_MAPPINGS = {
    "CappyMiniMaxH3AudioAwareCache": CappyMiniMaxH3AudioAwareCache,
    "CappyMiniMaxH3FastPath": CappyMiniMaxH3FastPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CappyMiniMaxH3AudioAwareCache": "Cappy MiniMax H3 Audio-Aware Cache",
    "CappyMiniMaxH3FastPath": "Cappy MiniMax H3 Fast Path (exact)",
}
