"""
ComfyUI 透過画像合成カスタムノード
背景画像に透過画像を合成するためのカスタムノード
"""

from .transparent_image_composer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ComfyUIにノードを登録
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
