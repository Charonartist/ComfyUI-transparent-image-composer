import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class TransparentImageComposer:
    """
    ComfyUIカスタムノード: 背景画像に透過画像を合成
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "x_position": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "y_position": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
            "optional": {
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light"],),
                "feather_edge": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "auto_transparent": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON",
                    "label_off": "OFF"
                }),
                "transparent_threshold": ("INT", {
                    "default": 240,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "transparent_color": ("STRING", {
                    "default": "white",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compose_images"
    CATEGORY = "image/compose"
    
    def tensor_to_pil(self, tensor):
        """テンソルをPIL画像に変換"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)  # バッチ次元を削除
        
        # テンソルの値域を[0,1]から[0,255]に変換
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        # numpy配列に変換
        numpy_image = tensor.cpu().numpy().astype(np.uint8)
        
        # チャンネル数に応じて処理
        if numpy_image.shape[2] == 3:  # RGB
            return Image.fromarray(numpy_image, 'RGB')
        elif numpy_image.shape[2] == 4:  # RGBA
            return Image.fromarray(numpy_image, 'RGBA')
        else:
            raise ValueError(f"Unsupported channel count: {numpy_image.shape[2]}")
    
    def pil_to_tensor(self, pil_image):
        """PIL画像をテンソルに変換"""
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)
        
        # バッチ次元を追加
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def parse_color(self, color_str):
        """色文字列をRGB値に変換"""
        color_map = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
        }
        
        color_str = color_str.lower().strip()
        
        # 名前から検索
        if color_str in color_map:
            return color_map[color_str]
        
        # RGB値として解析（例: "255,255,255" または "#ffffff"）
        if color_str.startswith('#'):
            try:
                hex_color = color_str[1:]
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                    return (r, g, b)
            except:
                pass
        
        if ',' in color_str:
            try:
                rgb = [int(x.strip()) for x in color_str.split(',')]
                if len(rgb) == 3:
                    return tuple(rgb)
            except:
                pass
        
        # デフォルトは白
        return (255, 255, 255)

    def create_alpha_mask(self, overlay_pil, feather_edge=0, auto_transparent=True, 
                         transparent_threshold=240, transparent_color="white"):
        """アルファマスクを作成（エッジのぼかし機能付き）"""
        if overlay_pil.mode == 'RGBA':
            alpha = overlay_pil.split()[3]
        elif auto_transparent:
            # 自動透過処理を行う
            target_rgb = self.parse_color(transparent_color)
            
            # RGB画像から指定した色に近い部分を透明にする
            np_img = np.array(overlay_pil)
            
            # 各ピクセルと目標色の距離を計算
            diff = np.abs(np_img.astype(np.float32) - np.array(target_rgb))
            distance = np.sqrt(np.sum(diff ** 2, axis=2))
            
            # 閾値を使って透過度を決定
            max_distance = np.sqrt(3 * (255 ** 2))  # RGB空間での最大距離
            threshold_distance = (255 - transparent_threshold) / 255.0 * max_distance
            
            # 距離に基づいてアルファ値を計算
            alpha_values = np.where(distance <= threshold_distance, 
                                  0,  # 透明
                                  np.clip((distance - threshold_distance) / threshold_distance * 255, 0, 255))
            
            alpha = Image.fromarray(alpha_values.astype(np.uint8), 'L')
        else:
            # 自動透過なしの場合、すべて不透明
            alpha = Image.new('L', overlay_pil.size, 255)
        
        # エッジをぼかす
        if feather_edge > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather_edge))
        
        return alpha
    
    def apply_blend_mode(self, background, overlay, mode):
        """ブレンドモードを適用"""
        if mode == "normal":
            return overlay
        
        # 正規化（0-1範囲）
        bg = np.array(background).astype(np.float32) / 255.0
        ov = np.array(overlay).astype(np.float32) / 255.0
        
        if mode == "multiply":
            result = bg * ov
        elif mode == "screen":
            result = 1 - (1 - bg) * (1 - ov)
        elif mode == "overlay":
            result = np.where(bg < 0.5, 2 * bg * ov, 1 - 2 * (1 - bg) * (1 - ov))
        elif mode == "soft_light":
            result = np.where(ov < 0.5, 
                            2 * bg * ov + bg**2 * (1 - 2 * ov),
                            2 * bg * (1 - ov) + np.sqrt(bg) * (2 * ov - 1))
        elif mode == "hard_light":
            result = np.where(ov < 0.5, 2 * bg * ov, 1 - 2 * (1 - bg) * (1 - ov))
        else:
            result = ov
        
        # 0-255範囲に戻す
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result, 'RGB')
    
    def compose_images(self, background_image, overlay_image, x_position, y_position, 
                      scale, opacity, blend_mode="normal", feather_edge=0, 
                      auto_transparent=True, transparent_threshold=240, transparent_color="white"):
        """画像を合成"""
        
        # テンソルをPIL画像に変換
        background_pil = self.tensor_to_pil(background_image)
        overlay_pil = self.tensor_to_pil(overlay_image)
        
        # オーバーレイ画像をRGBAに変換
        if overlay_pil.mode != 'RGBA':
            overlay_pil = overlay_pil.convert('RGBA')
        
        # スケール調整
        if scale != 1.0:
            new_width = int(overlay_pil.width * scale)
            new_height = int(overlay_pil.height * scale)
            overlay_pil = overlay_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 背景画像をRGBAに変換
        if background_pil.mode != 'RGBA':
            background_pil = background_pil.convert('RGBA')
        
        # 合成用の新しい画像を作成
        result_image = background_pil.copy()
        
        # オーバーレイ画像のアルファマスクを取得
        alpha_mask = self.create_alpha_mask(overlay_pil, feather_edge, auto_transparent, 
                                          transparent_threshold, transparent_color)
        
        # 透明度を適用
        if opacity < 1.0:
            alpha_mask = Image.eval(alpha_mask, lambda x: int(x * opacity))
        
        # ブレンドモードを適用（RGBチャンネルのみ）
        overlay_rgb = overlay_pil.convert('RGB')
        background_rgb = background_pil.convert('RGB')
        
        if blend_mode != "normal":
            # 合成領域のみを取得
            overlay_bbox = (max(0, x_position), 
                          max(0, y_position),
                          min(background_pil.width, x_position + overlay_pil.width),
                          min(background_pil.height, y_position + overlay_pil.height))
            
            if overlay_bbox[2] > overlay_bbox[0] and overlay_bbox[3] > overlay_bbox[1]:
                bg_crop = background_rgb.crop(overlay_bbox)
                
                # オーバーレイ画像の対応する部分を取得
                ov_x_start = max(0, -x_position)
                ov_y_start = max(0, -y_position)
                ov_x_end = ov_x_start + (overlay_bbox[2] - overlay_bbox[0])
                ov_y_end = ov_y_start + (overlay_bbox[3] - overlay_bbox[1])
                
                ov_crop = overlay_rgb.crop((ov_x_start, ov_y_start, ov_x_end, ov_y_end))
                alpha_crop = alpha_mask.crop((ov_x_start, ov_y_start, ov_x_end, ov_y_end))
                
                # ブレンドモードを適用
                blended = self.apply_blend_mode(bg_crop, ov_crop, blend_mode)
                
                # ブレンドした結果をRGBAに変換してアルファチャンネルを設定
                blended_rgba = blended.convert('RGBA')
                blended_rgba.putalpha(alpha_crop)
                
                # 結果を合成
                result_image.paste(blended_rgba, overlay_bbox[:2], blended_rgba)
        else:
            # 通常の合成
            result_image.paste(overlay_pil, (x_position, y_position), alpha_mask)
        
        # RGBに変換（ComfyUIはRGBを期待）
        result_image = result_image.convert('RGB')
        
        # テンソルに変換して返す
        return (self.pil_to_tensor(result_image),)


# ノード登録用のマップ
NODE_CLASS_MAPPINGS = {
    "TransparentImageComposer": TransparentImageComposer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparentImageComposer": "透過画像合成"
}
