#!/usr/bin/env python3
"""
透過画像合成ノードのテストスクリプト
ComfyUIの環境外でノードの基本機能をテストします
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

# テスト用の画像を作成
def create_test_images():
    """テスト用の背景画像と透過画像を作成"""
    
    # 背景画像（青いグラデーション）
    background = Image.new('RGB', (800, 600), color='lightblue')
    for y in range(600):
        for x in range(800):
            r = int(173 + (82 * x / 800))  # lightblue to blue gradient
            g = int(216 + (39 * y / 600))
            b = 230
            background.putpixel((x, y), (r, g, b))
    
    # 透過画像（赤い円）
    overlay = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay)
    draw.ellipse([20, 20, 180, 180], fill=(255, 0, 0, 200), outline=(128, 0, 0, 255))
    draw.text((85, 95), "TEST", fill=(255, 255, 255, 255))
    
    return background, overlay

def pil_to_tensor(pil_image):
    """PIL画像をテンソルに変換（ComfyUI形式）"""
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    numpy_image = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(numpy_image)
    
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)  # バッチ次元を追加
    
    return tensor

def test_transparent_image_composer():
    """ノードの基本機能をテスト"""
    
    try:
        # ノードクラスをインポート
        from transparent_image_composer import TransparentImageComposer
        
        # ノードインスタンスを作成
        composer = TransparentImageComposer()
        
        # テスト画像を作成
        background_pil, overlay_pil = create_test_images()
        
        # PIL画像をテンソルに変換
        background_tensor = pil_to_tensor(background_pil)
        overlay_tensor = pil_to_tensor(overlay_pil)
        
        print("入力画像のテンソル形状:")
        print(f"Background: {background_tensor.shape}")
        print(f"Overlay: {overlay_tensor.shape}")
        
        # ノードを実行
        print("\n基本的な合成をテスト中...")
        result = composer.compose_images(
            background_image=background_tensor,
            overlay_image=overlay_tensor,
            x_position=300,
            y_position=200,
            scale=1.5,
            opacity=0.8,
            blend_mode="normal",
            feather_edge=0
        )
        
        print(f"結果のテンソル形状: {result[0].shape}")
        
        # 結果を画像として保存
        result_tensor = result[0]
        if len(result_tensor.shape) == 4:
            result_tensor = result_tensor.squeeze(0)
        
        result_numpy = (result_tensor.cpu().numpy() * 255).astype(np.uint8)
        result_image = Image.fromarray(result_numpy, 'RGB')
        result_image.save('test_result_normal.png')
        print("結果を test_result_normal.png に保存しました")
        
        # 異なるブレンドモードでテスト
        print("\nブレンドモード 'overlay' をテスト中...")
        result_overlay = composer.compose_images(
            background_image=background_tensor,
            overlay_image=overlay_tensor,
            x_position=150,
            y_position=100,
            scale=2.0,
            opacity=0.9,
            blend_mode="overlay",
            feather_edge=5
        )
        
        result_tensor_overlay = result_overlay[0]
        if len(result_tensor_overlay.shape) == 4:
            result_tensor_overlay = result_tensor_overlay.squeeze(0)
        
        result_numpy_overlay = (result_tensor_overlay.cpu().numpy() * 255).astype(np.uint8)
        result_image_overlay = Image.fromarray(result_numpy_overlay, 'RGB')
        result_image_overlay.save('test_result_overlay.png')
        print("結果を test_result_overlay.png に保存しました")
        
        print("\n✅ 全てのテストが正常に完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== ComfyUI 透過画像合成ノード テスト ===")
    test_transparent_image_composer()
