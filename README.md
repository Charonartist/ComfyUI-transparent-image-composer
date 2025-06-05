# ComfyUI 透過画像合成ノード

ComfyUI用のカスタムノードで、背景画像に透過画像を合成するためのツールです。

## 機能

- 背景画像に透過画像（PNG、RGBA形式）を合成
- 位置調整（X, Y座標）
- サイズ調整（スケール）
- 透明度調整
- エッジのぼかし機能
- 複数のブレンドモード対応

## インストール方法

1. ComfyUIの `custom_nodes` フォルダにこのフォルダをコピーまたはクローン
2. ComfyUIを再起動

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Charonartist/ComfyUI-transparent-image-composer.git
```

## 使用方法

### 基本的な使い方

1. ComfyUIで新しいワークフローを作成
2. 「透過画像合成」ノードを追加（検索で "透過" や "TransparentImageComposer" で見つかります）
3. 入力を接続：
   - `background_image`: 背景となる画像
   - `overlay_image`: 合成したい透過画像
4. パラメータを調整

### パラメータ説明

#### 必須パラメータ

- **background_image**: 背景画像（IMAGE形式）
- **overlay_image**: 合成する透過画像（IMAGE形式）
- **x_position**: 横位置（-4096 ～ 4096、デフォルト: 0）
- **y_position**: 縦位置（-4096 ～ 4096、デフォルト: 0）
- **scale**: サイズ倍率（0.1 ～ 5.0、デフォルト: 1.0）
- **opacity**: 透明度（0.0 ～ 1.0、デフォルト: 1.0）

#### オプションパラメータ

- **blend_mode**: ブレンドモード
  - `normal`: 通常の合成
  - `multiply`: 乗算
  - `screen`: スクリーン
  - `overlay`: オーバーレイ
  - `soft_light`: ソフトライト
  - `hard_light`: ハードライト
- **feather_edge**: エッジのぼかし（0 ～ 50ピクセル、デフォルト: 0）

## 使用例

### 基本的な透過画像合成

```
Load Image (背景) → 透過画像合成 → Save Image
Load Image (透過画像) ↗
```

### 複数の画像を重ねる

```
Load Image (背景) → 透過画像合成 → 透過画像合成 → Save Image
Load Image (画像1) ↗            ↗
Load Image (画像2) ──────────────┘
```

## 対応画像形式

- **入力**: PNG、JPG、WEBP等（ComfyUIでサポートされている形式）
- **透過情報**: RGBAチャンネル、またはグレースケール（黒→不透明、白→透明）
- **出力**: RGB形式

## 技術仕様

- 座標系: 左上が(0,0)
- 負の座標指定可能（画像の一部が切れる）
- 高品質なリサンプリング（Lanczos法）
- メモリ効率的な処理

## トラブルシューティング

### よくある問題

1. **透過が効かない**
   - 画像がRGBA形式になっているか確認
   - JPGファイルは透過情報を持たないため、PNGを使用

2. **合成結果が暗い**
   - ブレンドモードを `normal` に変更
   - `opacity` 値を確認

3. **エッジがギザギザ**
   - `feather_edge` パラメータを1-5程度に設定

### パフォーマンス

- 大きな画像の場合、処理時間が長くなる場合があります
- バッチ処理では最初の画像のみが処理されます

## ライセンス

MIT License

## 作者

Claude (Anthropic AI Assistant)

## バージョン履歴

- v1.0.0: 初期リリース
  - 基本的な透過画像合成機能
  - 位置、スケール、透明度調整
  - ブレンドモード対応
  - エッジぼかし機能
