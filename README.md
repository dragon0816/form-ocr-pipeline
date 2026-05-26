# Form OCR Pipeline

混合 OCR 流水線，專為固定格式的問卷/訪客表設計。

## 架構

```
Native OS OCR（快速）→ 信心分流 → VLM 補強（低信心欄位）→ OpenCV 勾選框偵測
```

| 平台 | OCR 引擎 |
|------|---------|
| macOS  | Apple Vision Framework |
| Windows | Windows.Media.Ocr (WinRT) |
| Linux  | Tesseract (fallback) |

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# macOS
pip install -r requirements-macos.txt

# Windows
pip install -r requirements-windows.txt

# Linux（Tesseract fallback）
pip install -r requirements.txt
sudo apt install tesseract-ocr   # Debian/Ubuntu
```

VLM 模型（本機推理）：
```bash
ollama pull qwen2.5vl:7b
```

## 使用流程

### 步驟 1：建立模板（每種表單只需做一次）

```bash
python form_pipeline.py --new-template page_01.png
```

掃描一張表單，自動偵測所有欄位與勾選框位置，輸入名稱後儲存。

### 步驟 2：設定模板（依需求選擇要擷取的資訊）

```bash
python form_pipeline.py --config-template RS訪客表
```

互動式開/關欄位與勾選框，例如只保留「姓名」和「Email」。

```
╔══ 欄位設定 ══════════════════════════╗
║ [1] ✅  name         (姓名)          ║
║ [2] ☐   company      (公司)  ← 略過  ║
║ [3] ☐   department   (部門)  ← 略過  ║
║ [4] ☐   title        (職稱)  ← 略過  ║
║ [5] ✅  email        (Email)         ║
╚══════════════════════════════════════╝
▶ 2 3 4   ← 輸入編號切換開/關
```

### 步驟 3：批次處理

```bash
# 自動顯示模板選單
python form_pipeline.py page_02.png page_03.png

# 直接指定模板
python form_pipeline.py --use-template RS訪客表 page_02.png page_03.png
```

### 其他指令

```bash
python form_pipeline.py --list-templates    # 列出所有模板
python form_pipeline.py --no-template ...   # 強制完整模式
python form_pipeline.py --help
```

## 速度比較

| 模式 | 耗時/張 | 說明 |
|------|---------|------|
| 完整模式（無模板）| ~12s | 全頁 OCR + 所有欄位 VLM |
| Template 完整版 | ~5s  | 跳過全頁 OCR，7 個欄位 VLM |
| Template 精簡版 | ~2s  | 跳過全頁 OCR，2 個欄位 VLM |

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `form_pipeline.py` | 主程式：OCR 流水線 + 模板管理 CLI |
| `template_manager.py` | 模板 CRUD + 互動式 UI |
| `native_ocr.py` | 跨平台 OCR 模組（Apple Vision / WinRT / Tesseract）|
| `test_winrt.py` | Windows WinRT OCR 相容性測試 |
| `templates/` | 儲存的模板（JSON）|

## 模板格式

```json
{
  "name": "RS訪客表",
  "image_size": {"w": 1192, "h": 1685},
  "fields": {
    "name":  {"enabled": true,  "rx": 0.104, "ry": 0.105, "rw": 0.403, "rh": 0.029},
    "email": {"enabled": true,  "rx": 0.544, "ry": 0.158, "rw": 0.419, "rh": 0.025},
    "company": {"enabled": false, ...}
  },
  "checkboxes": [
    {"label": "NR-NTN", "enabled": true, "crx": 0.042, "cry": 0.277, ...}
  ]
}
```

座標以圖片寬高比例（0.0–1.0）儲存，自動支援不同 DPI 的掃描輸出。
