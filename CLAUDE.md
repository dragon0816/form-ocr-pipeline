# CLAUDE.md — Form OCR Pipeline 開發指引

本文件記錄此專案的架構決策、踩過的坑、以及繼續開發時需要知道的背景知識。
讓任何一台電腦上的 Claude Code 都能快速接手。

---

## 專案目標

處理固定格式的訪客問卷（A4 紙，掃描成 PNG）。從中提取：
- **文字欄位**：姓名、公司、部門、職稱、Email、手機、公司電話
- **勾選框**：NR-NTN / OT-NTN / WiFi / LTE5G 等應用項目

速度要求：批次處理，每張 < 5 秒（完整模式 ~12s，模板模式 ~2–5s）。

---

## 架構總覽

```
輸入圖片
   │
   ├─ [有模板] → process_fast()
   │    ├─ 空白偵測 (arr < 128, ≥ 0.3%)
   │    ├─ VLM 辨識 (qwen2.5vl:7b)        ← 只跑 enabled 欄位
   │    └─ 直查像素判斷勾選框              ← 跳過 OCR，直接看灰階
   │
   └─ [無模板] → process_form() 完整模式
        ├─ ① Native OCR (Apple Vision / WinRT / Tesseract)
        ├─ ② 標籤定位 → 自適應 ROI → 信心分流
        │    confidence ≥ 0.50 → 直接採用
        │    confidence <  0.50 → VLM 補強
        ├─ ③ OpenCV 勾選框偵測（文字左側像素密度）
        └─ [若 --new-template] → create_template()
```

---

## 關鍵常數與理由

```python
CONFIDENCE_THRESHOLD  = 0.50   # OCR 信心門檻；低於此值送 VLM
ROI_CONTENT_THRESHOLD = 0.003  # 0.3% 深色像素 → 認定為空白欄位
ROW_GAP               = 30     # px；同一列標籤的 Y 差距 < 30 就忽略（Fix-1）
CHECKBOX_FILL_THRESH  = 0.15   # 15% 填充率 → 勾選
VLM_MODEL             = "qwen2.5vl:7b"
```

**為什麼 ROW_GAP = 30？**

表單是雙欄排版，company/department、title/email 是同一列：
```
公司 [___________]   部門 [________]
職稱 [___________]   Email [_______]
```
Apple Vision 回傳的 Y 中心差只有 1px，導致自適應 ROI 高度縮到 1–19px。
加入 ROW_GAP：計算邊界時只看 |cy_diff| > 30px 的鄰近標籤，避免被同列標籤影響。

**為什麼是 `arr < 128` 而不是 `< 100`？**

藍色墨水和灰色手寫在灰階下約落在 100–127 範圍。用 < 100 時幾乎所有欄位都判定為空白（比例 < 0.02%），導致 VLM 全部跳過。改 < 128 後正確偵測到手寫內容。

**為什麼 ROI_CONTENT_THRESHOLD = 0.003？**

初始值 0.025（2.5%）太嚴格，即使有手寫也常低於此門檻。降至 0.003（0.3%）後只要有幾個筆畫像素就能通過。

---

## 表單欄位結構

此專案針對 RS訪客表（固定格式）設計。欄位定義在 `FIELD_DEFS`：

```python
FIELD_DEFS = [
    ("name",        ["姓名"],                          480),  # (key, 標籤變體, 向右搜尋範圍px)
    ("company",     ["公司"],                          480),
    ("department",  ["部門"],                          380),
    ("title",       ["職稱"],                          480),
    ("email",       ["Email", "Emal", "email"],        500),  # OCR 常打錯
    ("phone",       ["聯繫手機", "聯繄手機", "聯絡手機"], 380), # 三種 OCR 變體
    ("school_phone",["公司/學校電話", "公司學校電話"],   350),
]
```

勾選框白名單（在 `CHECKBOX_WHITELIST`）：精確匹配或前綴匹配（`其他：` 後面常帶亂碼）。

---

## 模板系統設計

### 職責分離（重要！）

```
--new-template IMAGE     偵測 + 命名，全部欄位預設啟用，不做選擇
--config-template NAME   載入後互動選擇要啟用哪些欄位
--use-template NAME      直接用模板設定處理，不再詢問
```

**設計原因**：使用者需要先看到表單包含哪些欄位，才能決定要擷取哪些。
不能在建立時就選，因為那時還不知道表單的結構（這是第一次看到這張表單）。

### 模板 JSON 格式

```json
{
  "version": 2,
  "name": "RS訪客表",
  "description": "",
  "calibrated_from": "page_01.png",
  "created_at": "2025-01-15T10:30:00",
  "image_size": {"w": 1192, "h": 1685},
  "fields": {
    "name": {
      "enabled": true,
      "label_cn": "姓名",
      "rx": 0.104, "ry": 0.105, "rw": 0.403, "rh": 0.029
    }
  },
  "checkboxes": [
    {"label": "NR-NTN", "enabled": true, "crx": 0.042, "cry": 0.277, "crw": 0.05, "crh": 0.02}
  ]
}
```

座標以比例儲存（0.0–1.0），_scale_bbox() 在讀取時根據實際圖片尺寸縮放，支援不同 DPI 的掃描輸出。

---

## 跨平台 OCR

| 平台 | 引擎 | 特點 |
|------|------|------|
| macOS | Apple Vision Framework | 回傳 normalized bbox（底部左角原點），需轉換為頂部左角 |
| Windows | WinRT Windows.Media.Ocr | 需 Windows 10 1809+ |
| Linux | Tesseract | fallback，需另裝 `tesseract-ocr` 系統套件 |

**Apple Vision 座標轉換**：Vision 用底左為原點，y 軸朝上；
`native_ocr.py` 在 `AppleVisionOCR.recognize()` 已自動轉換為圖片座標（頂左原點）。

---

## 開發環境設定

```bash
# ⚠️ 必須先啟用虛擬環境，系統 Python 3.14 沒有 PIL/cv2
source .venv/bin/activate

# 安裝
pip install -r requirements-macos.txt   # macOS
pip install -r requirements-windows.txt # Windows

# VLM 模型（需先裝 Ollama）
ollama pull qwen2.5vl:7b
```

**常見錯誤**：直接用 `python` 而不啟動 venv → `ModuleNotFoundError: PIL`。

---

## 效能特性

| 模式 | 耗時/張 | 瓶頸 |
|------|--------|------|
| 完整模式（無模板）| ~12s | Native OCR 1s + 6個 VLM 各 ~2s |
| Template 完整版 | ~5s | 7個 VLM 各 ~0.7s（KV cache 熱啟動）|
| Template 精簡版（2欄）| ~2s | 2個 VLM |

**VLM KV Cache**：同一 session 內第一次呼叫 ~2s（cold），之後 ~0.7s（cache warm）。
批次處理多張圖時，第二張起明顯加速。

---

## 檔案職責

```
form_pipeline.py     主程式（OCR 流水線 + CLI）
template_manager.py  模板 CRUD + 互動式 UI（完全獨立，可單獨 import）
native_ocr.py        跨平台 OCR 封裝（BBox, TextBlock dataclass）
test_winrt.py        Windows WinRT 6步驟相容性測試
templates/           使用者建立的模板（.gitignore 排除，不上傳）
```

---

## 已知問題 / 待改進

1. **勾選框漂移**：掃描歪斜或不同打印機邊距時，checkbox 座標可能偏移 5–15px。
   建議：定期以 `--no-template` 模式驗證，或在 `_scale_bbox` 加入旋轉校正。

2. **電話號碼 VLM 幻覺**：空白手機欄位有時 VLM 會捏造數字。Fix-2（空白偵測）
   已大幅改善，但極淡的底色印刷仍可能觸發 VLM。

3. **多頁 PDF**：目前只處理單張 PNG，批次需先用外部工具轉換：
   ```bash
   python -c "from pdf2image import convert_from_path; ..."
   ```

4. **`--config-template` 名稱含空格**：Shell 需要加引號：
   ```bash
   python form_pipeline.py --config-template "RS 訪客表"
   ```

---

## 常用開發指令

```bash
# 建立第一個模板
python form_pipeline.py --new-template page_01.png

# 設定要擷取的欄位（互動式）
python form_pipeline.py --config-template RS訪客表

# 批次處理（自動顯示模板選單）
python form_pipeline.py page_02.png page_03.png

# 強制完整模式（除錯用）
python form_pipeline.py --no-template page_01.png

# 列出所有模板
python form_pipeline.py --list-templates

# 只跑 OCR 模組（debug）
python native_ocr.py  # 直接執行會印出引擎資訊

# 列出所有模板（直接用 template_manager）
python template_manager.py
```

---

## Git / GitHub

倉庫：https://github.com/dragon0816/form-ocr-pipeline

**不上傳的內容（.gitignore）**：
- `templates/*.json`：含個人資料（姓名/Email），使用者自行管理
- `.claude/`：Claude Code 本地設定
- `*.png / *.jpg / *.pdf`：掃描圖片

每台新電腦 clone 後需自行建立模板：
```bash
git clone https://github.com/dragon0816/form-ocr-pipeline
cd form-ocr-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-macos.txt
ollama pull qwen2.5vl:7b
python form_pipeline.py --new-template your_form.png
```
