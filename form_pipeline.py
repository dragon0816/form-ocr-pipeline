#!/usr/bin/env python3
"""
form_pipeline.py — 混合 OCR 流水線  v5

流程：
  ① native_ocr  全頁文字 + bbox + confidence
  ② 標籤定位    以高信心印刷標籤為錨點，找值欄位區域
  ③ 信心分流    confidence ≥ 0.5 → 直接採用；< 0.5 → VLM 補強
  ④ 勾選框偵測  計算 □ 左側深色像素比例

v5 新增：模板管理器整合
  --new-template IMAGE   校準一張圖，互動式選擇欄位/勾選框，存模板
  --list-templates       列出所有模板
  --use-template NAME    指定模板（略過選擇選單）
  --no-template          強制完整模式（忽略模板）
  （無旗標）             若有模板 → 顯示選擇選單；若無 → 完整模式

v3/v4 修正（保留）：
  Fix-1  ROI 自適應高度（ROW_GAP 過濾同列標籤）
  Fix-2  空白偵測（深色像素 < 0.3% → 不送 VLM）
  Fix-3  勾選框白名單過濾
"""

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import ollama
from PIL import Image

from native_ocr import get_ocr_engine, TextBlock, BBox
import template_manager as tm

# ── 設定 ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.50
VLM_MODEL             = "qwen2.5vl:7b"
CHECKBOX_FILL_THRESH  = 0.15
PADDING               = 15
ROI_CONTENT_THRESHOLD = 0.003
ROW_GAP               = 30

FIELD_CN = {
    "name":         "姓名",
    "company":      "公司",
    "department":   "部門",
    "title":        "職稱",
    "email":        "Email",
    "phone":        "聯絡手機",
    "school_phone": "公司/學校電話",
}

# ── 表單欄位定義 ─────────────────────────────────────────────
FIELD_DEFS = [
    ("name",        ["姓名"],                           480),
    ("company",     ["公司"],                           480),
    ("department",  ["部門"],                           380),
    ("title",       ["職稱"],                           480),
    ("email",       ["Email", "Emal", "email"],         500),
    ("phone",       ["聯繫手機", "聯繄手機", "聯絡手機"],  380),
    ("school_phone",["公司/學校電話", "公司學校電話"],     350),
]

CHECKBOX_SECTION_LABELS = [
    "請問您會需要哪些應用",
    "計畫採購以下品項",
    "請問貴單位近期是否有採購",
]

# WinRT OCR 將表格 □ 辨識為下列字元之一
CHECKBOX_CHARS = frozenset("口囗□")

# 備用白名單：OCR 沒有「口」前綴但仍是 checkbox 的項目（用正確標籤）
CHECKBOX_WHITELIST = {
    # Section 1
    "NR-NTN", "IOT-NTN", "WiFi", "LTE/5G NR",
    "Power Electronics", "Signal Integrity",
    # Section 3
    "Call box", "Source Measurement Unit", "Oscilloscope",
    "Spectrum Analyzer", "Vector Network Analyzer",
    "LCR Meter", "Power Supply",
    # Section 4（含 OCR 常見殘字「有需求」→ normalize 成「六個月內有需求」）
    "三個月內有需求", "六個月內有需求", "一年內有需求", "無需求", "有需求",
}

# WinRT OCR 拼字錯誤修正表（偵測後套用，讓 label 回到正確名稱）
LABEL_NORMALIZE: dict[str, str] = {
    # Section 1
    "WlFi":                         "WiFi",
    "power EIectronics":            "Power Electronics",
    "Signallntegrity":              "Signal Integrity",
    "其他•":                        "其他",
    "其他 •":                       "其他",
    # Section 2
    "需要相關人員與我聯繫;產品,":      "需要相關人員與我聯繫;產品:",
    "安排DEMO ,產品:":              "安排DEMO;產品:",
    "提供報價;產品":                 "提供報價;產品:",
    # Section 3
    "CaII bOㄨ":                    "Call box",
    "source Measurement Unit":      "Source Measurement Unit",
    "OsciIIoscope":                 "Oscilloscope",
    "spectrum Ana lyzer":           "Spectrum Analyzer",
    "vector NetworkAnaIyzer":       "Vector Network Analyzer",
    "Meter":                        "LCR Meter",
    "PowerSupply":                  "Power Supply",
    # Section 4
    "二個月內有需求":               "三個月內有需求",
    "有需求":                       "六個月內有需求",
}


def match_whitelist(text: str) -> bool:
    t = text.strip()
    return any(t == w or t.startswith(w) for w in CHECKBOX_WHITELIST)


# ── 資料結構 ─────────────────────────────────────────────────
@dataclass
class FieldResult:
    key: str
    value: str
    confidence: float
    method: str           # "ocr" | "vlm" | "none"
    elapsed: float = 0.0
    roi: Optional[BBox]  = None   # 供模板校準用

@dataclass
class CheckboxResult:
    label: str
    checked: bool
    bbox: Optional[BBox] = None   # 量測區域（文字左側）

@dataclass
class FormResult:
    image_path: str
    fields: list[FieldResult]        = field(default_factory=list)
    checkboxes: list[CheckboxResult] = field(default_factory=list)
    total_elapsed: float             = 0.0

    def to_dict(self) -> dict:
        return {
            "image": self.image_path,
            "total_elapsed": round(self.total_elapsed, 3),
            "fields": {
                r.key: {
                    "value":      r.value,
                    "confidence": round(r.confidence, 2),
                    "method":     r.method,
                    "elapsed":    round(r.elapsed, 3),
                }
                for r in self.fields
            },
            "checkboxes": {r.label: r.checked for r in self.checkboxes},
        }


# ── Native OCR ───────────────────────────────────────────────
def run_native_ocr(image_path: str) -> list[TextBlock]:
    return get_ocr_engine().recognize(image_path)


# ── 標籤搜尋 ─────────────────────────────────────────────────
def find_label(blocks: list[TextBlock], variants: list[str]) -> Optional[TextBlock]:
    for b in blocks:
        if b.confidence >= 0.4:
            for v in variants:
                if v in b.text:
                    return b
    for b in blocks:
        for v in variants:
            if v in b.text:
                return b
    return None


def find_value_blocks(label: TextBlock, blocks: list[TextBlock],
                      max_right: int) -> list[TextBlock]:
    if not label.bbox:
        return []
    lx, ly, lw, lh = label.bbox.x, label.bbox.y, label.bbox.w, label.bbox.h
    margin = lh * 1.8
    PRINTED = {
        "姓名","公司","部門","職稱","Email","Emal",
        "聯繫手機","聯繄手機","聯絡手機","公司/學校電話",
        "Make ideas real","ROHDE","COMPANY RESTRICTED",
    }
    out = []
    for b in blocks:
        if b is label or not b.bbox:
            continue
        if b.confidence >= 0.85 and any(kw in b.text for kw in PRINTED):
            continue
        bx, by = b.bbox.x, b.bbox.y
        if (lx + lw + 5 <= bx <= lx + lw + max_right
                and ly - margin <= by <= ly + lh + margin):
            out.append(b)
    out.sort(key=lambda b: b.bbox.x)
    return out


def best_value(candidates: list[TextBlock]) -> tuple[str, float]:
    if not candidates:
        return "", 0.0
    return " ".join(b.text for b in candidates).strip(), \
           min(b.confidence for b in candidates)


# ── 空白偵測 ─────────────────────────────────────────────────
def roi_has_content(image: Image.Image, roi: BBox) -> bool:
    x  = max(0, roi.x)
    y  = max(0, roi.y)
    x2 = min(image.width,  roi.x + roi.w)
    y2 = min(image.height, roi.y + roi.h)
    if x2 <= x or y2 <= y:
        return False
    arr = np.array(image.crop((x, y, x2, y2)).convert("L"))
    return float(np.sum(arr < 128)) / arr.size >= ROI_CONTENT_THRESHOLD


# ── VLM 補強 ─────────────────────────────────────────────────
def vlm_read_crop(image: Image.Image, roi: BBox, field_key: str) -> tuple[str, float]:
    x  = max(0, roi.x - 80)
    y  = max(0, roi.y - 8)
    x2 = min(image.width,  roi.x + roi.w + PADDING)
    y2 = min(image.height, roi.y + roi.h + 8)
    crop = image.crop((x, y, x2, y2))

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    field_cn = FIELD_CN.get(field_key, field_key)
    prompt = (
        f"This is a cropped row from a form. "
        f"Please read ONLY the handwritten value for the field labeled '{field_cn}'. "
        "Ignore all printed text (labels, headers). "
        "If there is no handwritten value visible, reply with an empty string. "
        "Return only the raw value, nothing else."
    )
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip(), time.perf_counter() - t0


# ── 欄位提取（完整模式）──────────────────────────────────────
def extract_fields(blocks: list[TextBlock], image: Image.Image) -> list[FieldResult]:
    results: list[FieldResult] = []

    # Fix-1：蒐集各標籤 Y 中心
    label_ys: list[int] = []
    for _, variants, _ in FIELD_DEFS:
        lb = find_label(blocks, variants)
        if lb and lb.bbox:
            label_ys.append(lb.bbox.y + lb.bbox.h // 2)
    label_ys.sort()

    for key, variants, max_right in FIELD_DEFS:
        label = find_label(blocks, variants)
        if label is None:
            results.append(FieldResult(key=key, value="", confidence=0.0, method="none"))
            continue

        # Fix-1：自適應 ROI
        if label.bbox:
            lx, ly, lw, lh = label.bbox.x, label.bbox.y, label.bbox.w, label.bbox.h
            cy = ly + lh // 2
            prev = [y for y in label_ys if cy - y > ROW_GAP]
            nxt  = [y for y in label_ys if y - cy > ROW_GAP]
            top_y = (prev[-1] + cy) // 2 if prev else max(0, ly - 20)
            bot_y = (nxt[0]  + cy) // 2 if nxt  else ly + lh + 20
            roi = BBox(x=lx + lw + 5, y=top_y, w=max_right, h=max(lh, bot_y - top_y))
        else:
            roi = BBox(0, 0, image.width, 80)

        # 嘗試 OCR 結果
        candidates = find_value_blocks(label, blocks, max_right)
        text, conf = best_value(candidates)
        if conf >= CONFIDENCE_THRESHOLD and text:
            results.append(FieldResult(key=key, value=text, confidence=conf,
                                       method="ocr", roi=roi))
            continue

        # Fix-2：空白偵測
        if not roi_has_content(image, roi):
            results.append(FieldResult(key=key, value="", confidence=0.9,
                                       method="vlm", roi=roi))
            continue

        # VLM 補強
        print(f"    [{key}] OCR 信心 {conf:.0%} → VLM 補強…")
        val, t = vlm_read_crop(image, roi, key)
        results.append(FieldResult(key=key, value=val, confidence=0.9,
                                   method="vlm", elapsed=t, roi=roi))
    return results


# ── 勾選框偵測（完整模式）───────────────────────────────────
def detect_checkboxes(image_path: str, blocks: list[TextBlock]) -> list[CheckboxResult]:
    """
    WinRT OCR 感知的勾選框偵測。

    問題根源：WinRT OCR 將表格的 □ 辨識為「口」/「囗」字元，
    且常將同一列的多個 checkbox 項目合併成一個 OCR block，
    例如：'囗source Measurement Unit口OsciIIoscope'。

    策略：
    1. 修正 section_ranges（移除硬限的 250px 上限）
    2. 對含有「口」/「囗」的 block 以這些字元為分隔點切割，
       用字元寬度比例估算每個 checkbox 的像素座標
    3. 量測該座標的填充率，判斷是否打勾
    4. 對沒有「口」前綴但符合白名單的 block，沿用向左掃描像素的備用邏輯
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []

    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Section Y 範圍（移除 250px 硬限）─────────────────
    section_ys: list[int] = []
    for b in blocks:
        for lbl in CHECKBOX_SECTION_LABELS:
            if lbl in b.text and b.bbox:
                section_ys.append(b.bbox.y)

    if not section_ys:
        section_ranges = [(0, h_img)]
    else:
        ss = sorted(set(section_ys))
        section_ranges = [
            (max(0, sy - 10),
             ss[i + 1] if i + 1 < len(ss) else h_img)
            for i, sy in enumerate(ss)
        ]

    NON_LABEL_KW = set(CHECKBOX_SECTION_LABELS) | {
        "COMPANY RESTRICTED", "ROHDE", "Make ideas real",
        "基本資料", "填寫本文件", "若您有任何意見",
    }

    # ── 2. 解析 OCR block 中的「口」/「囗」標記 ──────────────
    # raw_items: (cb_x, cb_y, cb_size, label, method)
    # method = "marker" → 口字元 IS checkbox；"look_left" → 向左掃描備用
    raw_items: list[tuple[int, int, int, str, str]] = []

    for b in blocks:
        if not b.bbox:
            continue
        if not any(sy <= b.bbox.y <= ey for sy, ey in section_ranges):
            continue
        if any(kw in b.text for kw in NON_LABEL_KW):
            continue

        text = b.text
        bx, by, bw, bh = b.bbox.x, b.bbox.y, b.bbox.w, b.bbox.h
        cb_size = bh          # checkbox 高度 ≈ OCR block 高度

        if any(c in text for c in CHECKBOX_CHARS):
            # 此 block 含「口」/「囗」→ 以字元寬度比例估算各 checkbox 的 X 座標
            char_w = bw / max(len(text), 1)

            # ── 情形 A：「口」出現在第一個字元之後（前綴文字）
            # 例：'二個月內有需求口亠' 或 'Meter口PowerSupply'
            # 前綴文字自身就是一個項目，checkbox 在其左方
            first_cb = next((idx for idx, c in enumerate(text) if c in CHECKBOX_CHARS), -1)
            if first_cb > 2:
                prefix = text[:first_cb].strip()
                if len(prefix) >= 2:
                    raw_items.append((bx, by, bh, prefix, "look_left"))

            # ── 情形 B：逐一掃描「口」標記，取其後方的標籤
            i = 0
            while i < len(text):
                if text[i] in CHECKBOX_CHARS:
                    cb_x = int(bx + i * char_w)
                    j = i + 1
                    while j < len(text) and text[j] not in CHECKBOX_CHARS:
                        j += 1
                    label = text[i + 1:j].strip()
                    if len(label) >= 2 or not label:
                        lbl = label if label else f"cb_{cb_x}_{by}"
                        raw_items.append((cb_x, by, cb_size, lbl, "marker"))
                    i = j
                else:
                    i += 1

        elif match_whitelist(text) and len(text.strip()) >= 2:
            # 沒有「口」前綴，但符合白名單 → checkbox 在文字左側
            raw_items.append((bx, by, bh, text.strip(), "look_left"))

        else:
            # ── 情形 C：OCR 將 □ 誤識為非 ASCII 字元（如 'Ü'、'Û'、'Ö' 等）
            # 條件：第一個字元非 ASCII（isascii()==False）且是字母，非中文，非已知 checkbox char
            stripped = text.strip()
            c0 = stripped[0] if stripped else ""
            is_odd_leading = (
                bool(c0)
                and not c0.isascii()            # 非 ASCII（Ü、Û 等）
                and c0.isalpha()                # 是字母而非標點
                and not (0x4E00 <= ord(c0) <= 0x9FFF)  # 非中文
                and c0 not in CHECKBOX_CHARS
            )
            if is_odd_leading and len(stripped) >= 3:
                rest = stripped[1:].strip()
                if len(rest) >= 2:
                    raw_items.append((bx, by, bh, rest, "marker"))

    # ── 3. 去重 ────────────────────────────────────────────────
    # 優先保留 "marker" 方式；兩種重複條件：
    #   a) 中心距 < 14px（位置幾乎相同）
    #   b) 相同標籤 且 Y 距 < 25px（同一列的不同解析路徑）
    raw_items.sort(key=lambda x: 0 if x[4] == "marker" else 1)   # marker 優先
    kept: list[tuple[int, int, int, str, str]] = []
    for item in raw_items:
        cx, cy, _, label, _ = item
        dup = False
        for kx, ky, _, klabel, _ in kept:
            if abs(cx - kx) < 14 and abs(cy - ky) < 14:
                dup = True; break
            if label == klabel and abs(cy - ky) < 25 and abs(cx - kx) < 120:
                dup = True; break
        if not dup:
            kept.append(item)

    # ── 3b. 對座標型標籤嘗試從附近 OCR block 補回可讀名稱 ────
    NON_LABEL_KW2 = NON_LABEL_KW | CHECKBOX_CHARS
    patched: list[tuple[int, int, int, str, str]] = []
    for item in kept:
        cb_x, cb_y, cb_size, label, method = item
        if label.startswith("cb_"):
            # 找右側 ≤ 100px、Y 距 ≤ 20px 的最近 OCR block
            best_lbl, best_dist = label, float('inf')
            for b2 in blocks:
                if not b2.bbox:
                    continue
                if any(kw in b2.text for kw in NON_LABEL_KW):
                    continue
                h_gap = b2.bbox.x - (cb_x + cb_size)
                v_diff = abs((b2.bbox.y + b2.bbox.h / 2) - (cb_y + cb_size / 2))
                if -cb_size * 0.3 <= h_gap <= 100 and v_diff <= 20:
                    dist = max(0, h_gap) + v_diff
                    # 取「口」之後的第一段純文字
                    clean = b2.text.strip()
                    for c in CHECKBOX_CHARS:
                        clean = clean.replace(c, " ")
                    clean = clean.split()[0] if clean.split() else clean
                    if len(clean) >= 2 and dist < best_dist:
                        best_dist = dist
                        best_lbl = clean
            patched.append((cb_x, cb_y, cb_size, best_lbl, method))
        else:
            patched.append(item)
    # 修補後再做一次去重（修補可能讓不同座標的 cb 取得相同標籤）
    deduped2: list[tuple[int, int, int, str, str]] = []
    for item in patched:
        cx, cy, _, label, _ = item
        dup = False
        for kx, ky, _, klabel, _ in deduped2:
            if abs(cx - kx) < 14 and abs(cy - ky) < 14:
                dup = True; break
            if label == klabel and abs(cy - ky) < 25 and abs(cx - kx) < 120:
                dup = True; break
        if not dup:
            deduped2.append(item)
    kept = deduped2

    # ── 4. 量測填充率，產生結果 ──────────────────────────────
    results: list[CheckboxResult] = []
    for (cb_x, cb_y, cb_size, label, method) in kept:
        if method == "marker":
            # 「口」字元本身即 checkbox：量測其內部像素（排除邊框）
            margin = max(2, int(cb_size * 0.18))
            lx1 = max(0,     cb_x + margin)
            lx2 = min(w_img, cb_x + cb_size - margin)
            ly1 = max(0,     cb_y + margin)
            ly2 = min(h_img, cb_y + cb_size - margin)
        else:
            # look_left：checkbox 在文字左方 5–40px
            lx1 = max(0,     cb_x - 40)
            lx2 = max(0,     cb_x - 5)
            ly1 = max(0,     cb_y + 2)
            ly2 = min(h_img, cb_y + cb_size - 2)

        if lx2 <= lx1 or ly2 <= ly1:
            continue
        region = gray[ly1:ly2, lx1:lx2]
        if region.size == 0:
            continue

        fill = float(np.sum(region < 110)) / region.size
        results.append(CheckboxResult(
            label=label,
            checked=fill > CHECKBOX_FILL_THRESH,
            bbox=BBox(lx1, ly1, lx2 - lx1, ly2 - ly1),
        ))

    results.sort(key=lambda r: (r.bbox.y, r.bbox.x) if r.bbox else (0, 0))

    # ── 5. 套用 OCR 拼字修正 ─────────────────────────────────
    for r in results:
        r.label = LABEL_NORMALIZE.get(r.label, r.label)

    return results


# ── Template 模式：快速處理 ──────────────────────────────────
def _scale_bbox(d: dict, tw: int, th: int, sx: float, sy: float,
                prefix: str = "r") -> BBox:
    return BBox(
        x=int(d[f"{prefix}x"] * tw * sx),
        y=int(d[f"{prefix}y"] * th * sy),
        w=max(1, int(d[f"{prefix}w"] * tw * sx)),
        h=max(1, int(d[f"{prefix}h"] * th * sy)),
    )


def process_fast(image_path: str, img: Image.Image,
                 template: dict) -> FormResult:
    """
    Template 快速模式：跳過全頁 OCR，直接用模板座標處理。
    - 只處理 enabled=True 的欄位與勾選框
    - 欄位：空白偵測 → VLM
    - 勾選框：直接讀像素（不跑 OCR）
    """
    result = FormResult(image_path=image_path)
    t_start = time.perf_counter()
    iw, ih = img.size
    tw = template["image_size"]["w"]
    th = template["image_size"]["h"]
    sx, sy = iw / tw, ih / th

    if abs(sx - 1.0) > 0.05 or abs(sy - 1.0) > 0.05:
        print(f"  ⚠️  尺寸差異 {iw}×{ih} vs template {tw}×{th}，"
              f"自動縮放 ({sx:.3f}, {sy:.3f})")

    name = template.get("name", "?")
    print(f"\n  ⚡ Template：{name}（校準自：{template.get('calibrated_from','?')}）")

    # ── 欄位 ─────────────────────────────────────────────────
    print(f"\n  ① 欄位提取（Template ROI，跳過全頁 OCR）…")
    t0 = time.perf_counter()
    fields: list[FieldResult] = []

    for key, roi_data in template["fields"].items():
        # 略過停用欄位
        if not roi_data.get("enabled", True):
            continue

        roi = _scale_bbox(roi_data, tw, th, sx, sy, prefix="r")
        if not roi_has_content(img, roi):
            fields.append(FieldResult(key=key, value="", confidence=0.9,
                                      method="vlm", roi=roi))
            continue

        print(f"    [{key}] 裁切 ROI → VLM 辨識…")
        val, elapsed = vlm_read_crop(img, roi, key)
        fields.append(FieldResult(key=key, value=val, confidence=0.9,
                                  method="vlm", elapsed=elapsed, roi=roi))

    result.fields = fields
    print(f"     → 完成 ({time.perf_counter()-t0:.2f}s)")

    # ── 勾選框 ───────────────────────────────────────────────
    print(f"  ② 勾選框（Template 直查像素）…")
    t0 = time.perf_counter()
    img_bgr = cv2.imread(image_path)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    checkboxes: list[CheckboxResult] = []

    for cb in template["checkboxes"]:
        if not cb.get("enabled", True):
            continue
        b   = _scale_bbox(cb, tw, th, sx, sy, prefix="cr")
        x1  = max(0, b.x)
        y1  = max(0, b.y)
        x2  = min(gray.shape[1], b.x + b.w)
        y2  = min(gray.shape[0], b.y + b.h)
        if x2 <= x1 or y2 <= y1:
            continue
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            continue
        fill = float(np.sum(region < 100)) / region.size
        checkboxes.append(CheckboxResult(
            label=cb["label"], checked=fill > CHECKBOX_FILL_THRESH,
            bbox=BBox(x1, y1, x2 - x1, y2 - y1),
        ))

    result.checkboxes = checkboxes
    print(f"     → {len(checkboxes)} 個勾選框 ({time.perf_counter()-t0:.4f}s)")

    result.total_elapsed = time.perf_counter() - t_start
    return result


# ── 主流程 ────────────────────────────────────────────────────
def process_form(
    image_path:   str,
    template:     Optional[dict] = None,
    new_template: bool = False,
) -> FormResult:
    """
    template=None, new_template=False  →  完整模式
    template=None, new_template=True   →  完整模式 + 互動建立模板
    template=dict                      →  Template 快速模式
    """
    img = Image.open(image_path)

    # ── Template 快速模式 ────────────────────────────────────
    if template is not None:
        return process_fast(image_path, img, template)

    # ── 完整模式 ─────────────────────────────────────────────
    result = FormResult(image_path=image_path)
    t_start = time.perf_counter()

    if new_template:
        print(f"\n  🔧 新增模板模式：跑完整流水線後進入互動設定")

    # ① Native OCR
    print(f"\n  ① Native OCR…")
    t0 = time.perf_counter()
    blocks = run_native_ocr(image_path)
    print(f"     → {len(blocks)} 個區塊  ({time.perf_counter()-t0:.2f}s)")

    # ② 欄位提取
    print(f"  ② 欄位提取（信心 < {CONFIDENCE_THRESHOLD:.0%} 自動呼叫 VLM）…")
    t0 = time.perf_counter()
    result.fields = extract_fields(blocks, img)
    print(f"     → 完成 ({time.perf_counter()-t0:.2f}s)")

    # ③ 勾選框
    print(f"  ③ OpenCV 勾選框偵測…")
    t0 = time.perf_counter()
    result.checkboxes = detect_checkboxes(image_path, blocks)
    print(f"     → {len(result.checkboxes)} 個勾選框 ({time.perf_counter()-t0:.2f}s)")

    result.total_elapsed = time.perf_counter() - t_start

    # ── 建立模板（偵測完成後命名存檔，不做欄位選擇）────────────
    if new_template:
        fields_info = [
            {"key": f.key, "label_cn": FIELD_CN.get(f.key, f.key), "roi": f.roi}
            for f in result.fields
        ]
        checkboxes_info = [
            {"label": c.label, "bbox": c.bbox}
            for c in result.checkboxes if c.bbox
        ]
        tm.create_template(
            calibration_image=image_path,
            image_size=img.size,
            fields_info=fields_info,
            checkboxes_info=checkboxes_info,
        )

    return result


# ── 結果列印 ──────────────────────────────────────────────────
def print_result(form: FormResult) -> None:
    print(f"\n  ── 結果 ──")
    for f in form.fields:
        icon = "🤖" if f.method == "vlm" else ("✅" if f.method == "ocr" else "❌")
        print(f"  {icon} {f.key:15s}: {f.value}  [{f.confidence:.0%} / {f.method}]")

    checked   = [c for c in form.checkboxes if c.checked]
    unchecked = [c for c in form.checkboxes if not c.checked]
    print(f"\n  ☑  已勾選 ({len(checked)}):")
    for c in checked:
        print(f"       {c.label}")
    print(f"  □  未勾選 ({len(unchecked)}):")
    for c in unchecked[:5]:
        print(f"       {c.label}")
    if len(unchecked) > 5:
        print(f"       …（共 {len(unchecked)} 個）")
    print(f"\n  ⏱  總耗時: {form.total_elapsed:.2f}s")


# ── 執行入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="混合 OCR 流水線 v5（模板管理器）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
工作流程：
  步驟 1  建立模板（偵測表單佈局，全部欄位預設啟用）
          python form_pipeline.py --new-template page_01.png

  步驟 2  設定模板（選擇要擷取哪些欄位 / 勾選框）
          python form_pipeline.py --config-template RS訪客表

  步驟 3  批次處理（自動顯示選單，或直接指定）
          python form_pipeline.py page_02.png page_03.png
          python form_pipeline.py --use-template RS訪客表 page_02.png

其他：
  python form_pipeline.py --list-templates    列出所有模板
  python form_pipeline.py --no-template ...   強制完整模式
""",
    )
    parser.add_argument("images", nargs="*",
                        default=["page_01.png", "page_02.png"])
    parser.add_argument("--new-template", metavar="IMAGE",
                        help="步驟1：校準指定圖片，偵測欄位並建立新模板")
    parser.add_argument("--config-template", metavar="NAME",
                        help="步驟2：設定模板的欄位啟用狀態（開/關）")
    parser.add_argument("--list-templates", action="store_true",
                        help="列出所有已儲存的模板")
    parser.add_argument("--use-template", metavar="NAME",
                        help="指定要使用的模板（略過選擇選單）")
    parser.add_argument("--no-template", action="store_true",
                        help="強制完整模式，忽略所有模板")
    args = parser.parse_args()

    # ── --list-templates ────────────────────────────────────
    if args.list_templates:
        templates = tm.list_templates()
        tm.print_template_list(templates)
        print(f"\n  共 {len(templates)} 個模板，存於 {tm.TEMPLATES_DIR}/")
        sys.exit(0)

    # ── --new-template ───────────────────────────────────────
    if args.new_template:
        cal_image = args.new_template
        if not Path(cal_image).exists():
            print(f"❌ 找不到校準圖片：{cal_image}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("  步驟 1：建立新模板")
        print(f"  校準圖片：{cal_image}")
        print("=" * 60)

        result = process_form(cal_image, new_template=True)
        print_result(result)
        sys.exit(0)

    # ── --config-template ────────────────────────────────────
    if args.config_template:
        print(f"\n{'='*60}")
        print(f"  步驟 2：設定模板 —— {args.config_template}")
        print("=" * 60)
        tm.config_template(args.config_template)
        sys.exit(0)

    # ── 決定模板 ─────────────────────────────────────────────
    template: Optional[dict] = None

    if not args.no_template:
        if args.use_template:
            try:
                template = tm.load_template(args.use_template)
                print(f"\n  ✅ 已載入模板：{template.get('name','?')}")
            except FileNotFoundError as e:
                print(f"  ❌ {e}")
                sys.exit(1)
        else:
            templates = tm.list_templates()
            if templates:
                template = tm.select_template_interactive(templates)

    # ── 批次處理 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  混合 OCR 流水線  v5")
    mode = f"Template：{template.get('name','?')}" if template else "完整模式"
    print(f"  模式：{mode}")
    print("=" * 60)

    all_results: dict = {}

    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"\n  ⚠️  找不到 {img_path}")
            continue
        print(f"\n📄 {img_path}")
        form = process_form(img_path, template=template)
        print_result(form)
        all_results[img_path] = form.to_dict()

    out = "pipeline_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 結果存至 {out}")
    print("\n✅ 完成！\n")
