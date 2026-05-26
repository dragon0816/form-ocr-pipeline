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

CHECKBOX_WHITELIST = {
    "NR-NTN", "OT-NTN", "WiFi", "LTE/5G NR",
    "IOT-NTN", "JNR-NTN", "CiAN",
    "Signal Integrity", "Spectrum Analyzer",
    "Vector Network Analyzer", "Oscilloscope",
    "Power Supply", "Signal Generator",
    "六個月內有需求", "一年內有需求", "無需求", "有需求",
    "其他：",
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
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    section_ys: list[int] = []
    for b in blocks:
        for lbl in CHECKBOX_SECTION_LABELS:
            if lbl in b.text and b.bbox:
                section_ys.append(b.bbox.y)

    if not section_ys:
        section_ranges = [(0, gray.shape[0])]
    else:
        ss = sorted(set(section_ys))
        section_ranges = [
            (sy, min(sy + 250, ss[i+1] if i+1 < len(ss) else gray.shape[0]))
            for i, sy in enumerate(ss)
        ]

    NON_ITEM = set(CHECKBOX_SECTION_LABELS) | {
        "COMPANY RESTRICTED", "ROHDE", "Make ideas real",
        "基本資料", "填寫本文件", "若您有任何意見",
    }

    results: list[CheckboxResult] = []
    seen: set[str] = set()

    for b in blocks:
        if not b.bbox:
            continue
        if not any(sy <= b.bbox.y <= ey for sy, ey in section_ranges):
            continue
        if any(kw in b.text for kw in NON_ITEM):
            continue
        if len(b.text.strip()) < 2 or not match_whitelist(b.text):
            continue

        label = b.text.strip()
        if label in seen:
            continue
        seen.add(label)

        bx, by_, bw, bh = b.bbox.x, b.bbox.y, b.bbox.w, b.bbox.h
        lx1, lx2 = max(0, bx - 35), max(0, bx - 5)
        ly1, ly2 = max(0, by_ + 2), min(gray.shape[0], by_ + bh - 2)
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        region = gray[ly1:ly2, lx1:lx2]
        if region.size == 0:
            continue
        fill = float(np.sum(region < 100)) / region.size
        results.append(CheckboxResult(
            label=label, checked=fill > CHECKBOX_FILL_THRESH,
            bbox=BBox(lx1, ly1, lx2 - lx1, ly2 - ly1),
        ))

    results.sort(key=lambda r: r.bbox.y if r.bbox else 0)
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
