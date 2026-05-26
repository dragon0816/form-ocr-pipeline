#!/usr/bin/env python3
"""
gemma4:e2b OCR 能力測試腳本
測試項目：
  1. 印刷文字辨識能力
  2. 手寫中文/英文辨識能力
  3. 選項勾選辨識能力
"""

import base64
import json
import time
from pathlib import Path
import ollama

MODEL = "gemma4:e2b"
IMAGE_FILES = ["page_01.png", "page_02.png"]


# ─────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    """將圖片轉成 base64 字串供 ollama 使用"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask(prompt: str, image_path: str, label: str) -> dict:
    """對模型送出單一問題，回傳結果與耗時"""
    img_b64 = load_image_b64(image_path)
    print(f"\n  ▶ {label}")
    print(f"    提示: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    t0 = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        options={"temperature": 0},
    )
    elapsed = time.time() - t0

    answer = response["message"]["content"].strip()
    print(f"    回答 ({elapsed:.1f}s):\n{_indent(answer, 6)}")
    return {"label": label, "answer": answer, "elapsed": elapsed}


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 測試 1：印刷文字辨識
# ─────────────────────────────────────────────────────────────

PRINTED_TEXT_PROMPTS = [
    (
        "Please extract ALL the printed (non-handwritten) text from this form. "
        "List each printed label/heading exactly as it appears.",
        "印刷文字 — 全文擷取",
    ),
    (
        "What is the company name shown in the logo/header of this form? "
        "What is the form title in Chinese?",
        "印刷文字 — 標題與公司名稱",
    ),
]

# ─────────────────────────────────────────────────────────────
# 測試 2：手寫文字辨識（中文 + 英文）
# ─────────────────────────────────────────────────────────────

HANDWRITTEN_PROMPTS = [
    (
        "This form contains handwritten text. Please read and transcribe "
        "ALL handwritten content you can see, including names, emails, dates, "
        "and any free-text answers. Preserve the original language (Chinese or English).",
        "手寫文字 — 全部辨識",
    ),
    (
        "Read the handwritten fields: 姓名 (name), 公司 (company), 聯絡手機 (phone), "
        "部門 (department), Email. What values are written in each field?",
        "手寫文字 — 表頭聯絡資訊",
    ),
    (
        "There is a handwritten section at the bottom of the form. "
        "Please transcribe all handwritten text in that section as accurately as possible.",
        "手寫文字 — 底部備註區",
    ),
]

# ─────────────────────────────────────────────────────────────
# 測試 3：選項勾選辨識
# ─────────────────────────────────────────────────────────────

CHECKBOX_PROMPTS = [
    (
        "This form has many checkboxes (□). Please list EVERY checkbox option "
        "and indicate whether each one is checked (✓/X/filled) or unchecked (empty □). "
        "Group them by section if possible.",
        "勾選框 — 全部狀態列表",
    ),
    (
        "Look at the checkboxes in the section about '請問您的需求是否適用？' or "
        "similar product/solution categories. "
        "Which boxes are checked and which are unchecked?",
        "勾選框 — 產品需求區",
    ),
    (
        "In the measurement equipment section (計量單元 or similar), "
        "list all checkboxes and their checked/unchecked status.",
        "勾選框 — 量測設備區",
    ),
]

# ─────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────

def run_tests_for_image(image_path: str):
    img_label = Path(image_path).name
    results = []

    # ── 1. 印刷文字 ──────────────────────────────────────────
    section(f"[{img_label}]  測試 1：印刷文字辨識")
    for prompt, label in PRINTED_TEXT_PROMPTS:
        r = ask(prompt, image_path, label)
        results.append({"category": "印刷文字", **r})

    # ── 2. 手寫文字 ──────────────────────────────────────────
    section(f"[{img_label}]  測試 2：手寫文字辨識（中 / 英）")
    for prompt, label in HANDWRITTEN_PROMPTS:
        r = ask(prompt, image_path, label)
        results.append({"category": "手寫文字", **r})

    # ── 3. 勾選框 ────────────────────────────────────────────
    section(f"[{img_label}]  測試 3：選項勾選辨識")
    for prompt, label in CHECKBOX_PROMPTS:
        r = ask(prompt, image_path, label)
        results.append({"category": "勾選框", **r})

    return results


def print_summary(all_results: dict):
    section("測試摘要")
    total_time = 0
    for img, results in all_results.items():
        print(f"\n  📄 {img}")
        for r in results:
            elapsed = r["elapsed"]
            total_time += elapsed
            ans_preview = r["answer"][:60].replace("\n", " ")
            print(f"    [{r['category']}] {r['label']}")
            print(f"      → {ans_preview}{'...' if len(r['answer']) > 60 else ''}")
            print(f"      ⏱ {elapsed:.1f}s")
    print(f"\n  總耗時: {total_time:.1f}s")


def save_results(all_results: dict):
    out_path = "ocr_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 完整結果已存至 {out_path}")


def main():
    print(f"\n🔍 gemma4:e2b OCR 測試開始")
    print(f"   模型: {MODEL}")
    print(f"   圖片: {', '.join(IMAGE_FILES)}")

    all_results = {}
    for img in IMAGE_FILES:
        if not Path(img).exists():
            print(f"  ⚠️  找不到 {img}，跳過")
            continue
        all_results[img] = run_tests_for_image(img)

    print_summary(all_results)
    save_results(all_results)
    print("\n✅ 測試完成！\n")


if __name__ == "__main__":
    main()
