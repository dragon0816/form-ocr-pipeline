#!/usr/bin/env python3
"""
OCR 模型對比測試腳本
比較 deepseek-ocr:latest vs qwen2.5-vl:7b
（同時載入 gemma4:e2b 的已知結果作為基線）
"""

import base64
import json
import time
import sys
from pathlib import Path
import ollama

MODELS = ["deepseek-ocr:latest", "qwen2.5vl:7b"]
IMAGE_FILES = ["page_01.png", "page_02.png"]

# ─────────────────────────────────────────────────────────────
# 測試題目（與 gemma4 測試相同）
# ─────────────────────────────────────────────────────────────
TESTS = [
    # (slug, category, prompt)
    (
        "print_full",
        "🖨️ 印刷文字",
        "Please extract ALL the printed (non-handwritten) text from this form. "
        "List each printed label/heading exactly as it appears.",
    ),
    (
        "print_header",
        "🖨️ 印刷文字",
        "What is the company name shown in the logo/header of this form? "
        "What is the form title in Chinese?",
    ),
    (
        "hw_all",
        "✍️ 手寫辨識",
        "This form contains handwritten text. Please read and transcribe "
        "ALL handwritten content you can see, including names, emails, dates, "
        "and any free-text answers. Preserve the original language (Chinese or English).",
    ),
    (
        "hw_contact",
        "✍️ 手寫辨識",
        "Read the handwritten fields: 姓名 (name), 公司 (company), 聯絡手機 (phone), "
        "部門 (department), Email. What values are written in each field?",
    ),
    (
        "hw_bottom",
        "✍️ 手寫辨識",
        "There is a handwritten section at the bottom of the form. "
        "Please transcribe all handwritten text in that section as accurately as possible.",
    ),
    (
        "cb_all",
        "☑️ 勾選框",
        "This form has many checkboxes (□). Please list EVERY checkbox option "
        "and indicate whether each one is checked (✓/X/filled) or unchecked (empty □). "
        "Group them by section if possible.",
    ),
    (
        "cb_product",
        "☑️ 勾選框",
        "Look at the checkboxes in the section about product/solution categories. "
        "Which boxes are checked (filled/ticked) and which are unchecked (empty)?",
    ),
    (
        "cb_equip",
        "☑️ 勾選框",
        "In the measurement equipment section (Call box, Source Measurement Unit, "
        "Oscilloscope, Spectrum Analyzer, Vector Network Analyzer, LCR Meter, Power Supply), "
        "list each item and whether its checkbox is checked or unchecked.",
    ),
]

# ─────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask(model: str, prompt: str, image_path: str) -> tuple[str, float]:
    """送出問題，回傳 (回答, 耗時秒)"""
    img_b64 = load_image_b64(image_path)
    t0 = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"temperature": 0},
        )
        answer = response["message"]["content"].strip()
    except Exception as e:
        answer = f"[ERROR: {e}]"
    elapsed = time.time() - t0
    return answer, elapsed


def bar(score: float, width: int = 20) -> str:
    filled = int(score / 5 * width)
    return "█" * filled + "░" * (width - filled)


def truncate(text: str, n: int = 120) -> str:
    text = text.replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


# ─────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────

def run_model(model: str, image_files: list[str]) -> dict:
    """執行一個模型的所有測試，回傳結果字典"""
    results = {}
    print(f"\n{'='*65}")
    print(f"  模型: {model}")
    print("=" * 65)

    for img in image_files:
        if not Path(img).exists():
            print(f"  ⚠️  找不到 {img}，跳過")
            continue
        img_key = Path(img).name
        results[img_key] = {}

        for slug, category, prompt in TESTS:
            print(f"\n  [{img_key}] {category} — {slug}")
            answer, elapsed = ask(model, prompt, img)
            results[img_key][slug] = {"answer": answer, "elapsed": elapsed}

            status = "✅" if answer and not answer.startswith("[ERROR") else "❌"
            print(f"  {status} ({elapsed:.1f}s) {truncate(answer)}")

    return results


def print_comparison(all_results: dict):
    """橫向對比表格"""
    print(f"\n\n{'='*65}")
    print("  📊  橫向對比摘要")
    print("=" * 65)

    models = list(all_results.keys())

    for img in IMAGE_FILES:
        img_key = Path(img).name
        print(f"\n  📄 {img_key}")
        print(f"  {'測試':30s}", end="")
        for m in models:
            col = m.split(":")[0][-16:]
            print(f"  {col:>16s}", end="")
        print()
        print("  " + "-" * (30 + 18 * len(models)))

        for slug, category, _ in TESTS:
            label = f"{category} [{slug}]"
            print(f"  {label:30s}", end="")
            for m in models:
                r = all_results.get(m, {}).get(img_key, {}).get(slug, {})
                elapsed = r.get("elapsed", 0)
                answer  = r.get("answer", "")
                ok = "✅" if answer and not answer.startswith("[ERROR") else "❌"
                print(f"  {ok} {elapsed:>6.1f}s      ", end="")
            print()


def print_speed_summary(all_results: dict):
    print(f"\n\n{'='*65}")
    print("  ⏱  速度對比（各模型總耗時）")
    print("=" * 65)
    for model, img_data in all_results.items():
        total = sum(
            r.get("elapsed", 0)
            for img_results in img_data.values()
            for r in img_results.values()
        )
        n = sum(len(v) for v in img_data.values())
        avg = total / n if n else 0
        print(f"  {model:<30s}  總計 {total:>6.1f}s  平均 {avg:.1f}s/題")


def save_results(all_results: dict):
    out = "compare_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 完整結果存至 {out}")


def main():
    # 確認模型可用
    resp = ollama.list()
    model_list = resp.models if hasattr(resp, "models") else resp.get("models", [])
    available = {(m.model if hasattr(m, "model") else m.get("model", m.get("name", ""))) for m in model_list}
    to_run = []
    for m in MODELS:
        if m in available:
            to_run.append(m)
        else:
            print(f"  ⚠️  模型 {m} 尚未安裝，跳過")

    if not to_run:
        print("沒有可用的模型，結束。")
        sys.exit(1)

    print(f"\n🔍 OCR 對比測試開始")
    print(f"   模型: {', '.join(to_run)}")
    print(f"   圖片: {', '.join(IMAGE_FILES)}")

    all_results = {}
    for model in to_run:
        all_results[model] = run_model(model, IMAGE_FILES)

    print_comparison(all_results)
    print_speed_summary(all_results)
    save_results(all_results)
    print("\n✅ 對比測試完成！\n")


if __name__ == "__main__":
    main()
