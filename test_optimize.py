#!/usr/bin/env python3
"""
零成本優化方案測試
比較以下組合對速度與精度的影響：
  模型：qwen2.5vl:7b（基線）/ qwen2.5vl:7b-q4_K_M（INT4）/ qwen2.5vl:3b
  圖片：原尺寸 / resize 50% / resize 30%
"""

import base64
import io
import json
import sys
import time
from pathlib import Path

import ollama
from PIL import Image

# ── 設定 ────────────────────────────────────────────────────
MODELS = [
    "qwen2.5vl:3b",           # 小模型（7b-q4_K_M = 7b 同一模型，跳過）
]
RESIZE_SCALES = [1.0, 0.75, 0.5, 0.3]   # 原尺寸 / 75% / 50% / 30%
TEST_IMAGE = "page_01.png"               # 只用 page_01 做基準測試

# 代表性 prompt（涵蓋快/中/慢三種難度）
PROMPTS = [
    (
        "header",
        "🖨️ 標題",
        "What is the company name in the logo and the form title in Chinese? Answer in one short paragraph.",
    ),
    (
        "contact",
        "✍️ 聯絡資訊",
        "Read the handwritten fields: 姓名, 公司, 聯絡手機, Email. "
        "List each field and its value only.",
    ),
    (
        "checkbox",
        "☑️ 勾選框",
        "List ALL checkboxes and whether each is checked or unchecked. "
        "Format: - [item]: checked/unchecked",
    ),
    (
        "full_extract",
        "📄 全文擷取",
        "Extract ALL text visible in this form image — both printed and handwritten. "
        "Be thorough and structured.",
    ),
]

# ── 工具函式 ────────────────────────────────────────────────

def resize_image_b64(path: str, scale: float) -> tuple[str, tuple[int, int]]:
    """讀圖、縮放、轉 base64；回傳 (b64, (w, h))"""
    img = Image.open(path)
    orig_w, orig_h = img.size
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, img.size


def ask(model: str, prompt: str, img_b64: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
            options={"temperature": 0},
        )
        answer = resp["message"]["content"].strip()
    except Exception as e:
        answer = f"[ERROR: {e}]"
    return answer, time.time() - t0


def is_available(model: str) -> bool:
    resp = ollama.list()
    ml = resp.models if hasattr(resp, "models") else resp.get("models", [])
    names = {(m.model if hasattr(m, "model") else m.get("model", "")) for m in ml}
    return model in names


def scale_label(s: float) -> str:
    return f"{int(s*100)}%"


# ── 主程式 ──────────────────────────────────────────────────

def run():
    print(f"\n{'='*65}")
    print("  🚀 零成本優化方案測試")
    print(f"  圖片: {TEST_IMAGE}")
    print(f"  模型: {', '.join(MODELS)}")
    print(f"  縮圖: {[scale_label(s) for s in RESIZE_SCALES]}")
    print("=" * 65)

    orig_img = Image.open(TEST_IMAGE)
    print(f"\n  原始圖片尺寸: {orig_img.size[0]}×{orig_img.size[1]} px\n")

    # 收集結果
    results = {}   # results[model][scale][prompt_slug] = {answer, elapsed}

    for model in MODELS:
        if not is_available(model):
            print(f"  ⚠️  {model} 尚未安裝，跳過")
            continue

        print(f"\n{'─'*65}")
        print(f"  模型: {model}")
        print(f"{'─'*65}")
        results[model] = {}

        for scale in RESIZE_SCALES:
            label = scale_label(scale)
            img_b64, (w, h) = resize_image_b64(TEST_IMAGE, scale)
            size_kb = len(img_b64) * 3 // 4 // 1024
            print(f"\n  [{label}] {w}×{h}px  (~{size_kb} KB)")
            results[model][label] = {}

            for slug, category, prompt in PROMPTS:
                answer, elapsed = ask(model, prompt, img_b64)
                ok = "✅" if answer and "[ERROR" not in answer else "❌"
                tok_est = len(answer) // 4
                hint = "🎯" if elapsed <= 3 else ("⚡" if elapsed <= 8 else "🐢")
                print(f"    {ok}{hint} {category:10s} {elapsed:5.1f}s  "
                      f"~{tok_est:3d}tok  {answer[:60].replace(chr(10),' ')}…")
                results[model][label][slug] = {
                    "elapsed": round(elapsed, 2),
                    "tokens_est": tok_est,
                    "answer": answer,
                }

    # ── 摘要表 ──────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  📊 速度摘要（秒）— 越小越好，🎯 = ≤3s 達標")
    print("=" * 65)

    # 表頭
    slugs = [s for s, _, _ in PROMPTS]
    header = f"  {'模型+縮圖':28s}" + "".join(f"  {s:12s}" for s in slugs) + "  平均"
    print(header)
    print("  " + "─" * (len(header) - 2))

    all_avgs = []
    for model in MODELS:
        if model not in results:
            continue
        short_model = model.replace("qwen2.5vl:", "").replace("-q4_K_M", "-INT4")
        for scale_lbl, pdata in results[model].items():
            row_label = f"{short_model}+{scale_lbl}"
            times = [pdata[s]["elapsed"] for s in slugs if s in pdata]
            avg = sum(times) / len(times) if times else 0
            all_avgs.append((row_label, avg))
            cells = ""
            for s in slugs:
                t = pdata.get(s, {}).get("elapsed", 0)
                flag = "🎯" if t <= 3 else ("⚡" if t <= 8 else "")
                cells += f"  {t:4.1f}s {flag:2s}    "
            print(f"  {row_label:28s}{cells}  {avg:.1f}s")

    # 最佳組合
    print(f"\n  🏆 最快組合（平均最低）:")
    all_avgs.sort(key=lambda x: x[1])
    for rank, (name, avg) in enumerate(all_avgs[:3], 1):
        flag = "🎯" if avg <= 3 else ("⚡" if avg <= 8 else "")
        print(f"     #{rank} {name:30s}  avg {avg:.1f}s {flag}")

    # 儲存
    out = "optimize_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 完整結果存至 {out}")
    print("\n✅ 優化測試完成！\n")


if __name__ == "__main__":
    run()
