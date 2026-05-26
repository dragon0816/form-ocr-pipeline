#!/usr/bin/env python3
"""
test_winrt.py — Windows WinRT OCR 相容性測試

在 Windows 機器上執行此腳本以驗證 WinRT OCR 是否可用。
macOS / Linux 上執行會顯示跳過訊息。

執行方式：
  python test_winrt.py [image.png]

安裝套件（Windows only）：
  pip install winrt-runtime winrt-Windows.Media.Ocr ^
              winrt-Windows.Globalization ^
              winrt-Windows.Graphics.Imaging ^
              winrt-Windows.Storage.Streams

語言包（Windows 設定）：
  設定 → 時間與語言 → 語言與地區 → 新增語言 → 中文(繁體台灣)
"""

import platform
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  Windows WinRT OCR 相容性測試")
print(f"  OS: {platform.system()} {platform.release()} {platform.machine()}")
print("=" * 60)


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print("─" * 60)


# ── Step 1: 作業系統確認 ─────────────────────────────────────
section("Step 1 / 作業系統確認")
if platform.system() != "Windows":
    print(f"  ⚠️  此腳本只適用於 Windows（目前: {platform.system()}）")
    print("  macOS: 請使用 Apple Vision（native_ocr.py 已自動選用）")
    print("  Linux: 請安裝 Tesseract：brew/apt install tesseract")
    sys.exit(0)

print("  ✅ Windows 環境確認")
print(f"     版本: {platform.version()}")


# ── Step 2: Python 環境 ──────────────────────────────────────
section("Step 2 / Python 環境")
print(f"  Python: {sys.version}")
print(f"  執行檔: {sys.executable}")


# ── Step 3: winrt 套件安裝確認 ───────────────────────────────
section("Step 3 / winrt 套件安裝確認")

REQUIRED_PACKAGES = [
    ("winrt.windows.media.ocr",           "winrt-Windows.Media.Ocr"),
    ("winrt.windows.globalization",       "winrt-Windows.Globalization"),
    ("winrt.windows.graphics.imaging",    "winrt-Windows.Graphics.Imaging"),
    ("winrt.windows.storage.streams",     "winrt-Windows.Storage.Streams"),
]

missing = []
for module, pkg in REQUIRED_PACKAGES:
    try:
        __import__(module)
        print(f"  ✅ {module}")
    except ImportError:
        print(f"  ❌ {module}  →  pip install {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n  ⚠️  缺少 {len(missing)} 個套件，請執行：")
    print(f"     pip install winrt-runtime {' '.join(missing)}")
    sys.exit(1)

print("\n  ✅ 所有 winrt 套件已安裝")


# ── Step 4: OcrEngine 語言確認 ───────────────────────────────
section("Step 4 / OcrEngine 語言包確認")

from winrt.windows.media.ocr import OcrEngine
from winrt.windows.globalization import Language

lang_tests = [
    ("zh-TW", "中文(繁體)"),
    ("zh-CN", "中文(簡體)"),
    ("en-US", "英文"),
]

available_langs = []
for lang_tag, label in lang_tests:
    try:
        lang = Language(lang_tag)
        engine = OcrEngine.try_create_from_language(lang)
        if engine is not None:
            print(f"  ✅ {lang_tag}  {label}")
            available_langs.append(lang_tag)
        else:
            print(f"  ❌ {lang_tag}  {label}  → 需要安裝語言包")
    except Exception as e:
        print(f"  ❌ {lang_tag}  {label}  → 錯誤: {e}")

if not available_langs:
    print("\n  ⚠️  找不到可用的 OCR 語言包！")
    print("     請至「設定 → 時間與語言 → 語言與地區 → 新增語言 → 中文(繁體台灣)」")
    sys.exit(1)

print(f"\n  ✅ 可用語言: {', '.join(available_langs)}")


# ── Step 5: 匯入 native_ocr ──────────────────────────────────
section("Step 5 / native_ocr 模組載入")

try:
    from native_ocr import get_ocr_engine, WindowsWinRTOCR
    engine = get_ocr_engine()
    print(f"  ✅ get_ocr_engine() → {engine.name}")
    assert isinstance(engine, WindowsWinRTOCR), f"預期 WindowsWinRTOCR，得到 {type(engine)}"
    print("  ✅ 確認選用 WinRT 引擎（非 Tesseract）")
except Exception as e:
    print(f"  ❌ 載入失敗: {e}")
    sys.exit(1)


# ── Step 6: 實際 OCR 測試 ────────────────────────────────────
section("Step 6 / 實際 OCR 辨識測試")

# 決定測試圖片
test_images = sys.argv[1:] or [
    p for p in ["page_01.png", "page_02.png"] if Path(p).exists()
]

if not test_images:
    print("  ⚠️  找不到測試圖片（page_01.png / page_02.png）")
    print("  請將圖片放在同一目錄，或以參數指定：python test_winrt.py your_image.png")
    sys.exit(0)

for img_path in test_images:
    if not Path(img_path).exists():
        print(f"  ⚠️  {img_path} 不存在，跳過")
        continue

    print(f"\n  📄 {img_path}")
    t0 = time.perf_counter()
    try:
        blocks = engine.recognize(img_path)
        elapsed = time.perf_counter() - t0
        print(f"  ✅ 辨識完成：{len(blocks)} 個區塊  ({elapsed:.3f}s)")

        # 顯示前 10 個高信心區塊
        high_conf = [b for b in blocks if b.confidence >= 0.8]
        print(f"     高信心（≥80%）區塊：{len(high_conf)} 個")
        for b in blocks[:10]:
            conf_str = f"{b.confidence:.0%}" if b.confidence < 1.0 else "—"
            print(f"       [{conf_str}]  {b.text[:50]}")
        if len(blocks) > 10:
            print(f"       …（共 {len(blocks)} 個）")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ❌ 辨識失敗（{elapsed:.2f}s）: {e}")


# ── 完成 ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  ✅ WinRT OCR 相容性測試完成")
print("  下一步：執行 python form_pipeline.py 跑完整 OCR 流水線")
print("=" * 60)
print()
