"""
native_ocr.py — 跨平台本機 OCR 模組

自動偵測作業系統，選用最快的內建 OCR API：
  macOS   → Apple Vision Framework  (pyobjc)
  Windows → Windows.Media.Ocr       (winrt)
  Linux   → Tesseract                (pytesseract) [fallback]

統一輸出格式：List[TextBlock]
  text       : 辨識出的文字
  confidence : 信心值 0.0–1.0
  bbox       : BBox(x, y, w, h)，像素座標，左上角為原點
  source     : 來源引擎名稱
"""

import io
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── 資料結構 ───────────────────────────────────────────────

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def __str__(self):
        return f"({self.x},{self.y} {self.w}×{self.h})"


@dataclass
class TextBlock:
    text: str
    confidence: float        # 0.0 – 1.0
    bbox: Optional[BBox]     # pixel coords, top-left origin
    source: str              # "vision" / "winrt" / "tesseract"

    def __str__(self):
        conf = f"{self.confidence:.0%}" if self.confidence < 1.0 else "—"
        bbox = str(self.bbox) if self.bbox else "N/A"
        return f"[{conf}] {bbox}  {self.text}"


# ── 基礎類別 ───────────────────────────────────────────────

class OCRBackend:
    name: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def recognize(self, image_path: str) -> list[TextBlock]:
        raise NotImplementedError

    def install_hint(self) -> str:
        return ""


# ── macOS：Apple Vision ────────────────────────────────────

class AppleVisionOCR(OCRBackend):
    name = "Apple Vision"

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            import Vision  # noqa: F401
            return True
        except ImportError:
            return False

    def install_hint(self) -> str:
        return "pip install pyobjc-framework-Vision"

    def recognize(self, image_path: str) -> list[TextBlock]:
        import Vision
        from Foundation import NSURL
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        img_w, img_h = img.size

        abs_path = str(Path(image_path).resolve())
        image_url = NSURL.fileURLWithPath_(abs_path)

        request = Vision.VNRecognizeTextRequest.alloc().init()
        # Accurate 模式：使用神經網路，較慢但更準
        request.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate
        )
        # 語言優先順序：繁中 → 簡中 → 英文
        request.setRecognitionLanguages_(["zh-Hant", "zh-Hans", "en-US"])
        request.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
            image_url, {}
        )
        success, error = handler.performRequests_error_([request], None)

        if not success:
            raise RuntimeError(f"Vision OCR failed: {error}")

        results: list[TextBlock] = []
        for obs in request.results():
            candidates = obs.topCandidates_(1)
            if not candidates:
                continue
            candidate = candidates[0]
            text = str(candidate.string())
            conf = float(candidate.confidence())

            # Vision 座標系：原點在左下，Y 軸朝上，值為 0–1
            # 轉換為像素座標，原點在左上
            nb = obs.boundingBox()
            px = int(nb.origin.x * img_w)
            pw = int(nb.size.width  * img_w)
            ph = int(nb.size.height * img_h)
            py = int((1.0 - nb.origin.y - nb.size.height) * img_h)

            results.append(TextBlock(
                text=text,
                confidence=conf,
                bbox=BBox(px, py, pw, ph),
                source="vision",
            ))

        # 依 Y 座標排序（由上至下）
        results.sort(key=lambda b: b.bbox.y if b.bbox else 0)
        return results


# ── Windows：WinRT OCR ─────────────────────────────────────

class WindowsWinRTOCR(OCRBackend):
    name = "Windows WinRT OCR"

    def is_available(self) -> bool:
        if platform.system() != "Windows":
            return False
        try:
            import winrt.windows.media.ocr  # noqa: F401
            return True
        except ImportError:
            return False

    def install_hint(self) -> str:
        return (
            "pip install winrt-runtime "
            "winrt-Windows.Media.Ocr "
            "winrt-Windows.Globalization "
            "winrt-Windows.Graphics.Imaging "
            "winrt-Windows.Storage.Streams"
        )

    def recognize(self, image_path: str) -> list[TextBlock]:
        import asyncio
        return asyncio.run(self._recognize_async(image_path))

    async def _recognize_async(self, image_path: str) -> list[TextBlock]:
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.globalization import Language
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage.streams import (
            InMemoryRandomAccessStream,
            DataWriter,
        )
        from PIL import Image as PILImage

        # 選擇可用的語言引擎
        engine = None
        for lang_tag in ["zh-TW", "zh-CN", "en-US"]:
            lang = Language(lang_tag)
            e = OcrEngine.try_create_from_language(lang)
            if e is not None:
                engine = e
                break
        if engine is None:
            raise RuntimeError(
                "找不到可用的 OCR 語言套件。"
                "請至「設定 → 時間與語言 → 語言與地區」安裝繁中語言包。"
            )

        # PIL → BMP bytes → WinRT stream → SoftwareBitmap
        img = PILImage.open(image_path).convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        data = buf.getvalue()

        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream.get_output_stream_at(0))
        writer.write_bytes(data)
        await writer.store_async()
        writer.detach_stream()
        stream.seek(0)

        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()

        result = await engine.recognize_async(bitmap)

        results: list[TextBlock] = []
        for line in result.lines:
            words = line.words
            if not words:
                continue

            # 合併同行文字（CJK 不加空格，英數加空格）
            texts = [str(w.text) for w in words]
            text = _smart_join(texts)

            # 從首末字取 bounding rect
            r0 = words[0].bounding_rect
            r1 = words[-1].bounding_rect
            x = int(r0.x)
            y = int(r0.y)
            w = int(r1.x + r1.width) - x
            h = int(max(r0.height, r1.height))

            results.append(TextBlock(
                text=text,
                confidence=1.0,   # WinRT 不提供逐行信心值
                bbox=BBox(x, y, w, h),
                source="winrt",
            ))

        results.sort(key=lambda b: b.bbox.y if b.bbox else 0)
        return results


def _smart_join(words: list[str]) -> str:
    """CJK 字元之間不加空格，英數之間加空格"""
    result = ""
    for i, w in enumerate(words):
        if i == 0:
            result = w
            continue
        prev_last = result[-1] if result else ""
        cur_first = w[0] if w else ""
        # 若前後都是 ASCII，加空格
        if prev_last.isascii() and cur_first.isascii():
            result += " " + w
        else:
            result += w
    return result


# ── Fallback：Tesseract ────────────────────────────────────

class TesseractOCR(OCRBackend):
    name = "Tesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def install_hint(self) -> str:
        return (
            "# 安裝 Tesseract 執行檔：\n"
            "#   macOS:   brew install tesseract tesseract-lang\n"
            "#   Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "#   Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-chi-tra\n"
            "pip install pytesseract"
        )

    def recognize(self, image_path: str) -> list[TextBlock]:
        import pytesseract
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        data = pytesseract.image_to_data(
            img,
            lang="chi_tra+chi_sim+eng",
            output_type=pytesseract.Output.DICT,
        )

        results: list[TextBlock] = []
        n = len(data["text"])
        for i in range(n):
            text = str(data["text"][i]).strip()
            conf = int(data["conf"][i])
            if not text or conf < 0:
                continue
            results.append(TextBlock(
                text=text,
                confidence=conf / 100.0,
                bbox=BBox(
                    data["left"][i], data["top"][i],
                    data["width"][i], data["height"][i],
                ),
                source="tesseract",
            ))
        return results


# ── 自動選擇引擎 ───────────────────────────────────────────

def get_ocr_engine() -> OCRBackend:
    """
    偵測 OS，依優先順序選出可用的 OCR 引擎。
    找不到任何引擎時拋出 RuntimeError 並顯示安裝指示。
    """
    system = platform.system()

    priority: list[OCRBackend] = []
    if system == "Darwin":
        priority = [AppleVisionOCR(), TesseractOCR()]
    elif system == "Windows":
        priority = [WindowsWinRTOCR(), TesseractOCR()]
    else:
        priority = [TesseractOCR()]

    for backend in priority:
        if backend.is_available():
            return backend

    # 全部失敗 → 顯示安裝提示
    hints = "\n".join(
        f"  [{b.name}]\n  {b.install_hint()}"
        for b in priority
    )
    raise RuntimeError(
        f"找不到可用的 OCR 引擎（OS: {system}）。\n"
        f"請安裝以下其中一項：\n{hints}"
    )


# ── 快速測試 ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    images = sys.argv[1:] or ["page_01.png", "page_02.png"]

    print(f"\n{'='*60}")
    print("  跨平台 Native OCR 測試")
    print(f"  OS: {platform.system()} {platform.release()}")

    try:
        engine = get_ocr_engine()
    except RuntimeError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    print(f"  引擎: {engine.name}")
    print("=" * 60)

    all_results = {}

    for img_path in images:
        if not Path(img_path).exists():
            print(f"\n  ⚠️  找不到 {img_path}，跳過")
            continue

        print(f"\n  📄 {img_path}")
        t0 = time.perf_counter()
        try:
            blocks = engine.recognize(img_path)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  ❌ 辨識失敗：{e}")
            continue

        print(f"  ⏱  耗時：{elapsed:.3f}s")
        print(f"  📝 辨識出 {len(blocks)} 個文字區塊\n")

        for b in blocks:
            print(f"    {b}")

        all_results[img_path] = {
            "engine": engine.name,
            "elapsed": round(elapsed, 3),
            "blocks": [
                {
                    "text": b.text,
                    "confidence": round(b.confidence, 3),
                    "bbox": {"x": b.bbox.x, "y": b.bbox.y,
                             "w": b.bbox.w, "h": b.bbox.h}
                    if b.bbox else None,
                    "source": b.source,
                }
                for b in blocks
            ],
        }

    # 儲存結果
    import json
    out = "native_ocr_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 結果存至 {out}")
    print("\n✅ 完成！\n")
