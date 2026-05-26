#!/usr/bin/env python3
"""
template_manager.py — 表單模板管理器  v2

職責分離：
  建立模板  --new-template IMAGE
    → 跑 OCR 偵測所有欄位與勾選框座標
    → 詢問名稱與說明
    → 全部預設啟用後存檔
    → 不做欄位選擇（那是「設定」的工作）

  設定模板  --config-template NAME
    → 載入已存模板
    → 互動式切換哪些欄位 / 勾選框要啟用
    → 存回原檔

  使用模板  --use-template NAME  （或自動選單）
    → 照模板目前設定直接處理，不再詢問
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

TEMPLATES_DIR    = Path("templates")
TEMPLATE_VERSION = 2


# ── 內部工具 ─────────────────────────────────────────────────
def _ensure_dir() -> None:
    TEMPLATES_DIR.mkdir(exist_ok=True)


def _safe_name(name: str) -> str:
    result = ""
    for c in name:
        if c.isalnum() or c in "-_. " or ord(c) > 127:
            result += c
        else:
            result += "_"
    return result.strip() or "template"


def _template_path(name: str) -> Path:
    return TEMPLATES_DIR / f"{_safe_name(name)}.json"


# ── CRUD ─────────────────────────────────────────────────────
def list_templates() -> list[dict]:
    _ensure_dir()
    result = []
    for p in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                t = json.load(f)
            t["_name"] = p.stem
            t["_file"] = str(p)
            result.append(t)
        except Exception:
            pass
    return result


def load_template(name: str) -> dict:
    p = _template_path(name)
    if not p.exists():
        raise FileNotFoundError(f"找不到模板：{p}")
    with open(p, encoding="utf-8") as f:
        t = json.load(f)
    t["_name"] = p.stem
    t["_file"] = str(p)
    return t


def save_template_dict(template: dict) -> Path:
    _ensure_dir()
    p = _template_path(template.get("name", "template"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    return p


def delete_template(name: str) -> None:
    p = _template_path(name)
    if not p.exists():
        raise FileNotFoundError(f"找不到模板：{p}")
    p.unlink()
    print(f"  🗑️  已刪除模板：{name}")


# ── 顯示 ─────────────────────────────────────────────────────
def print_template_list(templates: list[dict]) -> None:
    if not templates:
        print("  （尚無模板。執行 --new-template <圖片> 建立第一個模板）")
        return

    W = 62
    bar  = "═" * W
    dash = "─" * W
    print(f"\n  ╔{bar}╗")
    print(f"  ║{'可用模板（共 ' + str(len(templates)) + ' 個）':^{W}}║")
    print(f"  ╠{bar}╣")

    for i, t in enumerate(templates, 1):
        name    = t.get("name",  t.get("_name", "?"))
        desc    = t.get("description", "")
        src     = t.get("calibrated_from", "?")
        created = t.get("created_at", "")[:10]
        iw      = t.get("image_size", {}).get("w", "?")
        ih      = t.get("image_size", {}).get("h", "?")
        fields  = t.get("fields", {})
        cbs     = t.get("checkboxes", [])

        enabled_keys = [k for k, v in fields.items() if v.get("enabled", True)]
        n_f_on  = len(enabled_keys)
        n_f_all = len(fields)
        n_cb_on  = sum(1 for c in cbs if c.get("enabled", True))
        n_cb_all = len(cbs)

        print(f"  ║ [{i}] {name}")
        if desc:
            print(f"  ║     💬 {desc}")
        print(f"  ║     欄位 {n_f_on}/{n_f_all}：{', '.join(enabled_keys) or '（全部停用）'}")
        print(f"  ║     勾選框 {n_cb_on}/{n_cb_all}  "
              f"校準：{src}  {iw}×{ih}  建立：{created}")
        if i < len(templates):
            print(f"  ╠{dash}╣")

    print(f"  ╚{bar}╝")


# ── 互動選擇模板 ─────────────────────────────────────────────
def select_template_interactive(templates: list[dict]) -> Optional[dict]:
    """
    顯示清單讓使用者選擇。
    Enter → None（使用完整模式）
    d<n>  → 刪除第 n 個模板，然後回傳 None
    """
    print_template_list(templates)
    if not templates:
        return None

    print(f"\n  選擇模板 (1–{len(templates)})，"
          "Enter = 完整模式，d<編號> = 刪除（如 d2）：",
          end="", flush=True)
    line = input().strip()

    if not line:
        return None

    if line.lower().startswith("d"):
        try:
            idx = int(line[1:]) - 1
            if 0 <= idx < len(templates):
                t = templates[idx]
                tname = t.get("name", t.get("_name", "?"))
                print(f"  確認刪除「{tname}」？(y/N)：", end="", flush=True)
                if input().strip().lower() == "y":
                    delete_template(t.get("_name", tname))
                    print("  ✅ 已刪除，使用完整模式")
                else:
                    print("  已取消")
        except (ValueError, FileNotFoundError) as e:
            print(f"  ⚠️  {e}")
        return None

    try:
        idx = int(line) - 1
        if 0 <= idx < len(templates):
            selected = templates[idx]
            print(f"  ✅ 已選：{selected.get('name', '?')}")
            return selected
    except ValueError:
        pass

    print("  ⚠️  無效選擇，使用完整模式")
    return None


# ── 通用切換選單 ─────────────────────────────────────────────
def _toggle_menu(
    items:    list[dict],
    key_fn:   callable,
    label_fn: callable,
    enabled:  dict,
    title:    str,
) -> None:
    """顯示開/關切換選單，直到使用者按 Enter 確認"""
    W = 60
    while True:
        print(f"\n  ╔══ {title} {'═' * max(0, W - 4 - len(title))}╗")
        for i, item in enumerate(items, 1):
            k  = key_fn(item)
            e  = "✅" if enabled.get(k, True) else "☐ "
            print(f"  ║ [{i:2d}] {e}  {label_fn(item)}")
        on    = sum(1 for v in enabled.values() if v)
        total = len(items)
        hint  = f" {on}/{total} ║ 輸入編號切換 ║ a=全選 ║ n=全不選 ║ Enter=確認 "
        print(f"  ╚═{hint}{'═' * max(0, W - 1 - len(hint))}╝")
        print("  ▶ ", end="", flush=True)

        line = input().strip().lower()
        if not line:
            break
        if line == "a":
            for k in enabled:
                enabled[k] = True
        elif line == "n":
            for k in enabled:
                enabled[k] = False
        else:
            for token in line.split():
                try:
                    n = int(token) - 1
                    if 0 <= n < len(items):
                        k = key_fn(items[n])
                        enabled[k] = not enabled[k]
                except ValueError:
                    pass


# ── 建立模板（只偵測 + 命名，不做選擇）──────────────────────
def create_template(
    calibration_image: str,
    image_size:        tuple[int, int],
    fields_info:       list[dict],     # [{key, label_cn, roi: BBox}]
    checkboxes_info:   list[dict],     # [{label, bbox: BBox}]
) -> Optional[dict]:
    """
    建立新模板：記錄所有偵測到的位置，全部預設啟用。
    不做欄位選擇 — 那是 config_template() 的工作。
    """
    iw, ih = image_size
    valid  = [f for f in fields_info if f.get("roi") is not None]

    # 顯示偵測結果摘要
    print(f"\n  ╔══ 新模板偵測結果 ══════════════════════════════════════╗")
    print(f"  ║  校準圖片：{calibration_image:<46}║")
    print(f"  ║  圖片尺寸：{iw}×{ih} px{'':<37}║")
    print(f"  ║{'─'*62}║")
    print(f"  ║  偵測到的欄位（{len(valid)} 個）：")
    for f in valid:
        print(f"  ║    • {f['key']:16s} ({f['label_cn']})")
    print(f"  ║{'─'*62}║")
    print(f"  ║  偵測到的勾選框（{len(checkboxes_info)} 個）：")
    labels = [c["label"] for c in checkboxes_info]
    # 每行最多 4 個
    for i in range(0, len(labels), 4):
        row = "  ".join(f"{l:<20}" for l in labels[i:i+4])
        print(f"  ║    {row}")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    # 命名
    print()
    print("  請輸入模板名稱（可含中文）：", end="", flush=True)
    name = input().strip()
    if not name:
        print("  ⚠️  名稱為空，取消建立")
        return None

    if _template_path(name).exists():
        print(f"  ⚠️  「{name}」已存在，是否覆蓋？(y/N)：",
              end="", flush=True)
        if input().strip().lower() != "y":
            print("  已取消")
            return None

    print("  說明（選填，Enter 跳過）：", end="", flush=True)
    desc = input().strip()

    # 組裝（全部啟用）
    field_map: dict = {}
    for f in valid:
        roi = f["roi"]
        field_map[f["key"]] = {
            "enabled":  True,
            "label_cn": f["label_cn"],
            "rx": round(roi.x / iw, 5),
            "ry": round(roi.y / ih, 5),
            "rw": round(roi.w / iw, 5),
            "rh": round(roi.h / ih, 5),
        }

    cb_list: list[dict] = []
    for cb in checkboxes_info:
        bbox = cb["bbox"]
        cb_list.append({
            "label":   cb["label"],
            "enabled": True,
            "crx": round(bbox.x / iw, 5),
            "cry": round(bbox.y / ih, 5),
            "crw": round(bbox.w / iw, 5),
            "crh": round(bbox.h / ih, 5),
        })

    template = {
        "version":         TEMPLATE_VERSION,
        "name":            name,
        "description":     desc,
        "calibrated_from": calibration_image,
        "created_at":      datetime.now().isoformat(timespec="seconds"),
        "image_size":      {"w": iw, "h": ih},
        "fields":          field_map,
        "checkboxes":      cb_list,
    }

    p = save_template_dict(template)
    print(f"\n  ✅ 模板已儲存：{p}")
    print(f"     {len(field_map)} 個欄位、{len(cb_list)} 個勾選框，全部預設啟用")
    print(f"  💡 執行 --config-template \"{name}\" 可自訂要擷取哪些項目")
    return template


# ── 設定模板（開/關欄位與勾選框）────────────────────────────
def config_template(name: str) -> Optional[dict]:
    """
    載入已存模板，讓使用者切換欄位與勾選框的啟用狀態，存回原檔。
    """
    try:
        template = load_template(name)
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return None

    tname  = template.get("name", name)
    fields = template.get("fields", {})
    cbs    = template.get("checkboxes", [])

    n_f_on  = sum(1 for v in fields.values() if v.get("enabled", True))
    n_cb_on = sum(1 for c in cbs if c.get("enabled", True))

    print(f"\n  ╔══ 設定模板：{tname} ══════════════════════════════════╗")
    print(f"  ║  當前啟用：欄位 {n_f_on}/{len(fields)}，勾選框 {n_cb_on}/{len(cbs)}")
    print(f"  ╚════════════════════════════════════════════════════════╝")

    # 欄位切換
    field_items   = [{"key": k, "label_cn": v.get("label_cn", k)}
                     for k, v in fields.items()]
    field_enabled = {k: v.get("enabled", True) for k, v in fields.items()}
    _toggle_menu(
        field_items,
        key_fn=lambda f: f["key"],
        label_fn=lambda f: f"{f['key']:16s}  ({f['label_cn']})",
        enabled=field_enabled,
        title="欄位設定",
    )

    # 勾選框切換
    if cbs:
        cb_items   = [{"label": c["label"]} for c in cbs]
        cb_enabled = {c["label"]: c.get("enabled", True) for c in cbs}
        _toggle_menu(
            cb_items,
            key_fn=lambda c: c["label"],
            label_fn=lambda c: c["label"],
            enabled=cb_enabled,
            title="勾選框設定",
        )
    else:
        cb_enabled = {}

    # 寫回 template dict
    for k, v in fields.items():
        v["enabled"] = field_enabled.get(k, True)
    for c in cbs:
        c["enabled"] = cb_enabled.get(c["label"], True)

    p = save_template_dict(template)

    on_f  = sum(1 for v in field_enabled.values() if v)
    on_cb = sum(1 for v in cb_enabled.values() if v) if cb_enabled else 0
    skipped = [k for k, v in field_enabled.items() if not v]

    print(f"\n  ✅ 模板已更新：{p}")
    print(f"     啟用欄位：{on_f}/{len(fields)}"
          + (f"  略過：{', '.join(skipped)}" if skipped else "  （全部）"))
    print(f"     啟用勾選框：{on_cb}/{len(cbs)}")
    return template


# ── 獨立執行：列出模板 ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    templates = list_templates()
    print_template_list(templates)
    if templates:
        print(f"\n  共 {len(templates)} 個模板，存於 {TEMPLATES_DIR}/")
