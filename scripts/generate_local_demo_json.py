# -*- coding: utf-8 -*-
"""
Generate local JSON outputs (no server) for TongueExpert pipeline.

Inputs:
- ../outputs/e2e_test/e2e_test_all.csv
- ../outputs/p14_embedding/p14_emb_test.csv

Outputs:
- ../outputs/demo_api_json/<id>.json
- ../outputs/demo_api_json/_index.json (available ids)
"""

import os
import json
from typing import Dict, Any, Optional, List

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

E2E_CSV = "../outputs/e2e_test/e2e_test_all.csv"
P14_CSV = "../outputs/p14_embedding/p14_emb_test.csv"
OUT_DIR = "../outputs/demo_api_json"

# If set to a value that doesn't exist, script will auto-fallback.
SAMPLE_ID: Optional[str] = "TE0000161"

# whether to include full vectors (large)
INCLUDE_FULL_VECTORS = False

# how many embedding dims to preview in JSON
EMB_PREVIEW_K = 10

# default split folder names (adjust if your split naming differs)
DEFAULT_SPLIT = "test"
ROI_DIR = "../outputs/roi"
MASK_DIR = "../outputs/pred_masks_original"


def pick_first_existing(row: pd.Series, candidates: List[str]):
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return None


def make_interpretation(color: Dict[str, Any], texture: Dict[str, Any], shape: Dict[str, Any]) -> Dict[str, Any]:
    # color
    r_mean = color["key_features"].get("r_mean", None)
    g_mean = color["key_features"].get("g_mean", None)
    l_mean = color["key_features"].get("lightness_mean", None)
    rg_diff = color["key_features"].get("rg_diff", None)
    r_over_g = color["key_features"].get("r_over_g", None)

    # texture
    hom = texture["key_features"].get("homogeneity", None)
    ent = texture["key_features"].get("entropy", None)

    # shape
    ratio = shape.get("width_height_ratio", None)

    signals = []
    parts = []

    # ---- Brightness (0~255 scale) ----
    if l_mean is not None:
        if l_mean >= 160:
            parts.append("舌体整体偏明亮（亮度较高）")
            signals.append({"domain": "color", "feature": "lightness_mean", "value": l_mean, "interpretation": "brightness_high"})
        elif l_mean <= 120:
            parts.append("舌体整体偏暗（亮度较低）")
            signals.append({"domain": "color", "feature": "lightness_mean", "value": l_mean, "interpretation": "brightness_low"})
        else:
            parts.append("舌体亮度中等")
            signals.append({"domain": "color", "feature": "lightness_mean", "value": l_mean, "interpretation": "brightness_mid"})

    # ---- Redness (relative to green) ----
    # prefer rg_diff; fall back to r_over_g; as last resort show r_mean only (less reliable)
    if rg_diff is not None or r_over_g is not None:
        # practical thresholds for 0~255 stats (tunable)
        is_red_high = (rg_diff is not None and rg_diff >= 15) or (r_over_g is not None and r_over_g >= 1.12)
        is_red_low = (rg_diff is not None and rg_diff <= 5) or (r_over_g is not None and r_over_g <= 1.02)

        if is_red_high:
            parts.append("舌色偏红（红相对绿更强）")
            signals.append({"domain": "color", "feature": "rg_diff", "value": rg_diff, "interpretation": "redness_high"})
        elif is_red_low:
            parts.append("舌色偏淡/偏白方向（红相对绿偏弱）")
            signals.append({"domain": "color", "feature": "rg_diff", "value": rg_diff, "interpretation": "redness_low"})
        else:
            parts.append("舌色红度处于中等水平")
            signals.append({"domain": "color", "feature": "rg_diff", "value": rg_diff, "interpretation": "redness_mid"})
    elif r_mean is not None:
        parts.append("舌色已量化（仅红通道均值可用，受曝光影响较大）")
        signals.append({"domain": "color", "feature": "r_mean", "value": r_mean, "interpretation": "red_channel_only"})

    # ---- Texture ----
    if hom is not None and ent is not None:
        if hom > 0.8 and ent < 7.2:
            parts.append("表面纹理较均匀（偏薄苔/平滑倾向）")
            signals.append({"domain": "texture", "feature": "homogeneity", "value": hom, "interpretation": "surface_homogeneous"})
        elif ent > 8.0:
            parts.append("表面纹理复杂度偏高（可能有苔覆盖/颗粒差异）")
            signals.append({"domain": "texture", "feature": "entropy", "value": ent, "interpretation": "surface_complex"})
        else:
            parts.append("表面纹理处于中等水平")

    # ---- Shape ----
    if ratio is not None:
        parts.append("舌形宽高比已计算")

    summary_cn = "；".join(parts) + "。"
    return {
        "summary_cn": summary_cn,
        "signals": signals,
        "disclaimer": "该结果用于科研/工程验证，不构成医疗诊断或建议。"
    }


def choose_id(e2e_ids: pd.Series, p14_ids: pd.Series, preferred: Optional[str]) -> str:
    e2e_set = set(e2e_ids.astype(str).tolist())
    p14_set = set(p14_ids.astype(str).tolist())
    inter = sorted(list(e2e_set.intersection(p14_set)))

    if preferred is not None and preferred in e2e_set and preferred in p14_set:
        return preferred

    if len(inter) == 0:
        return str(e2e_ids.iloc[0])

    return inter[0]


def make_artifacts(sid: str) -> Dict[str, Any]:
    roi_path = os.path.normpath(os.path.join(ROI_DIR, DEFAULT_SPLIT, f"{sid}.jpg"))
    mask_path = os.path.normpath(os.path.join(MASK_DIR, DEFAULT_SPLIT, f"{sid}.png"))
    return {
        "generated": {
            "roi_path": roi_path if os.path.exists(roi_path) else None,
            "mask_path": mask_path if os.path.exists(mask_path) else None
        }
    }


def main():
    assert os.path.exists(E2E_CSV), f"Missing: {E2E_CSV}"
    assert os.path.exists(P14_CSV), f"Missing: {P14_CSV}"

    os.makedirs(OUT_DIR, exist_ok=True)

    e2e = pd.read_csv(E2E_CSV)
    p14 = pd.read_csv(P14_CSV)

    if "id" not in e2e.columns:
        raise RuntimeError(f"e2e csv has no id column: {E2E_CSV} (cols={e2e.columns.tolist()})")
    if "id" not in p14.columns:
        raise RuntimeError(f"p14 csv has no id column: {P14_CSV} (cols={p14.columns.tolist()})")

    # write an index for easy picking
    e2e_set = set(e2e["id"].astype(str).tolist())
    p14_set = set(p14["id"].astype(str).tolist())
    inter = sorted(list(e2e_set.intersection(p14_set)))
    with open(os.path.join(OUT_DIR, "_index.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(inter), "ids": inter[:200]}, f, ensure_ascii=False, indent=2)

    sid = choose_id(e2e["id"], p14["id"], SAMPLE_ID)

    print(f"Preferred SAMPLE_ID = {SAMPLE_ID}")
    print(f"Chosen ID         = {sid}")
    print(f"Intersection size = {len(inter)}")
    if SAMPLE_ID is not None and SAMPLE_ID not in inter:
        print("⚠️ Preferred ID not found in BOTH files. Auto-fallback to an existing intersection id.")
        print("   Example ids:", inter[:5])

    row_e2e = e2e[e2e["id"].astype(str) == sid].iloc[0]
    row_p14 = p14[p14["id"].astype(str) == sid].iloc[0]

    # -------- P11 -> color_features --------
    # Prefer GT columns (your e2e_test_all.csv includes them)
    # -------- P11 -> color_features (use GT avg channels + derived stable metrics) --------
    r_mean = pick_first_existing(row_e2e, ["p11_gt_tg_Red_avg", "p11_gt_tg_Red_mid"])
    g_mean = pick_first_existing(row_e2e, ["p11_gt_tg_Green_avg", "p11_gt_tg_Green_mid"])
    b_mean = pick_first_existing(row_e2e, ["p11_gt_tg_Blue_avg", "p11_gt_tg_Blue_mid"])
    l_mean = pick_first_existing(row_e2e, ["p11_gt_tg_L_avg", "p11_gt_tg_L_mid"])
    gray_mean = pick_first_existing(row_e2e, ["p11_gt_tg_gray_avg", "p11_gt_tg_gray_mid"])

    # last fallback (rare): use pred dims if GT not present
    if r_mean is None:
        r_mean = pick_first_existing(row_e2e, ["p11_pred_00", "p11_pred_01"])
    if g_mean is None:
        g_mean = pick_first_existing(row_e2e, ["p11_pred_04", "p11_pred_05"])
    if b_mean is None:
        b_mean = pick_first_existing(row_e2e, ["p11_pred_08", "p11_pred_09"])
    if l_mean is None:
        l_mean = pick_first_existing(row_e2e, ["p11_pred_14", "p11_pred_15"])
    if gray_mean is None:
        gray_mean = pick_first_existing(row_e2e, ["p11_pred_72", "p11_pred_73"])

    eps = 1e-6
    rg_diff = (r_mean - g_mean) if (r_mean is not None and g_mean is not None) else None
    rb_diff = (r_mean - b_mean) if (r_mean is not None and b_mean is not None) else None
    r_over_g = (r_mean / (g_mean + eps)) if (r_mean is not None and g_mean is not None) else None
    r_over_b = (r_mean / (b_mean + eps)) if (r_mean is not None and b_mean is not None) else None

    color_features = {
        "dim": 76,
        "key_features": {
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
            "lightness_mean": l_mean,
            "gray_mean": gray_mean,
            "rg_diff": rg_diff,
            "rb_diff": rb_diff,
            "r_over_g": r_over_g,
            "r_over_b": r_over_b
        }
    }

    if INCLUDE_FULL_VECTORS:
        full = {}
        for i in range(76):
            k = f"p11_pred_{i:02d}"
            if k in row_e2e:
                full[f"c{i:02d}"] = float(row_e2e[k])
        color_features["full_vector"] = full

    # -------- P12 -> shape_features --------
    shape_features = {
        "area_px": float(row_e2e.get("p12_num_tg", 0.0)),
        "width_px": float(row_e2e.get("p12_tg_width", 0.0)),
        "height_px": float(row_e2e.get("p12_tg_height", 0.0)),
        "width_height_ratio": float(row_e2e.get("p12_tg_w_div_h", 0.0)),
        "source": "mask_original_size"
    }

    # -------- P13 -> texture_features --------
    homogeneity = pick_first_existing(row_e2e, ["p13_gt_tg_homogeneity"])
    energy = pick_first_existing(row_e2e, ["p13_gt_tg_energy"])
    contrast = pick_first_existing(row_e2e, ["p13_gt_tg_contrast"])
    entropy = pick_first_existing(row_e2e, ["p13_gt_tg_entropy"])

    # fallback to pred dims if GT absent
    if homogeneity is None:
        homogeneity = pick_first_existing(row_e2e, ["p13_pred_12"])
    if energy is None:
        energy = pick_first_existing(row_e2e, ["p13_pred_13"])
    if contrast is None:
        contrast = pick_first_existing(row_e2e, ["p13_pred_10"])
    if entropy is None:
        entropy = pick_first_existing(row_e2e, ["p13_pred_15"])

    texture_features = {
        "dim": 16,
        "key_features": {
            "homogeneity": homogeneity,
            "energy": energy,
            "contrast": contrast,
            "entropy": entropy
        }
    }

    if INCLUDE_FULL_VECTORS:
        full = {}
        for i in range(16):
            k = f"p13_pred_{i:02d}"
            if k in row_e2e:
                full[f"t{i:02d}"] = float(row_e2e[k])
        texture_features["full_vector"] = full

    # -------- P14 -> representation --------
    emb_cols = [c for c in p14.columns if c.startswith("p14_emb_")]
    emb_cols_sorted = sorted(emb_cols, key=lambda x: int(x.split("_")[-1]))

    preview = {}
    for i, c in enumerate(emb_cols_sorted[:EMB_PREVIEW_K]):
        preview[f"e{i:03d}"] = float(row_p14[c])

    representation = {
        "embedding_dim": len(emb_cols_sorted),
        "embedding_preview": preview,
        "intended_usage": ["similarity_search", "clustering", "downstream_classification"]
    }

    if INCLUDE_FULL_VECTORS:
        emb_full = {}
        for i, c in enumerate(emb_cols_sorted):
            emb_full[f"e{i:03d}"] = float(row_p14[c])
        representation["embedding"] = emb_full

    interpretation = make_interpretation(color_features, texture_features, shape_features)

    payload = {
        "meta": {
            "id": sid,
            "model_version": "tongue_expert_v1",
            "pipeline": ["p11_color", "p12_shape", "p13_texture", "p14_embedding"],
            "device": "cpu"
        },
        "artifacts": make_artifacts(sid),
        "outputs": {
            "color_features": color_features,
            "shape_features": shape_features,
            "texture_features": texture_features,
            "representation": representation
        },
        "interpretation": interpretation
    }

    out_path = os.path.join(OUT_DIR, f"{sid}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("✅ JSON saved:", os.path.abspath(out_path))
    print("\n--- Preview ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
