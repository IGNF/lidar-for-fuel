"""Story 3.8 — mesure rapide des écarts de l'étape 3 sur la chaîne complète (script ponctuel).

Applique l'étape 3 (prototype `ffm2`, portage littéral de ffuelmetrics2) au PAD produit par
lidar-for-fuel, puis compare les métriques aux `fuel_metrics.tif` de la chaîne R complète.

Mode A (contrôle) : prototype appliqué au PAD R, sur la même fenêtre → écart attendu nul.
Mode B : prototype appliqué au PAD Python → écart de la chaîne complète (= écart dû au PAD puisque A = 0).

Usage :
    python mesure_rapide_etape3.py --proto PROTO.py --r-dir R/10_10_2026/ \
        --py-pad NEW_pad_profile_1m.tif [--py-pad-old OLD_pad_profile_1m.tif] \
        --out-dir DIR [--save-tif FILE]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Groupes de la note INRAE (Q6)
HEIGHTS = ["CBH2", "CBH", "CBHlow", "CTH_0p0012", "CTH_0p0100", "CDH", "FSG"]
LOADS = ["CFL", "LFL", "CLFL", "FL_k1", "FL_k2"]
DENSITIES = ["CBDcan", "CanBDmax"]
CLASSES = ["ProfType"]
KEY = HEIGHTS + LOADS + DENSITIES + CLASSES
# Tolérances candidates (à discuter avec l'INRAE)
TOL_HEIGHT_M = [0.0, 1.0, 2.0]  # 0 = identique ; 1 m = une strate
TOL_LOAD_REL = [0.05, 0.10]  # relatif, avec un plancher absolu
LOAD_FLOOR = 0.01  # kg/m²
DENS_FLOOR = 0.001  # kg/m³


def load_proto(path):
    spec = importlib.util.spec_from_file_location("proto_ffm2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ffm2


def read_pad(path, window=None):
    with rasterio.open(path) as s:
        names = s.descriptions
        idx = [i for i, n in enumerate(names) if n and n.startswith("PAD_1_")]
        pad = s.read([i + 1 for i in idx], window=window).astype(np.float64)
        transform = s.window_transform(window) if window is not None else s.transform
        crs = s.crs
    zb = np.array([float(names[i].split("_")[2]) for i in idx])
    order = np.argsort(zb)
    return pad[order], zb[order], transform, crs


def apply_step3(ffm2, pad, zb):
    nb, h, w = pad.shape
    first = ffm2(pad[:, 0, 0], zb, 1.0)
    names = list(first.keys())
    out = np.full((len(names), h, w), np.nan)
    for r in range(h):
        for c in range(w):
            o = ffm2(pad[:, r, c], zb, 1.0)
            out[:, r, c] = [o[n] for n in names]
    return names, out


def band_stats(name, a, e):
    """a = Python, e = R (aplatis)."""
    va, ve = np.isfinite(a), np.isfinite(e)
    both = va & ve
    d = a[both] - e[both]
    ad = np.abs(d)
    st = {
        "metric": name,
        "n_both": int(both.sum()),
        "n_py_only": int((va & ~ve).sum()),
        "n_r_only": int((~va & ve).sum()),
        "r_mean": float(np.mean(e[both])) if both.any() else np.nan,
        "bias": float(np.mean(d)) if both.any() else np.nan,
        "mae": float(np.mean(ad)) if both.any() else np.nan,
        "p99": float(np.percentile(ad, 99)) if both.any() else np.nan,
        "max": float(ad.max()) if both.any() else np.nan,
        "exact": float(np.mean(ad <= 1e-4 * np.maximum(1, np.abs(e[both])))) if both.any() else np.nan,
    }
    if not both.any():
        return st
    base = name.split("_0p")[0]
    if name in HEIGHTS or base in ("LadderBot", "LadderTop", "FSG"):
        for t in TOL_HEIGHT_M:
            st[f"in_{t:g}m"] = float(np.mean(ad <= t + 1e-6))
    elif name in LOADS or base in ("LadderLoad", "CFL", "SFLeq0", "SFLeq_intot"):
        for t in TOL_LOAD_REL:
            st[f"in_{int(t * 100)}pct"] = float(np.mean(ad <= np.maximum(t * np.abs(e[both]), LOAD_FLOOR)))
    elif name in DENSITIES or name.startswith(("CanBD", "CBD")):
        for t in TOL_LOAD_REL:
            st[f"in_{int(t * 100)}pct"] = float(np.mean(ad <= np.maximum(t * np.abs(e[both]), DENS_FLOOR)))
    return st


def confusion(a, e):
    m = np.isfinite(a) & np.isfinite(e)
    cm = np.zeros((4, 4), int)
    for i in range(1, 5):
        for j in range(1, 5):
            cm[i - 1, j - 1] = int(np.sum((e[m] == i) & (a[m] == j)))
    return cm


def compare(names, py, ref, ref_names):
    assert list(names) == list(ref_names), "ordre des bandes différent de R"
    return {n: band_stats(n, py[b].ravel(), ref[b].ravel()) for b, n in enumerate(names)}


def fmt(v, nd=3):
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v * 100:.1f} %" if nd == "pct" else f"{v:.{nd}f}"
    return str(v)


def md_table(stats, metrics, pct_cols):
    cols = ["n_both", "n_py_only", "n_r_only", "r_mean", "bias", "mae", "p99", "max", "exact"] + pct_cols
    head = "| Métrique | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = []
    for m in metrics:
        s = stats[m]
        cells = []
        for c in cols:
            v = s.get(c, np.nan)
            cells.append(fmt(v, "pct") if (c == "exact" or c.startswith("in_")) else fmt(v))
        rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", required=True)
    ap.add_argument("--r-dir", required=True)
    ap.add_argument("--py-pad", required=True)
    ap.add_argument("--py-pad-old")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-tif")
    args = ap.parse_args()
    ffm2 = load_proto(args.proto)
    r_dir = Path(args.r_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pad_py, zb_py, tr_py, crs = read_pad(args.py_pad)
    _, h, w = pad_py.shape
    with rasterio.open(r_dir / "fuel_metrics.tif") as s:
        tr_r = s.transform
        col = (tr_py.c - tr_r.c) / tr_r.a
        row = (tr_py.f - tr_r.f) / tr_r.e
        assert abs(col - round(col)) < 1e-6 and abs(row - round(row)) < 1e-6, "grilles non alignées"
        win = Window(int(round(col)), int(round(row)), w, h)
        ref = s.read(window=win).astype(np.float64)
        ref_names = list(s.descriptions)
    pad_r, zb_r, _, _ = read_pad(r_dir / "pad_1m.tif", window=win)

    res = {"window_in_R": [win.col_off, win.row_off, w, h], "n_pixels": h * w}
    res["pad_all_nan"] = {
        "python": int(np.all(~np.isfinite(pad_py), axis=0).sum()),
        "R": int(np.all(~np.isfinite(pad_r), axis=0).sum()),
    }
    # Mode A : contrôle de parité de l'étape 3 seule
    names, fa = apply_step3(ffm2, pad_r, zb_r)
    stats_a = compare(names, fa, ref, ref_names)
    res["mode_A_max_mismatch_share"] = max(1 - s["exact"] for s in stats_a.values() if s["n_both"])
    res["mode_A_validity_mismatch"] = sum(s["n_py_only"] + s["n_r_only"] for s in stats_a.values())
    # Mode B : chaîne complète
    names, fb = apply_step3(ffm2, pad_py, zb_py)
    stats_b = compare(names, fb, ref, ref_names)
    cm_b = confusion(fb[names.index("ProfType")].ravel(), ref[names.index("ProfType")].ravel())
    stats_old, cm_old = None, None
    if args.py_pad_old:
        pad_o, zb_o, tr_o, _ = read_pad(args.py_pad_old)
        assert tr_o == tr_py and pad_o.shape == pad_py.shape, "ancien PAD sur une autre grille"
        res["pad_all_nan"]["python_old"] = int(np.all(~np.isfinite(pad_o), axis=0).sum())
        names, fo = apply_step3(ffm2, pad_o, zb_o)
        stats_old = compare(names, fo, ref, ref_names)
        cm_old = confusion(fo[names.index("ProfType")].ravel(), ref[names.index("ProfType")].ravel())

    if args.save_tif:
        prof = dict(
            driver="GTiff",
            height=h,
            width=w,
            count=len(names),
            dtype="float32",
            crs=crs,
            transform=tr_py,
            nodata=np.nan,
            compress="deflate",
        )
        with rasterio.open(args.save_tif, "w", **prof) as d:
            d.write(fb.astype(np.float32))
            for i, n in enumerate(names, 1):
                d.set_band_description(i, n)

    (out_dir / "stats_mode_B.json").write_text(
        json.dumps(
            {
                "meta": res,
                "stats": stats_b,
                "confusion_ProfType_rows_R_cols_Py": cm_b.tolist(),
                "stats_old": stats_old,
                "confusion_old": None if cm_old is None else cm_old.tolist(),
            },
            indent=1,
            default=float,
        )
    )

    pct_h = [f"in_{t:g}m" for t in TOL_HEIGHT_M]
    pct_l = [f"in_{int(t * 100)}pct" for t in TOL_LOAD_REL]
    md = [f"# Résultats bruts — mesure rapide étape 3\n\n```json\n{json.dumps(res, indent=1, default=float)}\n```\n"]
    md.append("## Mode B (PAD Python → prototype) contre R — hauteurs (m)\n\n" + md_table(stats_b, HEIGHTS, pct_h))
    md.append("\n## Mode B — charges (kg/m²)\n\n" + md_table(stats_b, LOADS, pct_l))
    md.append("\n## Mode B — densités (kg/m³)\n\n" + md_table(stats_b, DENSITIES, pct_l))
    md.append("\n## Mode B — ProfType\n\n" + md_table(stats_b, CLASSES, []))
    md.append(
        "\nMatrice de confusion ProfType (lignes = R 1..4, colonnes = Python 1..4) :\n\n```\n"
        + "\n".join(" ".join(f"{v:6d}" for v in r) for r in cm_b)
        + "\n```\n"
    )
    if stats_old:
        md.append(
            "\n## Ancien PAD Python (avant correctifs v0.3.0) contre R — métriques clés\n\n"
            + md_table(stats_old, HEIGHTS, pct_h)
            + "\n"
            + md_table(stats_old, LOADS, pct_l)
            + "\n"
            + md_table(stats_old, DENSITIES + CLASSES, pct_l)
        )
        md.append(
            "\nMatrice ProfType (ancien PAD) :\n\n```\n"
            + "\n".join(" ".join(f"{v:6d}" for v in r) for r in cm_old)
            + "\n```\n"
        )
    md.append("\n## Mode B — toutes les bandes\n\n" + md_table(stats_b, names, []))
    (out_dir / "resultats_bruts.md").write_text("\n".join(md))
    print(json.dumps(res, indent=1, default=float))
    for m in KEY:
        s = stats_b[m]
        print(f"{m:12s} n={s['n_both']:6d} mae={s['mae']:.4f} p99={s['p99']:.4f} exact={s['exact'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
