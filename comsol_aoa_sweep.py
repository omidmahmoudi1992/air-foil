# -*- coding: utf-8 -*-
"""
comsol_aoa_sweep.py
-------------------
Sweep زاویه‌ی حمله برای ایرفویل در COMSOL از طریق پایتون (کتابخانه‌ی MPh).

چه کاری می‌کند؟
    1) به کامسول وصل می‌شود (سرور محلی یا سرور در حال اجرا).
    2) برای هر زاویه‌ی حمله، مطالعه‌ی پایا را حل می‌کند (ادامه‌دهی خودکار از جواب قبلی).
    3) نیروی برآ (و در صورت وجود، نیروی پسا) را از Probe می‌خواند و Cl و Cd را حساب می‌کند.
    4) (اختیاری) برای هر زاویه حل زمان‌مند URANS می‌زند و میانگین برآ را گزارش می‌کند.
    5) خروجی: aoa_sweep_results.csv + نمودار Cl-α + نمودار قطبی Cl-Cd.

پیش‌نیازهای مدل .mph
--------------------
- پارامترهای Global:  c=1[m]، Re=5e5، rho، mu، aoa، U0، u0، v0   (نگاه کن به راهنما، بخش ۴)
- پروب Global Variable Probe با نام «Lift»:
      -int_air(spf.T_stressy*cos(aoa) - spf.T_stressx*sin(aoa))
- (اختیاری) پروب دوم با نام «Drag» برای نیروی پسا:
      -int_air(spf.T_stressx*cos(aoa) + spf.T_stressy*sin(aoa))
- مطالعه‌ی پایا با نام «Study 1» و (اختیاری) زمان‌مند با نام «Study 2».

اجرا
----
    python comsol_aoa_sweep.py --model airfoil_horizontal.mph --aoa 0:5:25
    python comsol_aoa_sweep.py --aoa 0,2,4,6,8,10,12,15,18,21,25 --hmax 0.02 --urans
    python comsol_aoa_sweep.py --port 2036          # اتصال به سرور در حال اجرا
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mph

# ----------------------------- تنظیمات -----------------------------
STUDY = "Study 1"
STUDY_URANS = "Study 2"
PROBE_LIFT = "Lift"
PROBE_DRAG = "Drag"
HMAX_PARAM = "hmax"

RESULTS_CSV = "aoa_sweep_results.csv"
PLOT_CL_ALPHA = "aoa_sweep_Cl_alpha.png"
PLOT_POLAR = "aoa_sweep_polar.png"


# ----------------------------- ابزارها -----------------------------
def scalar(value) -> float:
    """اولین عدد را از خروجی (احتمالاً آرایه‌ای) MPh بیرون می‌کشد."""
    return float(np.asarray(value, dtype=float).ravel()[0])


def parse_alphas(text: str) -> list[float]:
    """«0,2,4» یا «0:5:25» را به لیست زاویه‌ها تبدیل می‌کند."""
    text = text.strip()
    if ":" in text:
        nums = [float(x) for x in text.split(":")]
        if len(nums) == 2:
            start, stop = nums
            step = 5.0
        else:
            start, step, stop = nums
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(n)]
    return [float(x) for x in text.replace(" ", "").split(",")]


def numeric_param(model, name: str, default: float | None = None) -> float | None:
    """مقدار عددی یک پارامتر Global را از رشته‌ی کامل آن بیرون می‌کشد (مثلاً '7.294 [m/s]')."""
    try:
        raw = str(model.parameter(name))
    except Exception:                     # noqa: BLE001
        return default
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
    return float(m.group()) if m else default


def safe_evaluate(model, expr: str, dataset: str | None = None):
    """اگر عبارت/پروب وجود نداشت، None برمی‌گرداند به‌جای خطا."""
    try:
        if dataset:
            return np.asarray(model.evaluate(expr, dataset=dataset)).ravel()
        return scalar(model.evaluate(expr))
    except Exception:                     # noqa: BLE001
        return None


# ----------------------------- اتصال -----------------------------
def connect(cores: int = 4, port: int | None = None):
    if port:
        print(f"[اتصال] به سرور COMSOL روی پورت {port} ...")
        client = mph.Client(port=port)
    else:
        print(f"[اتصال] سرور COMSOL با {cores} هسته بالا می‌آید ...")
        client = mph.start(cores=cores)
    print(f"[اتصال] آماده. مدل‌های باز: {client.names()}")
    return client


# ----------------------------- حل یک زاویه -----------------------------
def solve_alpha(model, alpha: float, urans: bool, urans_tail: float):
    """زاویه را ست می‌کند، حل پایا (و در صورت درخواست URANS) می‌زند و نتایج را برمی‌گرداند."""
    model.parameter("aoa", f"{alpha} [deg]")

    row = {"aoa": alpha, "solve_time_s": np.nan, "lift_N_per_m": np.nan,
           "drag_N_per_m": np.nan, "cl": np.nan, "cd": np.nan,
           "cl_urans_mean": np.nan, "cl_urans_std": np.nan, "note": ""}

    # --- پایا ---
    try:
        t0 = time.perf_counter()
        model.solve(STUDY)                     # ادامه‌دهی خودکار از جواب زاویه‌ی قبلی
        row["solve_time_s"] = time.perf_counter() - t0
    except Exception as exc:                   # noqa: BLE001
        row["note"] = f"stationary failed: {exc}"[:120]
        return row

    lift = safe_evaluate(model, PROBE_LIFT)
    drag = safe_evaluate(model, PROBE_DRAG)
    row["lift_N_per_m"] = np.nan if lift is None else lift
    row["drag_N_per_m"] = np.nan if drag is None else drag

    c = numeric_param(model, "c", 1.0)
    rho = numeric_param(model, "rho", 1.225)
    u0 = numeric_param(model, "U0")
    if u0 is None:                             # اگر U0 نبود، از Re حساب کن
        re_, mu = numeric_param(model, "Re", 5e5), numeric_param(model, "mu", 1.787e-5)
        u0 = re_ * mu / (rho * c)
    q = 0.5 * rho * u0**2                      # فشار دینامیکی
    if lift is not None:
        row["cl"] = lift / (q * c)
    if drag is not None:
        row["cd"] = drag / (q * c)

    # --- زمان‌مند (اختیاری) ---
    if urans:
        try:
            t1 = time.perf_counter()
            model.solve(STUDY_URANS)
            series = safe_evaluate(model, PROBE_LIFT, dataset=f"{STUDY_URANS}/Solution 2")
            if series is not None and series.size:
                n = max(10, int(series.size * urans_tail))   # مثلاً ۴۰٪ آخر
                tail = series[-n:]
                row["cl_urans_mean"] = float(np.mean(tail) / (q * c))
                row["cl_urans_std"] = float(np.std(tail) / (q * c))
            row["note"] += f" | urans {time.perf_counter() - t1:.0f}s"
        except Exception as exc:               # noqa: BLE001
            row["note"] += f" | urans failed: {exc}"[:80]

    return row


# ----------------------------- نمودارها -----------------------------
def make_plots(rows: list[dict], outdir: Path, demo: bool = False):
    """نمودار Cl-α و نمودار قطبی Cl-Cd را می‌سازد."""
    aoa = np.array([r["aoa"] for r in rows], dtype=float)
    cl = np.array([r["cl"] for r in rows], dtype=float)
    cd = np.array([r["cd"] for r in rows], dtype=float)
    cl_u = np.array([r["cl_urans_mean"] for r in rows], dtype=float)

    ok = ~np.isnan(cl)

    # ---------- Cl-α ----------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(aoa[ok], cl[ok], "o-", color="#12376b", lw=2, ms=6, label="RANS (steady)")
    if np.any(~np.isnan(cl_u)):
        ax.plot(aoa[~np.isnan(cl_u)], cl_u[~np.isnan(cl_u)], "s--",
                color="#c0392b", lw=1.8, ms=6, label="URANS (time-averaged)")
    if np.any(ok):
        i_max = int(np.nanargmax(cl))
        ax.annotate(f"Cl,max = {cl[i_max]:.2f}\n@ α = {aoa[i_max]:.0f}°",
                    xy=(aoa[i_max], cl[i_max]),
                    xytext=(aoa[i_max] - 17, cl[i_max] - 0.28),
                    fontsize=10, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Angle of attack  α  [deg]", fontsize=12)
    ax.set_ylabel("Lift coefficient  $C_L$", fontsize=12)
    ax.set_title("NACA0012 — lift curve (Re = 5×10$^5$)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower right")
    if demo:
        ax.text(0.5, 0.5, "DEMO — synthetic data", fontsize=26, color="red", alpha=0.18,
                ha="center", va="center", rotation=20, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(outdir / PLOT_CL_ALPHA, dpi=160, facecolor="white")
    plt.close(fig)

    # ---------- قطب Cl-Cd ----------
    if np.any(~np.isnan(cd)):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(cd[~np.isnan(cd)], cl[~np.isnan(cd)], "o-", color="#1f6f4a", lw=2, ms=6)
        for a, x, y in zip(aoa, cd, cl):
            if not np.isnan(x) and not np.isnan(y):
                ax.annotate(f"{a:.0f}°", (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=9, color="0.35")
        ax.set_xlabel("Drag coefficient  $C_D$", fontsize=12)
        ax.set_ylabel("Lift coefficient  $C_L$", fontsize=12)
        ax.set_title("Drag polar — NACA0012", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, ls=":")
        if demo:
            ax.text(0.5, 0.5, "DEMO", fontsize=26, color="red", alpha=0.18,
                    ha="center", va="center", rotation=20, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(outdir / PLOT_POLAR, dpi=160, facecolor="white")
        plt.close(fig)


def write_csv(rows: list[dict], outdir: Path):
    path = outdir / RESULTS_CSV
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Sweep زاویه‌ی حمله برای ایرفویل در COMSOL")
    ap.add_argument("--model", default="airfoil_horizontal.mph")
    ap.add_argument("--aoa", default="0:5:25",
                    help="زاویه‌ها: «0:5:25» یا «0,2,4,6,...,25»")
    ap.add_argument("--hmax", type=float, default=None,
                    help="اندازه‌ی المان [m]؛ اگر بدهی، قبل از sweep یک‌بار مش را می‌سازد")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--urans", action="store_true", help="مطالعه‌ی زمان‌مند هم اجرا شود")
    ap.add_argument("--urans-tail", type=float, default=0.4,
                    help="چند درصد آخر سری زمانی برای میانگین‌گیری (پیش‌فرض 0.4)")
    ap.add_argument("--out", default=".", help="پوشه‌ی خروجی")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    alphas = parse_alphas(args.aoa)
    print(f"[لیست] زاویه‌ها: {alphas}")

    client = connect(cores=args.cores, port=args.port)
    model = client.load(args.model)

    if args.hmax is not None:
        model.parameter(HMAX_PARAM, f"{args.hmax:.6f} [m]")
        print(f"[مش] ساخت مش با hmax = {args.hmax} m ...")
        model.mesh()

    rows = []
    print("\n" + "=" * 78)
    print(f"{'α [deg]':>8} | {'t [s]':>7} | {'Cl':>8} | {'Cd':>8} | {'Cl(URANS)':>10} | یادداشت")
    print("=" * 78)

    for alpha in alphas:
        row = solve_alpha(model, alpha, args.urans, args.urans_tail)
        rows.append(row)
        print(f"{row['aoa']:>8.1f} | {row['solve_time_s']:>7.1f} | "
              f"{row['cl']:>8.3f} | {row['cd']:>8.4f} | {row['cl_urans_mean']:>10.3f} | "
              f"{row['note']}")
        write_csv(rows, outdir)                 # ذخیره پس از هر زاویه (ضدکرش)

    csv_path = write_csv(rows, outdir)
    make_plots(rows, outdir)

    cl_max = max((r["cl"] for r in rows if not np.isnan(r["cl"])), default=float("nan"))
    print("=" * 78)
    print(f"[پایان] {len(rows)} زاویه حل شد. بیشترین Cl = {cl_max:.3f}")
    print(f"[خروجی] {csv_path}")
    print(f"[خروجی] {outdir / PLOT_CL_ALPHA}")
    if any(not np.isnan(r["cd"]) for r in rows):
        print(f"[خروجی] {outdir / PLOT_POLAR}")

    model.save(str(Path(args.model).with_name(Path(args.model).stem + "_sweep.mph")))


if __name__ == "__main__":
    main()
