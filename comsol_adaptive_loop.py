# -*- coding: utf-8 -*-
"""
comsol_adaptive_loop.py
-----------------------
رهبری حل ایرفویل در COMSOL از طریق پایتون + حلقه‌ی تطبیقی مش با منطق «زمان‌محور».

این همان کاری است که در GUI کامسول وجود ندارد:
    حل کن -> زمان حل و مقدار برآ را بسنج -> اگر حل کند بود مش را ریز کن،
    اگر از بودجه‌ی زمانی گذشت مش را درشت کن -> دوباره حل کن -> تا همگرایی.

نیازمندی‌ها
-----------
1) COMSOL Multiphysics 5.6 یا جدیدتر، نصب‌شده روی همین ماشین با لایسنس معتبر.
2) pip install MPh
3) مدل .mph با این پیش‌نیازها:
   - پارامتر Global به نام «hmax» (اندازه‌ی المان دامنه) مثلا:  hmax = 0.05 [m]
   - در Mesh -> Size -> Custom، فیلد element size برابر با عبارت hmax گذاشته شده باشد
     (و اگر نود Size ناحیه‌ای داری، همان را هم روی hmax بگذار).
   - فیزیک Turbulent Flow, k-omega SST + مطالعه‌ی Stationary با نام «Study 1».
   - یک Global Variable Probe به نام «Lift»:
        -int_air(spf.T_stressy*cos(aoa) - spf.T_stressx*sin(aoa))
     که int_air یک Integration Coupling روی مرز سطح ایرفویل است.
4) (اختیاری) مطالعه‌ی زمان‌مند با نام «Study 2» برای مرحله‌ی URANS.

اجرا
----
    python comsol_adaptive_loop.py                 # حلقه‌ی تطبیقی روی Study 1
    python comsol_adaptive_loop.py --port 2036     # اتصال به سرور در حال اجرا
    python comsol_adaptive_loop.py --urans         # بعد از همگرایی، زمان‌مند هم حل کن
    python comsol_adaptive_loop.py --aoa 25 --aoa-step 5   # از زاویه‌ی فعلی تا 25 درجه، پله‌پله
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time

import numpy as np
import mph


def _scalar(value) -> float:
    """اولین مقدار عددی را از خروجی آرایه‌ای MPh بیرون می‌کشد."""
    return float(np.asarray(value, dtype=float).ravel()[0])

# ----------------------------- تنظیمات -----------------------------
MODEL_PATH = "airfoil_horizontal.mph"   # مسیر مدل .mph که در GUI ساختـی
STUDY = "Study 1"                       # نام مطالعه‌ی پایا
STUDY_URANS = "Study 2"                 # نام مطالعه‌ی زمان‌مند (اختیاری)
PROBE = "Lift"                          # نام Global Variable Probe
HMAX_PARAM = "hmax"                     # نام پارامتر اندازه‌ی المان

HMAX_START = 0.050      # [m] اندازه‌ی المان شروع (دست‌ودل‌باز: ~5% وتر)
HMAX_MIN = 0.008        # [m] ریزترین حالتی که اجازه می‌دهیم
HMAX_MAX = 0.150        # [m] درشت‌ترین حالت
TOL = 0.02              # همگرایی: تغییر نسبی برآ بین دو تکرار
TIME_BUDGET = 900.0     # [s] سقف زمانی هر حل (برای منطق زمان‌محور)
MAX_ITER = 8            # حداکثر تعداد تکرار (ریز/درشت کردن)
LOG_CSV = "adaptive_log.csv"


# ----------------------------- اتصال به COMSOL -----------------------------
def connect(cores: int = 4, port: int | None = None):
    """اتصال به COMSOL؛ یا سرور محلی را بالا می‌آورد یا به سرور در حال اجرا وصل می‌شود."""
    if port:
        print(f"[اتصال] به سرور COMSOL روی پورت {port} وصل می‌شوم ...")
        client = mph.Client(port=port)          # comsol mphserver -port 2036
    else:
        print(f"[اتصال] سرور COMSOL را با {cores} هسته بالا می‌آورم ...")
        client = mph.start(cores=cores)         # سرور محلی + اتصال
    print(f"[اتصال] آماده. مدل‌های باز: {client.names()}")
    return client


# ----------------------------- گام حل -----------------------------
def mesh_and_solve(model, hmax: float):
    """مش را با اندازه‌ی داده‌شده می‌سازد، مطالعه‌ی پایا را حل می‌کند و (زمان، برآ) برمی‌گرداند."""
    model.parameter(HMAX_PARAM, f"{hmax:.6f} [m]")
    t0 = time.perf_counter()
    model.mesh()                                 # اجرای توالی مش
    model.solve(STUDY)                           # حل مطالعه‌ی پایا
    elapsed = time.perf_counter() - t0
    lift = _scalar(model.evaluate(PROBE))        # نیروی برآ از پروب
    return elapsed, lift


def try_solve_safely(model, hmax: float):
    """اگر حل واگرا/خطا داد، یک پله درشت‌تر تلاش می‌کند."""
    try:
        return mesh_and_solve(model, hmax)
    except Exception as exc:                     # noqa: BLE001
        print(f"[هشدار] حل با hmax={hmax:.4f} شکست خورد: {exc}")
        return None, None


# ----------------------------- حلقه‌ی تطبیقی -----------------------------
def adaptive_loop(model, hmax: float) -> tuple[float, float]:
    """حلقه‌ی زمان‌محور: تا همگرایی برآ یا اتمام بودجه‌ی تکرار."""
    rows, prev_lift, best = [], None, (hmax, None)

    print("\n" + "=" * 74)
    print(f"{'تکرار':>6} | {'hmax [m]':>9} | {'زمان حل [s]':>11} | {'Lift [N/m]':>11} | تصمیم")
    print("=" * 74)

    for it in range(1, MAX_ITER + 1):
        elapsed, lift = try_solve_safely(model, hmax)

        if lift is None:                          # حل شکست خورد -> درشت‌تر کن
            hmax = min(hmax * 1.3, HMAX_MAX)
            rows.append([it, f"{hmax:.5f}", "fail", "", "coarsen (solve failed)"])
            _write_log(rows)
            continue

        rel = "" if prev_lift is None else f"{abs(lift - prev_lift) / abs(prev_lift):.3%}"
        decision = "-"

        if prev_lift is not None and abs(lift - prev_lift) / abs(prev_lift) < TOL:
            decision = "converged"
        elif elapsed > TIME_BUDGET:
            hmax = min(hmax * 1.4, HMAX_MAX)      # خیلی کند بود -> درشت‌تر
            decision = "coarsen (over time budget)"
        elif hmax > HMAX_MIN:
            hmax = max(hmax * 0.7, HMAX_MIN)      # سریع و هنوز همگرا نشده -> ریزتر
            decision = "refine"
        else:
            decision = "stop (hmax at minimum)"

        rows.append([it, f"{hmax:.5f}", f"{elapsed:.1f}", f"{lift:.4f}", decision])
        _write_log(rows)
        print(f"{it:>6} | {hmax:>9.5f} | {elapsed:>11.1f} | {lift:>11.4f} | {decision}"
              + (f"   (Δ={rel})" if rel else ""))

        if lift is not None and (best[1] is None or hmax <= best[0]):
            best = (hmax, lift)

        if decision.startswith("converged") or decision.startswith("stop"):
            print("=" * 74)
            print(f"[پایان] hmax نهایی = {hmax:.5f} m ، Lift = {lift:.4f} N/m")
            return hmax, lift

        prev_lift = lift

    print("=" * 74)
    print("[پایان] سقف تکرار پر شد. بهترین حالت:", best)
    return best


def _write_log(rows):
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "hmax [m]", "solve_time [s]", "Lift [N/m]", "decision"])
        w.writerows(rows)


# ----------------------------- مرحله‌ی زمان‌مند (URANS) -----------------------------
def run_urans(model, n_samples: int = 200):
    """حل زمان‌مند با همان مشِ نهایی و گزارش میانگین برآ (برای مقایسه با تصویر مرجع)."""
    print("\n[URANS] حل زمان‌مند روی مشِ نهاییِ تطبیقی ...")
    t0 = time.perf_counter()
    model.solve(STUDY_URANS)
    print(f"[URANS] زمان حل: {time.perf_counter() - t0:.1f} s")

    # نمونه‌برداری از برآ در طول زمان (پیش‌فرض: کل بازه‌ی حل)
    t = np.asarray(model.evaluate("t", "s", dataset="Study 2/Solution 2")).ravel()
    lift = np.asarray(model.evaluate(PROBE, dataset="Study 2/Solution 2")).ravel()
    n = min(len(lift), n_samples)
    lift_tail = lift[-n:]                        # N نمونه‌ی آخر = بعد از حالت گذرا

    print(f"[URANS] میانگین Lift (بازه‌ی توسعه‌یافته): "
          f"{statistics.fmean(lift_tail):.4f} N/m   |   انحراف معیار: "
          f"{statistics.pstdev(lift_tail):.4f} N/m")
    return t, lift


# ----------------------------- زاویه‌ی حمله پله‌ای -----------------------------
def ramp_aoa(model, aoa_target: float, step: float = 5.0):
    """زوایه‌ی حمله را پله‌پله بالا می‌برد (ادامه‌دهی از جواب قبلی) تا همگرایی بهتر."""
    aoa = float(model.parameter("aoa"))
    while aoa < aoa_target:
        aoa = min(aoa + step, aoa_target)
        model.parameter("aoa", f"{aoa} [deg]")
        print(f"\n[AOA] زاویه‌ی حمله -> {aoa} درجه")
        mesh_and_solve(model, float(model.parameter(HMAX_PARAM).split()[0]))
    return aoa


# ----------------------------- main -----------------------------
def main():
    global TOL, TIME_BUDGET

    ap = argparse.ArgumentParser(description="حلقه‌ی تطبیقی مش برای ایرفویل در COMSOL")
    ap.add_argument("--model", default=MODEL_PATH, help="مسیر فایل .mph")
    ap.add_argument("--cores", type=int, default=4, help="تعداد هسته‌های سرور COMSOL")
    ap.add_argument("--port", type=int, default=None, help="اتصال به سرور در حال اجرا (مثلا 2036)")
    ap.add_argument("--hmax", type=float, default=HMAX_START, help="اندازه‌ی المان شروع [m]")
    ap.add_argument("--tol", type=float, default=TOL, help="تلورانس همگرایی برآ")
    ap.add_argument("--budget", type=float, default=TIME_BUDGET, help="بودجه‌ی زمانی هر حل [s]")
    ap.add_argument("--aoa", type=float, default=None, help="زاویه‌ی هدف [deg]")
    ap.add_argument("--aoa-step", type=float, default=5.0, help="پله‌ی تغییر زاویه [deg]")
    ap.add_argument("--urans", action="store_true", help="بعد از همگرایی، حل زمان‌مند هم بزن")
    args = ap.parse_args()

    TOL, TIME_BUDGET = args.tol, args.budget

    client = connect(cores=args.cores, port=args.port)
    model = client.load(args.model)
    print(f"[مدل] {args.model} بارگذاری شد. (Study ها: {model/'studies'})")

    if args.aoa is not None:
        ramp_aoa(model, args.aoa, args.aoa_step)

    hmax_final, lift_final = adaptive_loop(model, args.hmax)

    model.save(args.model.replace(".mph", "_adapted.mph"))
    print(f"[ذخیره] مدل نهایی در {args.model.replace('.mph', '_adapted.mph')}")

    if args.urans:
        run_urans(model)
        model.save(args.model.replace(".mph", "_adapted_urans.mph"))


if __name__ == "__main__":
    main()
