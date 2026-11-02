# -*- coding: utf-8 -*-
"""Create four COMSOL setup files from the user's NACA0012 case.

Outputs (setup only; this script does NOT solve or generate a 3D mesh):
  airfoil_RANS_2D.mph          copied from the user's 2D base model
  airfoil_URANS_2D.mph         same 2D base + stationary-to-transient study
  airfoil_DES_3D_PILOT.mph     extruded 3D periodic-span pilot, DES RBVM
  airfoil_LES_3D_PILOT.mph     extruded 3D periodic-span pilot, LES RBVM

Requires COMSOL Multiphysics 6.3 + CFD Module, and MPh installed in the
same Windows Python environment that previously ran comsol_aoa_sweep.py.

The two 3D models are setup templates, not validated or mesh-converged LES/DES.
They deliberately are not meshed or solved; refine the mesh and time step before
using them for quantitative conclusions.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import mph
import jpype

DEFAULT_BASE = Path(r"C:\Users\ShemrounShop\Desktop\airfoil_positive_drag.mph")
DEFAULT_TLIST = "range(0,0.005,1.0052)"


def java_tags(java_list) -> list[str]:
    try:
        return [str(x) for x in java_list.tags()]
    except Exception:
        return []


def next_tag(java_list, prefix: str) -> str:
    existing = set(java_tags(java_list))
    i = 1
    while f"{prefix}{i}" in existing:
        i += 1
    return f"{prefix}{i}"


def save_2d_variants(client, base: Path, outdir: Path, tlist: str) -> None:
    rans_path = outdir / "airfoil_RANS_2D.mph"
    shutil.copy2(base, rans_path)
    print(f"[OK] RANS 2D copy: {rans_path}")

    model = client.load(str(base))
    try:
        jm = model.java
        study_tag = next_tag(jm.study(), "std")
        jm.study().create(study_tag)
        jm.study(study_tag).label("URANS 2D - transient k-omega")
        jm.study(study_tag).create("stat", "Stationary")
        jm.study(study_tag).create("time", "Transient")
        jm.study(study_tag).feature("time").set("tlist", tlist)
        model.save(str(outdir / "airfoil_URANS_2D.mph"))
        print(f"[OK] URANS 2D setup: {outdir / 'airfoil_URANS_2D.mph'}")
    finally:
        client.remove(model)


def naca0012_rotated_points(chord: float, aoa_deg: float, n: int) -> tuple[list[float], list[float]]:
    """Closed NACA 0012 polygon, rotated clockwise about quarter chord.

    The 3D build uses zero rotation to match the horizontal 2D airfoil; the
    angle of attack is represented by the inlet velocity vector.
    """
    theta = [math.pi * i / (n - 1) for i in range(n)]
    xs = [0.5 * (1.0 - math.cos(t)) for t in theta]

    def thickness(xc: float) -> float:
        # Closed trailing edge coefficient.
        return 5.0 * 0.12 * (
            0.2969 * math.sqrt(max(xc, 0.0))
            - 0.1260 * xc
            - 0.3516 * xc**2
            + 0.2843 * xc**3
            - 0.1036 * xc**4
        ) * chord

    upper = [(x * chord, thickness(x)) for x in xs]
    lower = [(x * chord, -thickness(x)) for x in reversed(xs[1:-1])]
    points = upper + lower

    ang = math.radians(-aoa_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    pivot_x, pivot_y = 0.25 * chord, 0.0
    xr, yr = [], []
    for x, y in points:
        dx, dy = x - pivot_x, y - pivot_y
        xr.append(pivot_x + ca * dx - sa * dy)
        yr.append(pivot_y + sa * dx + ca * dy)
    return xr, yr


def box_selection(comp, tag: str, *, xmin: float, xmax: float,
                  ymin: float, ymax: float, zmin: float, zmax: float):
    comp.selection().create(tag, "Box")
    sel = comp.selection(tag)
    sel.set("entitydim", jpype.JInt(2))
    # Avoid COMSOL's default "intersects" behavior, which also selects faces
    # that merely touch a thin coordinate slab; keep only faces inside the box.
    sel.set("condition", "inside")
    for key, value in (("xmin", xmin), ("xmax", xmax),
                       ("ymin", ymin), ("ymax", ymax),
                       ("zmin", zmin), ("zmax", zmax)):
        sel.set(key, float(value))
    return sel


def create_3d_pilot(client, outpath: Path, physics_candidates: list[str],
                    label: str, span: float, tlist: str, naca_points: int,
                    geometry_only: bool = False) -> None:
    model = client.create(label.replace(" ", "_"))
    jm = model.java
    try:
        # Shared parameters: c=1 m, Re=5e5, AoA=25 deg, air at the user's values.
        jm.param().set("c", "1[m]")
        jm.param().set("Re", "5e5")
        jm.param().set("aoa", "25[deg]")
        jm.param().set("rho", "1.225[kg/m^3]")
        jm.param().set("mu", "1.787e-5[Pa*s]")
        jm.param().set("U0", "Re*mu/(rho*c)")
        jm.param().set("Uinf", "U0")  # retained for the pending physics setup
        jm.param().set("u0", "U0*cos(aoa)")
        jm.param().set("v0", "U0*sin(aoa)")
        jm.param().set("w0", "0[m/s]")
        jm.param().set("Lspan", f"{span:.6g}[m]")

        jm.component().create("comp1", True)
        comp = jm.component("comp1")
        geom = comp.geom().create("geom1", 3)
        geom.lengthUnit("m")

        # Keep the NACA0012 chord horizontal, matching the user's 2D model.
        # The 25-degree flow direction is represented by u0/v0 parameters,
        # to be applied as a velocity-field inlet once the COMSOL API settings are verified.
        # Then extrude spanwise to make a periodic 3D unit cell.
        geom.feature().create("wp1", "WorkPlane")
        geom.feature("wp1").set("quickplane", "xy")
        g2 = geom.feature("wp1").geom()
        g2.feature().create("rect1", "Rectangle")
        g2.feature("rect1").set("pos", [-8.0, -10.0])
        g2.feature("rect1").set("size", [24.0, 20.0])

        xfoil, yfoil = naca0012_rotated_points(1.0, 0.0, naca_points)
        g2.feature().create("foil1", "Polygon")
        g2.feature("foil1").set("x", xfoil)
        g2.feature("foil1").set("y", yfoil)
        g2.feature("foil1").set("type", "solid")
        g2.feature().create("dif1", "Difference")
        g2.feature("dif1").selection("input").set(["rect1"])
        g2.feature("dif1").selection("input2").set(["foil1"])
        g2.run()

        geom.feature().create("ext1", "Extrude")
        geom.feature("ext1").set("extrudefrom", "workplane")
        geom.feature("ext1").set("workplane", "wp1")
        # Select the Difference result inside the work plane explicitly.
        geom.feature("ext1").selection("input").set(["wp1.dif1"])
        geom.feature("ext1").set("distance", float(span))
        geom.run()

        # Coordinate-based boundary selections; periodic pair is z=0 and z=Lspan.
        eps = 1e-5
        box_selection(comp, "selIn", xmin=-8-eps, xmax=-8+eps,
                      ymin=-10-eps, ymax=10+eps, zmin=-eps, zmax=span+eps)
        box_selection(comp, "selOut", xmin=16-eps, xmax=16+eps,
                      ymin=-10-eps, ymax=10+eps, zmin=-eps, zmax=span+eps)
        box_selection(comp, "selTop", xmin=-8-eps, xmax=16+eps,
                      ymin=10-eps, ymax=10+eps, zmin=-eps, zmax=span+eps)
        box_selection(comp, "selBottom", xmin=-8-eps, xmax=16+eps,
                      ymin=-10-eps, ymax=-10+eps, zmin=-eps, zmax=span+eps)
        box_selection(comp, "selFoil", xmin=-0.15, xmax=1.15,
                      ymin=-0.40, ymax=0.40, zmin=-eps, zmax=span+eps)
        z0sel = box_selection(comp, "selSpan0", xmin=-8-eps, xmax=16+eps,
                              ymin=-10-eps, ymax=10+eps, zmin=-eps, zmax=eps)
        z1sel = box_selection(comp, "selSpan1", xmin=-8-eps, xmax=16+eps,
                              ymin=-10-eps, ymax=10+eps,
                              zmin=span-eps, zmax=span+eps)

        # Air material.
        jm.material().create("mat1", "Common", "comp1")
        jm.material("mat1").label("Air (constant properties)")
        jm.material("mat1").propertyGroup("def").set("density", "rho")
        jm.material("mat1").propertyGroup("def").set("dynamicviscosity", "mu")
        jm.material("mat1").selection().all()

        # Save a recovery/template file before attempting version-specific physics IDs.
        geometry_base_path = outpath.parent / "airfoil_3D_geometry_base.mph"
        model.save(str(geometry_base_path))
        print(f"[OK] 3D geometry-only base: {geometry_base_path}")
        if geometry_only:
            return

        # Exact COMSOL 6.3 interface identifiers taken from the user's Model Java exports.
        last_error = None
        for ptype in physics_candidates:
            try:
                comp.physics().create("spf", ptype, "geom1")
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                try:
                    comp.physics().remove("spf")
                except Exception:  # noqa: BLE001
                    pass
        if last_error is not None:
            raise RuntimeError(
                f"COMSOL did not accept physics interface type {physics_candidates}. "
                f"A geometry-only recovery file was saved at {geometry_base_path}. "
                f"Last API message: {last_error}"
            )

        spf = comp.physics("spf")
        # Restrict the default wall to the extruded airfoil surface.
        try:
            spf.feature("wall1").selection().named("selFoil")
        except Exception:
            pass

        # Match the user's verified 2D inlet setup: componentwise velocity field.
        spf.feature().create("inl1", "InletBoundary", 2)
        spf.feature("inl1").selection().named("selIn")
        spf.feature("inl1").set("ComponentWise", "VelocityFieldComponentWise")
        spf.feature("inl1").set(
            "u0", jpype.JArray(jpype.JString)(["u0", "v0", "0"])
        )

        spf.feature().create("out1", "OutletBoundary", 2)
        spf.feature("out1").selection().named("selOut")

        spf.feature().create("openTop", "OpenBoundary", 2)
        spf.feature("openTop").selection().named("selTop")
        spf.feature().create("openBottom", "OpenBoundary", 2)
        spf.feature("openBottom").selection().named("selBottom")

        # The COMSOL Periodic Flow Condition uses both span faces as one pair.
        span_entities = []
        for sel in (z0sel, z1sel):
            try:
                span_entities.extend(int(e) for e in sel.entities(2))
            except Exception:
                pass
        span_entities = sorted(set(span_entities))
        if len(span_entities) != 2:
            raise RuntimeError(
                f"Expected the two span end faces (z=0 and z=Lspan), got entity IDs {span_entities}."
            )
        spf.create("pfc1", "PeriodicFlowCondition", 2)
        spf.feature("pfc1").selection().set(jpype.JArray(jpype.JInt)(span_entities))

        # Setup-only transient study: deliberately do not call mesh.run() or study.run().
        jm.study().create("std1")
        jm.study("std1").label(label + " - Time Dependent")
        jm.study("std1").create("time", "Transient")
        jm.study("std1").feature("time").set("tlist", tlist)

        comp.mesh().create("mesh1")
        try:
            comp.mesh("mesh1").autoMeshSize(6)
        except Exception:
            pass

        model.save(str(outpath))
        print(f"[OK] {label}: {outpath}")
    finally:
        client.remove(model)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create four NACA0012 COMSOL setup files (no solves).")
    parser.add_argument("--base", default=str(DEFAULT_BASE), help="Local 2D base MPH file")
    parser.add_argument("--out", default=None, help="Output folder; default: <base folder>/airfoil_four_models")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--span", type=float, default=0.2, help="3D periodic span in meters (default 0.2 m = 0.2c)")
    parser.add_argument("--naca-points", type=int, default=100, help="Points per airfoil side")
    parser.add_argument("--tlist", default=DEFAULT_TLIST, help="Output times for URANS/DES/LES setup")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Create the 3D geometry base only; do not try to create DES/LES physics")
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    if not base.is_file():
        print(f"[ERROR] Base model not found: {base}", file=sys.stderr)
        return 2
    outdir = Path(args.out).expanduser().resolve() if args.out else base.parent / "airfoil_four_models"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Setup-only run: no study solves and no 3D mesh generation.")
    print("[INFO] DES/LES files are 3D pilot templates, not validated production simulations.")
    print(f"[INFO] Base: {base}")
    print(f"[INFO] Output: {outdir}")

    client = mph.start(cores=args.cores)
    try:
        save_2d_variants(client, base, outdir, args.tlist)
        if args.geometry_only:
            create_3d_pilot(
                client, outdir / "airfoil_3D_geometry_base.mph", [],
                "3D geometry base", args.span, args.tlist, args.naca_points,
                geometry_only=True,
            )
            print("[DONE] 3D geometry base saved; DES/LES physics were not attempted.")
            return 0

        create_3d_pilot(
            client, outdir / "airfoil_DES_3D_PILOT.mph",
            ["DESRBVMSpalartAllmaras"],
            "DES 3D pilot", args.span, args.tlist, args.naca_points,
        )
        create_3d_pilot(
            client, outdir / "airfoil_LES_3D_PILOT.mph",
            ["LESRBVM"],
            "LES 3D pilot", args.span, args.tlist, args.naca_points,
        )
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    readme = outdir / "README_four_models.txt"
    readme.write_text(
        "COMSOL NACA0012 four-model setup\n"
        "Conditions: NACA0012, c=1 m, Re=5e5, alpha=25 deg, rho=1.225 kg/m^3, mu=1.787e-5 Pa*s.\n"
        "RANS/URANS are based on the user's 2D MPH model. DES/LES are 3D pilot templates, span=0.2 m, periodic in span.\n"
        "No models were solved. The 3D pilot mesh is not built and is not validated; refine wall/boundary-layer and wake mesh,\n"
        "verify periodic selections, initialize from an appropriate RANS field, and perform time-step/domain/mesh studies before quantitative use.\n",
        encoding="utf-8",
    )
    print(f"[OK] Notes: {readme}")
    print("[DONE] Review all four MPH files in COMSOL before solving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
