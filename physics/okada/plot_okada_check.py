# Usage:       python physics/okada/plot_okada_check.py
# Description: Visual sanity-check plots for the Okada (1985) opening-only dike
#              model (faithful NumPy port of okadaMod.m, Beauducel centroid
#              convention). Reproduces the synthetic test case from
#              fokada_iokada_synthetic_rosenthal.m and sweeps dip & strike.
#              Topography correction omitted (depth = centroid depth, no +z).
# Convention:  E,N,depth,L,W in metres; opening in metres; strike CW from North
#              (fault dips to RIGHT of trace); dip in [0,90]; nu = 0.25.
# Date:        2026-06-10

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless HPC backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
NU = 0.25

# --- Standard synthetic test values (from fokada_iokada_synthetic_rosenthal.m) ---
REF = dict(e0=0.0, n0=0.0, depth=10_000.0, strike=0.0, dip=90.0,
           L=1_000.0, W=1_000.0, opn=10.0)
SCALE = 30_000.0       # grid half-width, m  (linspace(-1,1)*scale)
NGRID = 128


# ----------------------------------------------------------------------
# Faithful NumPy port of okadaMod.m (opening-only, centroid-referenced)
# ----------------------------------------------------------------------
def okada_mod(e, n, e0, n0, depth, strike, dip, L, W, opn, nu=NU):
    """Surface ENU displacement (m) from an opening rectangular dislocation.

    All length inputs in metres; angles in degrees; opn (opening) in metres.
    Returns (uE, uN, uZ), each same shape as e/n.
    """
    U3 = opn
    strike = np.radians(strike)
    dip = np.radians(dip)
    e1 = e - e0
    n1 = n - n0

    # centroid-referenced geographic -> Okada internal (x, y, d)
    d = depth + np.sin(dip) * W / 2.0                 # deeper-edge depth
    ec = e1 + np.cos(strike) * np.cos(dip) * W / 2.0
    nc = n1 - np.sin(strike) * np.cos(dip) * W / 2.0
    x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2.0
    y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W
    p = y * np.cos(dip) + d * np.sin(dip)
    q = y * np.sin(dip) - d * np.cos(dip)

    cosd, sind = np.cos(dip), np.sin(dip)

    def I5(xi, eta, R, db):
        X = np.sqrt(xi**2 + q**2)
        if cosd > np.finfo(float).eps:
            val = (1 - 2*nu) * 2 / cosd * np.arctan(
                (eta*(X + q*cosd) + X*(R + X)*sind) / (xi*(R + X)*cosd))
            return np.where(xi == 0, 0.0, val)
        return -(1 - 2*nu) * xi * sind / (R + db)

    def I4(db, eta, R):
        if cosd > np.finfo(float).eps:
            return (1 - 2*nu) / cosd * (np.log(R + db) - sind*np.log(R + eta))
        return -(1 - 2*nu) * q / (R + db)

    def I3(eta, R):
        yb = eta*cosd + q*sind
        db = eta*sind - q*cosd
        if cosd > np.finfo(float).eps:
            return (1 - 2*nu)*(yb/(cosd*(R + db)) - np.log(R + eta)) \
                + sind/cosd * I4(db, eta, R)
        return (1 - 2*nu)/2 * (eta/(R + db) + yb*q/(R + db)**2 - np.log(R + eta))

    def I1(xi, eta, R):
        db = eta*sind - q*cosd
        if cosd > np.finfo(float).eps:
            return (1 - 2*nu)*(-xi/(cosd*(R + db))) - sind/cosd*I5(xi, eta, R, db)
        return -(1 - 2*nu)/2 * xi*q/(R + db)**2

    def ux_tf(xi, eta):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        return q**2/(R*(R + eta)) - I3(eta, R)*sind**2

    def uy_tf(xi, eta):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        u = -(eta*sind - q*cosd)*q/(R*(R + xi)) \
            - sind*xi*q/(R*(R + eta)) - I1(xi, eta, R)*sind**2
        return u + np.where(q != 0, sind*np.arctan(xi*eta/(q*R)), 0.0)

    def uz_tf(xi, eta):
        R = np.sqrt(xi**2 + eta**2 + q**2)
        db = eta*sind - q*cosd
        u = (eta*cosd + q*sind)*q/(R*(R + xi)) \
            + cosd*xi*q/(R*(R + eta)) - I5(xi, eta, R, db)*sind**2
        return u - np.where(q != 0, cosd*np.arctan(xi*eta/(q*R)), 0.0)

    def chinnery(f):
        return f(x, p) - f(x, p - W) - f(x - L, p) + f(x - L, p - W)

    ux = U3/(2*np.pi) * chinnery(ux_tf)
    uy = U3/(2*np.pi) * chinnery(uy_tf)
    uz = U3/(2*np.pi) * chinnery(uz_tf)

    uE = np.sin(strike)*ux - np.cos(strike)*uy
    uN = np.cos(strike)*ux + np.sin(strike)*uy
    return uE, uN, uz


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_grid():
    xv = np.linspace(-1, 1, NGRID) * SCALE
    yv = np.linspace(-1, 1, NGRID) * SCALE
    return np.meshgrid(xv, yv)


def fault_surface_projection(e0, n0, strike_deg, dip_deg, L, W):
    """4 surface-projected corners of the fault rectangle, in km."""
    s = np.radians(strike_deg)
    d = np.radians(dip_deg)
    strike_unit = np.array([np.sin(s), np.cos(s)])          # along strike (E,N)
    dip_az = np.array([np.cos(s), -np.sin(s)])              # right of strike
    corners = []
    for xi in (-L/2, L/2):
        for w in (-W/2, W/2):
            pE = e0 + xi*strike_unit[0] + w*np.cos(d)*dip_az[0]
            pN = n0 + xi*strike_unit[1] + w*np.cos(d)*dip_az[1]
            corners.append((pE/1000, pN/1000))
    # order for polygon: (-L/2,-W/2),(-L/2,+W/2),(+L/2,+W/2),(+L/2,-W/2)
    return [corners[0], corners[1], corners[3], corners[2]]


def add_fault(ax, **geom):
    pts = fault_surface_projection(geom["e0"], geom["n0"], geom["strike"],
                                   geom["dip"], geom["L"], geom["W"])
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="k",
                         lw=1.5, ls="--"))


def plot_field(ax, X, Y, U, title, sym=True, vmax=None, colorbar=True):
    U_mm = U * 1000.0                                   # m -> mm
    if vmax is None:
        vmax = np.nanmax(np.abs(U_mm))
    kw = dict(cmap="RdBu_r", shading="auto")
    if sym:
        kw.update(vmin=-vmax, vmax=vmax)
    im = ax.pcolormesh(X/1000, Y/1000, U_mm, **kw)
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title(title)
    ax.set_aspect("equal")
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Displacement (mm)")
    return im


def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, name + ".png")
    pdf = os.path.join(OUT_DIR, name + ".pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  saved {png}")
    print(f"  saved {pdf}")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_reference(X, Y):
    print("[1/3] Reference synthetic case (vertical dike, matches MATLAB)...")
    uE, uN, uZ = okada_mod(X, Y, **REF)
    print(f"  max |uE|={np.abs(uE).max()*1000:.2f} mm, "
          f"|uN|={np.abs(uN).max()*1000:.2f} mm, "
          f"|uZ|={np.abs(uZ).max()*1000:.2f} mm")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, U, lbl in zip(axes, (uE, uN, uZ), ("East", "North", "Up")):
        plot_field(ax, X, Y, U, f"{lbl} displacement")
        add_fault(ax, **REF)
    fig.suptitle("Okada opening dike  |  strike=0, dip=90, L=W=1km, "
                 "opening=10m, centroid depth=10km", fontsize=12)
    savefig(fig, "okada_reference_synthetic")


def fig_dip_sweep(X, Y):
    print("[2/3] Dip sweep (sill -> dike), full ENU...")
    cases = [dict(REF, dip=d, label=f"dip = {d}°") for d in (10, 45, 90)]
    enu_grid(X, Y, cases, "okada_dip_sweep",
             "Dip sweep (strike=0, L=W=1km): sill bowl -> dike, full ENU")


def fig_strike_sweep(X, Y):
    print("[3/3] Strike sweep (rotation check), full ENU...")
    cases = [dict(REF, dip=45.0, strike=s, L=3000.0, W=1000.0,
                  label=f"strike = {s}°") for s in (0, 45, 90)]
    enu_grid(X, Y, cases, "okada_strike_sweep",
             "Strike sweep (dip=45, L=3km, W=1km): ENU rotates with strike")


# Parameter sets for the ENU grid (one row each). All lengths in metres.
CASES = [
    dict(label="Vertical dike: strike=0, dip=90, L=W=1km, op=10m, dc=10km",
         e0=0., n0=0., depth=10_000., strike=0.,  dip=90., L=1_000., W=1_000., opn=10.),
    dict(label="Sill: strike=0, dip=5, L=W=2km, op=10m, dc=8km",
         e0=0., n0=0., depth=8_000.,  strike=0.,  dip=5.,  L=2_000., W=2_000., opn=10.),
    dict(label="Inclined dike: strike=45, dip=60, L=3km, W=1km, op=10m, dc=8km",
         e0=0., n0=0., depth=8_000.,  strike=45., dip=60., L=3_000., W=1_000., opn=10.),
    dict(label="Shallow dike: strike=0, dip=90, L=2km, W=1km, op=5m, dc=4km",
         e0=0., n0=0., depth=4_000.,  strike=0.,  dip=90., L=2_000., W=1_000., opn=5.),
    dict(label="NE elongated dike: strike=30, dip=75, L=5km, W=1km, op=8m, dc=6km",
         e0=0., n0=0., depth=6_000.,  strike=30., dip=75., L=5_000., W=1_000., opn=8.),
]


def enu_grid(X, Y, cases, fname, suptitle):
    """One row per case; columns = E, N, Up; shared symmetric scale per row.

    Each case is a geom dict (e0,n0,depth,strike,dip,L,W,opn) plus a 'label'.
    """
    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4.2 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    comps = ["East", "North", "Up"]
    for r, case in enumerate(cases):
        geom = {k: case[k] for k in
                ("e0", "n0", "depth", "strike", "dip", "L", "W", "opn")}
        uE, uN, uZ = okada_mod(X, Y, **geom)
        fields = [uE, uN, uZ]
        vmax = max(np.nanmax(np.abs(f)) for f in fields) * 1000.0   # shared, mm
        print(f"  row {r}: {case['label']}  -> peaks (mm) "
              f"|E|={np.abs(uE).max()*1000:6.2f} "
              f"|N|={np.abs(uN).max()*1000:6.2f} "
              f"|Up|={np.abs(uZ).max()*1000:6.2f}")
        ims = []
        for c, (U, comp) in enumerate(zip(fields, comps)):
            ax = axes[r, c]
            im = plot_field(ax, X, Y, U, comp if r == 0 else "",
                            vmax=vmax, colorbar=False)
            add_fault(ax, **geom)
            ims.append(im)
            if c > 0:
                ax.set_ylabel("")
        cb = fig.colorbar(ims[-1], ax=list(axes[r, :]), fraction=0.020, pad=0.02)
        cb.set_label("Displacement (mm)")
        axes[r, 0].text(-0.32, 0.5, case["label"], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=8)
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.04,
                        hspace=0.28, wspace=0.15)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{fname}.{ext}")
        fig.savefig(path, dpi=150)
        print(f"  saved {path}")
    plt.close(fig)


def fig_enu_grid(X, Y):
    """Varied-geometry ENU grid (one row per parameter set)."""
    print("[ENU grid] %d parameter sets x (E,N,Up)..." % len(CASES))
    enu_grid(X, Y, CASES, "okada_enu_grid",
             "Okada opening dislocation: ENU surface displacement "
             "for varied geometries (shared scale per row)")


if __name__ == "__main__":
    print("=" * 60)
    print("Okada opening-dike visual check (okadaMod.m NumPy port)")
    print("=" * 60)
    X, Y = make_grid()
    print(f"  grid: {NGRID}x{NGRID} over +/-{SCALE/1000:.0f} km")
    fig_reference(X, Y)
    fig_dip_sweep(X, Y)
    fig_strike_sweep(X, Y)
    fig_enu_grid(X, Y)
    print(f"\nDone. Figures in {OUT_DIR}")
