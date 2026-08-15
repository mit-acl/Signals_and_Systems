"""
control_materials.py

Control utilities for 16.002.
All environment/setup is opt-in via setup_environment().
"""

__version__ = "16.002-0.1"

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import control.matlab as cmat

import importlib.util
from dataclasses import dataclass
from typing import List
from IPython.display import Math, display, Markdown, Latex, HTML

#import scipy.linalg
from scipy.linalg import solve_continuous_lyapunov, svd, sqrtm, cholesky, eigvals, eigh # symmetric matrices
from scipy.signal import residue
import re

from types import SimpleNamespace

from pathlib import Path
import sys

# repo_root/16_06_Class/notebooks → repo_root/16_06_Class
repo_root = Path.cwd().parents[0]
sys.path.insert(0, str(repo_root / "16_06_Class"))

import courseutils.basic_material as bm
bm.setup_environment()

# constants
r2d = 180.0 / np.pi
tpi = 2 * np.pi

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# -------------------------------
# Environment helpers
# -------------------------------

def _require_package(name):
    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"Required package '{name}' not found. "
            "Please install it following the course instructions."
        )

def setup_environment(*, verbose=False):
    """
    Opt-in environment setup for control_materials.
    - checks that control, scipy, sympy are available
    - sets control plotting defaults (if control is available)
    - configures matplotlib fonts/sizes consistent with course
    """
    # check required packages
    _require_package("control")
    _require_package("scipy")
    _require_package("sympy")

    # now import control and set defaults
    import control as ct
    # set defaults for Nyquist plotting (explicit, not on import)
    try:
        ct.set_defaults("nyquist", max_curve_magnitude=100)
    except Exception:
        # not fatal; just a best-effort setting
        pass

    # stop annoying warnings
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in divide"
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide"
    )
    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # matplotlib style consistent with basic_material
    from matplotlib import rcParams
    rcParams["font.serif"] = "cmr14"
    rcParams.update({"font.size": 10})
    plt.rcParams["figure.figsize"] = [8, 5.0]
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.autolayout"] = True

    if verbose:
        import sympy as sym
        import platform
        print("control_materials: environment set")
        print("Python:", platform.python_version())
        print("SymPy:", sym.__version__)

# -------------------------------
# Utility functions
# -------------------------------

###########################################################################################
###########################################################################################
def caption(txt, fig=None, xloc=0.5, yloc=-0.05):
    """
    Add a centered caption to a figure (below the axes).
    If fig is None, uses the current figure.
    """
    if fig is None:
        fig = plt.gcf()
    fig.text(xloc, yloc, txt, ha="center", size=MEDIUM_SIZE, color="blue")

###########################################################################################
###########################################################################################
def Read_data(file_name, comments=None, cols=None):
    """Load data from a CSV file with specified comments and columns.

    Parameters:
    -----------
    file_name : str
        Path to the CSV file
    comments : list, optional
        Characters indicating comment lines. Default: ["#", "F"]
    cols : list, optional
        Columns to read. Default: [0]
    """
    file_path = Path(file_name)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Preserve the existing default behavior
    if comments is None:
        comments = ["#", "F"]
    
    try:
        data = np.loadtxt(file_path, comments=comments)
    except ValueError as e:
        raise ValueError(f"Failed to parse {file_name}: {e}") from e
    
    if cols is not None:
        return data[:, cols]
    return data

###########################################################################################
###########################################################################################
def pretty_row_print(X,msg="",sigfigs=None,decimals=3,complex_decimals=2,verbose=None,bracket=None):
    """
    Pretty print a row of real or complex numbers.

    Exactly one of sigfigs or decimals should be used.
    """
    if isinstance(X, str):
        X, msg = msg, X

    if sigfigs is not None:
        decimals = None

    X = np.asarray(X).squeeze().ravel()

    def fmt_real(x):
        return f"{x:.{sigfigs}g}" if sigfigs is not None else f"{x:.{decimals}f}"

    def fmt_complex(x):
        r = x.real
        i = x.imag

        if sigfigs is not None:
            r_str = f"{r:.{sigfigs}g}"
            i_mag_str = f"{abs(i):.{sigfigs}g}"
        else:
            r_str = f"{r:.{complex_decimals}f}"
            i_mag_str = f"{abs(i):.{complex_decimals}f}"

        if abs(i) < 1e-12:
            return r_str

        if abs(r) < 1e-12:
            sign = "-" if i < 0 else ""
            return f"({sign}{i_mag_str}i)"

        sign = "-" if i < 0 else "+"
        return f"({r_str} {sign} {i_mag_str}i)"

    out = []
    for x in X:
        x = complex(x)
        if abs(x.imag) > 1e-12:
            out.append(fmt_complex(x))
        else:
            out.append(fmt_real(x.real))

    body = ", ".join(out)

    if bracket is None:
        row = f"{msg} {body}".strip()
    else:
        if isinstance(bracket, str):
            left, right = bracket
        else:
            left, right = bracket
        row = f"{msg} {left}{body}{right}".strip()

    if verbose:
        return row
    else:
        display(HTML(
            f"<pre style='font-size:12px; margin:0; line-height:1.15;'>{row}</pre>"
        ))

###########################################################################################
###########################################################################################
def feedback_ff(G, K, Kff):
    if isinstance(G, (int, float, np.number)):
        G = ct.tf([G], [1])
    elif isinstance(K, ct.StateSpace):
        G = ct.ss2tf(G)
    elif not isinstance(G, ct.TransferFunction):
        raise TypeError("G must be a scalar, TransferFunction, or StateSpace")

    if isinstance(K, (int, float, np.number)):
        K = ct.tf([K], [1])
    elif isinstance(K, ct.StateSpace):
        K = ct.ss2tf(K)
    elif not isinstance(K, ct.TransferFunction):
        raise TypeError("K must be a scalar, TransferFunction, or StateSpace")

    NG, DG = ct.tfdata(G)
    NG = np.atleast_1d(np.squeeze(NG))
    DG = np.atleast_1d(np.squeeze(DG))

    NC, DC = ct.tfdata(K)
    NC = np.atleast_1d(np.squeeze(NC))
    DC = np.atleast_1d(np.squeeze(DC))

    NGDC = np.convolve(NG, DC)
    NGNC = np.convolve(NG, NC)
    DGDC = np.convolve(DG, DC)

    max_len = max(len(DGDC), len(NGNC), len(NGDC))
    NGNC = np.pad(NGNC, (max_len - len(NGNC), 0), "constant")
    NGDC = np.pad(NGDC, (max_len - len(NGDC), 0), "constant")
    DGDC = np.pad(DGDC, (max_len - len(DGDC), 0), "constant")

    return ct.tf(Kff * NGDC + NGNC, DGDC + NGNC)

###########################################################################################
###########################################################################################
def writeGc(filename, Gc):
    """
    Write controller info to filename. Each piece on its own line:
     - zeros (real parts comma separated)
     - poles (real parts comma separated)
     - DC gain (single number)
    """
    zs = [float(np.real(z)) for z in Gc.zeros()]
    ps = [float(np.real(p)) for p in Gc.poles()]

    num, den = ct.tfdata(Gc)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    gain = float(num[0] / den[0]) if (len(num) and len(den)) else 0.0

    with open(filename, "w") as f:
        f.write("zeros:" + ",".join(f"{z:4.2f}" for z in zs) + "\n")
        f.write("poles:" + ",".join(f"{p:4.2f}" for p in ps) + "\n")
        f.write("gain:" + f"{gain:4.2f}" + "\n")

######################################################   
# sympy helpers
#####################################################
def round_constants(expr, ndigits=3):
    return expr.xreplace({
        c: sp.Float(c, ndigits) for c in expr.atoms()
        if c.is_Number and not c.is_Integer
    })


######################################################   
# TF helpers
######################################################
def write_two_column_array(col1, col2, filename, sigfigs=4, title1='Residue',title2='Poles'):
    with open(filename, "w") as f:
        f.write("\\begin{array}{cc}\n")
        f.write(title1+" & "+title2+" \\\\\n")
        for a, b in zip(col1, col2):
            f.write(f"{a:.{sigfigs}f} & {b:.{sigfigs}f} \\\\\n")
        f.write("\\end{array}\n")


def write_latex_array(X, filename, msgs=None, cols=1, tol=1e-12, decimals=None, sigfigs=None):
    """
    Write a list/array of (possibly complex) numbers to a LaTeX array
    that can be \\input{} directly.

    Parameters
    ----------
    X : iterable
        Numbers (real or complex)
    filename : str
        Output .tex file
    msgs : str
        latex label
    cols : int
        Number of columns in the array
    tol : float
        Imaginary-part tolerance for treating numbers as real
    """

    # accept sigfigs as alias for decimals
    if decimals is None and sigfigs is None:
        decimals = 2   # default
    elif decimals is None:
        decimals = sigfigs
    elif sigfigs is None:
        pass
    else:
        if decimals != sigfigs:
            raise ValueError("decimals and sigfigs must match if both are given")

    X = np.atleast_1d(X).astype(complex)

    used = np.zeros(len(X), dtype=bool)
    entries = []

    def fmt_real(x):
        return f"{x:.{decimals}f}"

    for i, z in enumerate(X):
        if used[i]:
            continue

        zr, zi = z.real, z.imag

        # try to find conjugate
        paired = False
        if abs(zi) > tol:
            for j in range(i + 1, len(X)):
                if used[j]:
                    continue
                zj = X[j]
                if (abs(zj.real - zr) < tol and
                    abs(zj.imag + zi) < tol):
                    # conjugate pair found
                    entries.append(
                        rf"({fmt_real(zr)} \pm {fmt_real(abs(zi))}i)"
                    )
                    used[i] = used[j] = True
                    paired = True
                    break

        if paired:
            continue

        # no conjugate pair
        if abs(zi) < tol:
            entries.append(fmt_real(zr))
        else:
            sign = "+" if zi >= 0 else "-"
            entries.append(
                rf"({fmt_real(zr)} {sign} {fmt_real(abs(zi))}i)"
            )
        used[i] = True

    # split into rows
    rows = [
        entries[i:i+cols]
        for i in range(0, len(entries), cols)
    ]

    with open(filename, "w") as f:
        f.write("\\begin{array}{%s}\n" % ("c" * cols))
        if msgs:
            f.write(msgs + "\n")
        for r in rows:
            f.write("  " + " & ".join(r) + " \\\\\n")
        f.write("\\end{array}\n")

###########################################################################################
###########################################################################################
def show_tf_latex(P, label=None, sigfigs=2, show=True, factor=False,
                  name=None, time_constant=False):
    ''' 
    P: system
    label
    show
    factor
    time_constant: if True, normalize first order real factors to (s/a + 1)
    '''

    if P is None:
        return f"G"

    is_discrete = P.dt is not None and P.dt > 0
    var = "z" if is_discrete else "s"
    
    if label is None:
        label = f"G({var})"
    if name is not None:
        label = name

    num, den = ct.tfdata(P)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    if factor:
        Kn, rnum, qnum = factor_poly_real(num)
        Kd, rden, qden = factor_poly_real(den)

        # cancel common real roots (returns deterministic sorted lists now)
        rnum_c, rden_c = cancel_common_real_roots(rnum, rden, tol=1e-6)

        # ensure deterministic ordering (in case upstream callers pass unsorted lists)
        rnum_c = sorted(rnum_c, key=lambda r: (abs(r), r))
        rden_c = sorted(rden_c, key=lambda r: (abs(r), r))

        # quadratics: sort by C (which is a^2 + b^2) then B for stability
        qnum = sorted(qnum, key=lambda bc: (bc[1], bc[0]))
        qden = sorted(qden, key=lambda bc: (bc[1], bc[0]))

        # apply time-constant normalization AFTER sorting the physical roots so ordering
        # is done on the actual root locations (more intuitive)
        if time_constant:
            def normalize_real_roots(rlist):
                new_roots = []
                gain_scale = 1.0
                for r in rlist:
                    if np.isreal(r):
                        r = float(np.real(r))
                        a = -r
                        if a != 0:
                            gain_scale *= a
                            new_roots.append(-a)  # store as root of (s/a + 1)
                        else:
                            new_roots.append(r)
                    else:
                        new_roots.append(r)
                return np.array(new_roots), gain_scale

            rnum_c, scale_num = normalize_real_roots(rnum_c)
            rden_c, scale_den = normalize_real_roots(rden_c)

            Kn *= scale_num
            Kd *= scale_den

        # build latex bodies with the now-ordered lists
        num_body = factors_to_latex(rnum_c, qnum, var, sigfigs,
                                    time_constant=time_constant)
        den_body = factors_to_latex(rden_c, qden, var, sigfigs,
                                    time_constant=time_constant)

        frac = build_frac_latex_gain_in_numer(Kn, num_body, Kd, den_body, sigfigs)

    else:
        num_tex = _poly_to_latex(num, sigfigs=sigfigs, var=var, discrete=is_discrete)
        den_tex = _poly_to_latex(den, sigfigs=sigfigs, var=var, discrete=is_discrete)
        frac = rf"\dfrac{{{num_tex}}}{{{den_tex}}}"

    latex_str = rf"${label} = {frac}$"

    if show is True:
        display(Math(latex_str))
        return None

    return latex_str

###########################################################################################
###########################################################################################
def _sci_to_latex(s):
    """
    Convert '4.4e-06' -> '4.4 \\times 10^{-6}'
    """
    if "e" in s:
        base, exp = s.split("e")
        return rf"{base} \times 10^{{{int(exp)}}}"
    return s

###########################################################################################
###########################################################################################
def tf_to_latex(G, sigfigs=2, factor=False, time_constant=False):
    """
    Backward-compatible wrapper for show_tf_latex.
    Returns LaTeX string (without surrounding $).
    """
    latex_str = show_tf_latex(G,label=None,sigfigs=sigfigs,
        show=False,factor=factor,time_constant=time_constant)
    # remove outer $...$ added by show_tf_latex
    if latex_str.startswith("$") and latex_str.endswith("$"):
        latex_str = latex_str[1:-1]
    return latex_str

###########################################################################################
###########################################################################################
def _matrix_to_latex(M, sigfigs=4, tol=1e-12, exp_thresh=1e-3):
    M = np.atleast_2d(np.array(M, dtype=float))

    def fmt(x):
        if abs(x) < tol:
            return "0"

        # use exponential form for small/large numbers
        if abs(x) < exp_thresh or abs(x) >= 10/exp_thresh:
            mant, exp = f"{x:.{sigfigs}e}".split("e")
            exp = int(exp)
            return rf"{mant}e^{{{exp}}}"
        else:
            return f"{x:.{sigfigs}g}"

    rows = []
    for row in M:
        rows.append(" & ".join(fmt(x) for x in row))

    body = r" \\ ".join(rows)
    return r"\begin{bmatrix} " + body + r" \end{bmatrix}"

###########################################################################################
###########################################################################################
def show_ss_latex(P, label=None, sigfigs=4, name=None, show=True):
    """
    Display a StateSpace system as LaTeX with A, B, C, D matrices.
    """

    if not isinstance(P, ct.StateSpace):
        raise TypeError("Input must be a control.StateSpace object")

    dt = getattr(P, "dt", None)

    if dt is None or dt == 0:
        is_discrete = False
    else:
        is_discrete = True

    var = "k" if is_discrete else "t"

    # label handling
    if label is None and name is None:
        label = ""
    elif label is None:
        label = name
    else:
        label = label

    A, B, C, D = P.A, P.B, P.C, P.D

    A_tex = _matrix_to_latex(A, sigfigs)
    B_tex = _matrix_to_latex(B, sigfigs)
    C_tex = _matrix_to_latex(C, sigfigs)
    D_tex = _matrix_to_latex(D, sigfigs)

    if is_discrete:
        eqn = (
            r"\begin{aligned}"
            r"x_{k+1} &= " + A_tex + r" x_k + " + B_tex + r" u_k \\ "
            r"y_k &= " + C_tex + r" x_k + " + D_tex + r" u_k"
            r"\end{aligned}"
        )
    else:
        eqn = (
            r"\begin{aligned}"
            r"\dot{x}(t) &= " + A_tex + r" x(t) + " + B_tex + r" u(t) \\ "
            r"y(t) &= " + C_tex + r" x(t) + " + D_tex + r" u(t)"
            r"\end{aligned}"
        )

    if label:
        latex_str = label + ":\n" + eqn
    else:
        latex_str = eqn

    if show:
        display(Math(latex_str))
        return None
    else:
        return Math(latex_str)

###########################################################################################
###########################################################################################
def fmt(x, sigfigs=4, tol=1e-10):
    x = float(x)
    if abs(x) < tol:
        return f"{0:.{sigfigs}f}"
    return f"{x:.{sigfigs}f}"

###########################################################################################
###########################################################################################
def build_frac_latex(Kn, num_body, Kd, den_body, sigfigs=4, tol=1e-8):
    #fmt = lambda x: np.format_float_positional(x, precision=sigfigs, trim='-')

    if num_body == "1":
        num_body = None
    if den_body == "1":
        den_body = None

    K = Kn / Kd

    if den_body is None and num_body is None:
        return rf"\displaystyle {fmt(K, sigfigs)}"

    if den_body is None:
        if abs(K - 1) < tol:
            return rf"\displaystyle {num_body}"
        return rf"\displaystyle {fmt(K, sigfigs)}\,{num_body}"

    if num_body is None:
        return rf"\displaystyle {fmt(K, sigfigs)}\,\dfrac{{1}}{{{den_body}}}"

    if abs(K - 1) < tol:
        return rf"\displaystyle \dfrac{{{num_body}}}{{{den_body}}}"

    return rf"\displaystyle {fmt(K)}\,\dfrac{{{num_body}}}{{{den_body}}}"

###########################################################################################
###########################################################################################
def residue_tf(G, time_constant=False, tol=1e-12):
    """
    Partial fraction expansion of transfer function G.

    Standard form:
        r, a, k  corresponds to  r / (s + a)

    Time constant form:
        r_tc, a, k  corresponds to  r_tc / (s/a + 1)

        where
            a = -p
            r_tc = r / a

        For a stable pole p < 0, a > 0.
        Integrators (p ≈ 0) are returned as a = 0 and r/s.
    """
    num = np.squeeze(G.num)
    den = np.squeeze(G.den)
    r, p, k = residue(num, den)

    if not time_constant:
        return r, -p, k

    r_tc = []
    a_vals = []
    for ri, pi in zip(r, p):
        # integrator
        if abs(pi) < tol:
            r_tc.append(ri)
            a_vals.append(0.0)
            continue

        a = -pi            # define break frequency
        r_new = ri / a     # rescale residue

        r_tc.append(r_new)
        a_vals.append(a)

    return np.array(r_tc), np.array(a_vals), k

###########################################################################################
###########################################################################################
def factors_to_latex(real_roots, quads, var="s", sigfigs=4, tol=1e-8, time_constant=False):
    parts = []

    real_roots = list(np.asarray(real_roots, dtype=float))

    quads_out = []
    for B, C in quads:
        disc = B * B - 4 * C
        scale = max(1.0, abs(B * B), abs(C))
        if abs(disc) <= tol * scale:
            r = -B / 2.0
            real_roots.extend([r, r])
        else:
            quads_out.append((B, C))

    real_roots = np.asarray(real_roots, dtype=float)

    zero_mask = np.abs(real_roots) < tol
    n_zero = np.sum(zero_mask)

    if n_zero == 1:
        parts.append(var)
    elif n_zero > 1:
        parts.append(f"{var}^{n_zero}")

    nz_roots = real_roots[~zero_mask]

    if len(nz_roots) > 0:
        # preserve incoming order
        nz_roots = np.round(nz_roots, sigfigs)

        clustered = []
        for r in nz_roots:
            if not clustered:
                clustered.append([r, 1])
            else:
                if abs(r - clustered[-1][0]) <= tol * max(1.0, abs(clustered[-1][0])):
                    clustered[-1][1] += 1
                else:
                    clustered.append([r, 1])

        for r, mult in clustered:
            a = -r

            if time_constant and abs(a) > tol:
                a_fmt = fmt(abs(a), sigfigs)
                if a > 0:
                    factor = rf"\left(\frac{{{var}}}{{{a_fmt}}}+1\right)"
                else:
                    factor = rf"\left(\frac{{{var}}}{{{a_fmt}}}-1\right)"
            else:
                if a > 0:
                    factor = f"({var}+{fmt(a, sigfigs)})"
                else:
                    factor = f"({var}-{fmt(abs(a), sigfigs)})"

            if mult > 1:
                factor += f"^{mult}"

            parts.append(factor)

    for B, C in quads_out:
        quad = f"{var}^2"

        if abs(B) > tol:
            signB = "-" if B < 0 else "+"
            quad += f"{signB}{fmt(abs(B), sigfigs)}{var}"

        if abs(C) > tol:
            signC = "-" if C < 0 else "+"
            quad += f"{signC}{fmt(abs(C), sigfigs)}"

        parts.append(f"({quad})")

    return "1" if not parts else "".join(parts)

###########################################################################################
###########################################################################################
def cancel_common_roots(K_num, roots_num, K_den, roots_den, tol=1e-6):
    """
    Remove common roots between num and den (within tol), update K_num/K_den accordingly.
    Returns K_num_new, roots_num_new, K_den_new, roots_den_new
    """
    num_roots = roots_num.copy()
    den_roots = roots_den.copy()
    used_den = [False]*len(den_roots)
    remaining_num = []
    for r in num_roots:
        found = False
        for j, rd in enumerate(den_roots):
            if not used_den[j] and abs(r - rd) <= tol:
                # cancel this root pair
                used_den[j] = True
                found = True
                break
        if not found:
            remaining_num.append(r)
    remaining_den = [rd for j,rd in enumerate(den_roots) if not used_den[j]]
    # If canceled roots, adjust scalar gain by multiplying factor (den/numer) from polynomial values.
    # Simpler: leave K_num/K_den unchanged (they are leading coeffs). Cancelling roots does not change K_total.
    return K_num, remaining_num, K_den, remaining_den

###########################################################################################
###########################################################################################
# ---------- build fraction latex ----------
def build_fraction_latex_from_roots(K_num, roots_num, K_den, roots_den,
                                    var='s', sigfigs=4, Tol=1e-9, cancel=False):
    # optionally cancel common roots
    if cancel:
        K_num, roots_num, K_den, roots_den = cancel_common_roots(K_num, roots_num, K_den, roots_den, tol=1e-6)

    # build bodies from numeric roots (must be symmetric with factor builder)
    num_parts = []
    for r in sorted(roots_num, key=lambda z: (round(z.real,8), round(z.imag,8))):
        num_parts.append(_build_linear_factor_from_root(r, var=var, Tol=Tol, sigfigs=sigfigs))
    den_parts = []
    for r in sorted(roots_den, key=lambda z: (round(z.real,8), round(z.imag,8))):
        den_parts.append(_build_linear_factor_from_root(r, var=var, Tol=Tol, sigfigs=sigfigs))

    num_body = "1" if len(num_parts) == 0 else "".join(num_parts)
    den_body = "1" if len(den_parts) == 0 else "".join(den_parts)

    # compute K_total
    if abs(K_den) < 1e-16:
        K_total = np.inf
    else:
        K_total = float(K_num) / float(K_den)
    K_tex = fmt(K_total, sigfigs)

    # clean
    def clean(b):
        if b is None:
            return None
        b = str(b).strip()
        return None if b in ("", "1") else b

    nb = clean(num_body)
    db = clean(den_body)

    # assemble with safe spacing and +/- handling
    if nb is None and db is None:
        if np.isfinite(K_total):
            return rf"\displaystyle {K_tex}"
        else:
            return r"\displaystyle 0"

    if db is None:
        if abs(K_total - 1) < Tol:
            return rf"\displaystyle {nb}"
        elif abs(K_total + 1) < Tol:
            return rf"\displaystyle -{nb}"
        else:
            return rf"\displaystyle {K_tex}\,{nb}"

    if nb is None:
        if abs(K_total - 1) < Tol:
            return rf"\displaystyle \dfrac{{1}}{{{db}}}"
        elif abs(K_total + 1) < Tol:
            return rf"\displaystyle -\dfrac{{1}}{{{db}}}"
        else:
            return rf"\displaystyle {K_tex}\,\dfrac{{1}}{{{db}}}"

    # both present
    if abs(K_total - 1) < Tol:
        return rf"\displaystyle \dfrac{{{nb}}}{{{db}}}"
    elif abs(K_total + 1) < Tol:
        return rf"\displaystyle -\dfrac{{{nb}}}{{{db}}}"
    else:
        return rf"\displaystyle {K_tex}\,\dfrac{{{nb}}}{{{db}}}"


###########################################################################################
###########################################################################################
def cancel_common_real_roots(rnum, rden, tol=1e-6):
    rnum = list(rnum)
    rden = list(rden)

    rnum_out = []
    rden_used = [False]*len(rden)

    for rn in rnum:
        cancelled = False
        for j, rd in enumerate(rden):
            if not rden_used[j] and abs(rn - rd) < tol:
                rden_used[j] = True
                cancelled = True
                break
        if not cancelled:
            rnum_out.append(rn)

    rden_out = [rd for j, rd in enumerate(rden) if not rden_used[j]]
    return rnum_out, rden_out

def build_frac_latex_gain_in_numer(Kn, num_body, Kd, den_body, sigfigs=4, Tol=1e-9):
    """
    Build LaTeX for:
        (Kn/Kd) * (num_body / den_body)
    with the net gain placed INSIDE the numerator.

    Mathtext-safe version:
      - no \\displaystyle
      - no \\dfrac
      - suitable for Matplotlib and notebooks
    """

    def clean(body):
        if body is None:
            return None
        b = str(body).strip()
        return None if b in ("", "1") else b

    nb = clean(num_body)
    db = clean(den_body)

    # net gain
    K = Kn / Kd
    Ktex = fmt(K, sigfigs)

    # ---- cases ----

    # pure scalar
    if nb is None and db is None:
        return rf"{Ktex}"

    # numerator only
    if db is None:
        if abs(K - 1) < Tol:
            return rf"{nb}"
        elif abs(K + 1) < Tol:
            return rf"-{nb}"
        else:
            return rf"{Ktex}\,{nb}"

    # denominator only
    if nb is None:
        if abs(K - 1) < Tol:
            return rf"\dfrac{{1}}{{{db}}}"
        elif abs(K + 1) < Tol:
            return rf"-\dfrac{{1}}{{{db}}}"
        else:
            return rf"\dfrac{{{Ktex}}}{{{db}}}"

    # full fraction
    if abs(K - 1) < Tol:
        return rf"\dfrac{{{nb}}}{{{db}}}"
    elif abs(K + 1) < Tol:
        return rf"-\dfrac{{{nb}}}{{{db}}}"
    else:
        return rf"\dfrac{{{Ktex}\,{nb}}}{{{db}}}"

###########################################################################################
###########################################################################################
def group_real_roots(real_roots, tol=1e-6):
    groups = []
    for r in real_roots:
        for g in groups:
            if abs(r - g[0]) < tol:
                g.append(r)
                break
        else:
            groups.append([r])
    return groups

###########################################################################################
###########################################################################################
def poly_factors_to_latex(K, real_roots, quads, sigfigs=4):
    terms = []

    # real roots
    for g in group_real_roots(real_roots):
        r = g[0]
        mult = len(g)

        # ---- HANDLE ZERO ROOT CLEANLY ----
        if abs(r) < 1e-12:
            term = "s"
            if mult > 1:
                term += f"^{mult}"
            terms.append(term)
            continue
        # -----------------------------------

        a = -r
        term = f"(s {'+' if a >= 0 else '-'} {abs(a):.{sigfigs}g})"
        if mult > 1:
            term += f"^{mult}"
        terms.append(term)

    # quadratic factors
    for B, C in quads:
        terms.append(
            f"(s^2 + {B:.{sigfigs}g}s + {C:.{sigfigs}g})"
        )

    # no factors → return empty string, not "1"
    body = "".join(terms)

    # only include K if it is not 1
    if abs(K - 1.0) > 1e-12:
        return f"{K:.{sigfigs}g}" + body

    return body

###########################################################################################
###########################################################################################
def _poly_to_latex(coefs, sigfigs=4, var="s", discrete=False, Tol = 1e-12):
    terms = []

    coefs = np.atleast_1d(coefs).astype(float)

    # Trim leading near-zero coefficients
    while len(coefs) > 1 and abs(coefs[0]) < Tol:
        coefs = coefs[1:]

    n = len(coefs)

    for i, val in enumerate(coefs):

        if abs(val) < Tol:
            continue

        # sign and magnitude
        sign = "-" if val < 0 else "+"
        mag = abs(val)

        # format once to significant figures
        coeff_str = f"{mag:.{sigfigs}f}"
        coeff_str = _sci_to_latex(coeff_str)

        if discrete:
            power = i
            if power == 0:
                term_body = coeff_str
            else:
                term_body = rf"{'' if coeff_str == '1' else coeff_str}{var}^{{-{power}}}"
        else:
            degree = n - 1 - i
            if degree > 1:
                term_body = rf"{'' if abs(mag-1.0)<Tol else coeff_str}{var}^{degree}"
            elif degree == 1:
                term_body = rf"{'' if abs(mag-1.0)<Tol else coeff_str}{var}"
            else:
                term_body = coeff_str

        terms.append((sign, term_body))

    if not terms:
        return "0"

    # first term: suppress leading '+'
    first_sign, first_term = terms[0]
    result = ("" if first_sign == "+" else "-") + first_term

    for sign, term in terms[1:]:
        result += f" {sign} {term}"

    return result

###########################################################################################
###########################################################################################
def factor_poly_real(coeffs, tol=1e-6):
    """
    coeffs: highest to lowest
    returns:
        K          : leading coefficient
        real_roots : list of real roots, with repeats preserved
        quads      : list of (B, C) for s^2 + B s + C
    """
    coeffs = np.atleast_1d(np.asarray(coeffs, dtype=float))

    if coeffs.size == 1:
        return coeffs[0], [], []

    K = coeffs[0]
    roots = np.roots(coeffs)

    used = np.zeros(len(roots), dtype=bool)
    real_roots = []
    quads = []

    for i, r in enumerate(roots):
        if used[i]:
            continue

        imag_tol = tol * (1.0 + abs(r.real))

        if abs(r.imag) <= imag_tol:
            real_roots.append(float(r.real))
            used[i] = True
            continue

        for j in range(i + 1, len(roots)):
            if used[j]:
                continue
            if abs(roots[j] - np.conj(r)) <= imag_tol:
                used[i] = True
                used[j] = True
                a = float(r.real)
                b = float(abs(r.imag))
                quads.append((-2*a, a*a + b*b))
                break

    # convert nearly repeated-root quadratics into double real roots
    quads_out = []
    extra_real_roots = []

    for B, C in quads:
        disc = B*B - 4*C
        disc_tol = tol * max(1.0, abs(B*B), abs(C))
        if abs(disc) <= disc_tol:
            r = -B / 2.0
            extra_real_roots.extend([r, r])
        else:
            quads_out.append((B, C))

    real_roots.extend(extra_real_roots)

    # cluster real roots numerically, but preserve multiplicity by re-expanding
    real_roots.sort()
    clustered = []

    for r in real_roots:
        if not clustered:
            clustered.append([r, 1])
        elif abs(r - clustered[-1][0]) <= tol * max(1.0, abs(clustered[-1][0])):
            clustered[-1][1] += 1
        else:
            clustered.append([r, 1])

    real_roots_out = []
    for r, mult in clustered:
        real_roots_out.extend([r] * mult)

    real_roots_out = sorted(real_roots_out, key=lambda r: (abs(r), r))
    quads_out = sorted(quads_out, key=lambda q: (q[1], q[0]))

    return K, real_roots_out, quads_out

###########################################################################################
###########################################################################################
def pid(Kp = 0, Ki = 0, Kd = 0):
    '''return tf form of a PID controller given Kp,Ki,Kd'''
    s = ct.tf((1,0),(1))
    return ct.tf(Kp,1) + Ki/s + Kd*s

###########################################################################################
###########################################################################################
def nyquist(*args, **kwargs):
    """
    Wrapper around control.nyquist_plot that always suppresses title
    unless explicitly overridden.
    """
    kwargs.setdefault("title", "")
    return ct.nyquist_plot(*args, **kwargs)

###########################################################################################
###########################################################################################
def write_latex_constants(S0, filename="./figs/constants.tex", idname=None, fmt="%.2f", sigfigs=None):
    '''
    consts = {"wn": wn,
        "zeta": zeta,
        "c1": c1,
        "c2": c2}
    filename
    idname
    fmt="%.2f"
    '''
    if sigfigs is not None:
        # allow letters only (TeX control sequence safe)
        fmt = f"%.{sigfigs}f"

    def sanitize_letters(s):
        return re.sub(r"[^A-Za-z]", "", s)

    suffix = ""
    if idname:
        suffix = sanitize_letters(idname).capitalize()

    with open(filename, "w") as f:
        f.write("% Auto-generated by Python. Do not edit.\n")
        for name, val in S0.items():
            macro = sanitize_letters(name) + suffix
            f.write(r"\def\%s{%s}" % (macro, fmt % val) + "\n")

###########################################################################################
###########################################################################################
def write_tf_latex(G, filename, label, sigfigs=4,
                   factor=False, inline=False,
                   name=None, time_constant=False,show=False):

    if show:
        show_tf_latex(G,label=label,sigfigs=sigfigs,show=True,
        factor=factor,name=name,time_constant=time_constant)

    latex_str = show_tf_latex(G,label=label,sigfigs=sigfigs,show=False,
        factor=factor,name=name,time_constant=time_constant)

    # remove outer $...$ from show_tf_latex
    if latex_str.startswith("$") and latex_str.endswith("$"):
        latex_str = latex_str[1:-1]

    with open(filename, "w") as f:
        if inline:
            f.write("$\n")
            f.write(latex_str + "\n")
            f.write("$\n")
        else:
            f.write("\\[\n")
            f.write(latex_str + "\n")
            f.write("\\]\n")

###########################################################################################
###########################################################################################
def normalize_tf(G):
    '''factor out non-unity gain for leading coefficient of the denominator'''
    if isinstance(G, ct.StateSpace):
        G = ct.ss2tf(G)

    num, den = ct.tfdata(G)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    if den[0] != 0:
        num = num/den[0]
        den = den/den[0]

    return ct.tf(num,den)

###########################################################################################
###########################################################################################
def find_double_real_poles(real_poles, tol=1e-5):
    """
    Identify real poles that occur more than once (within tolerance)
    and return one representative value per location.
    """
    real_poles = np.asarray(real_poles, dtype=float)
    real_poles = np.sort(real_poles)

    doubles = []
    i = 0
    n = len(real_poles)

    while i < n - 1:
        if abs(real_poles[i+1] - real_poles[i]) < tol:
            # representative value (average of cluster)
            cluster = [real_poles[i]]
            j = i + 1
            while j < n and abs(real_poles[j] - real_poles[i]) < tol:
                cluster.append(real_poles[j])
                j += 1
            doubles.append(np.mean(cluster))
            i = j
        else:
            i += 1

    return doubles

###########################################################################################
###########################################################################################
def U(t):
    """
    Unit step function.

    Parameters
    ----------
    t : array_like

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    u = np.zeros_like(t)
    u[t >= 0] = 1
    return u


###########################################################################################
###########################################################################################
def legend_best_combined(ax, candidates=None,
                         w_text=10.0, w_data=1.0,
                         **legend_kwargs):
    """
    Place legend minimizing overlap with both text (incl. AnchoredText)
    and plotted data, similar to loc='best' but text aware.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    candidates : iterable of str
        Legend locations to consider.
    w_text : float
        Weight for text overlap (large).
    w_data : float
        Weight for data overlap (smaller).
    **legend_kwargs :
        Passed to ax.legend().

    Returns
    -------
    legend : matplotlib.legend.Legend
    """
    if candidates is None:
        candidates = [
            "upper right",
            "upper left",
            "lower left",
            "lower right",
        ]

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ---- text / annotation bboxes ----
    text_bboxes = []

    for t in ax.texts:
        text_bboxes.append(t.get_window_extent(renderer))

    for a in ax.artists:
        if hasattr(a, "get_window_extent"):
            try:
                text_bboxes.append(a.get_window_extent(renderer))
            except Exception:
                pass

    # ---- data bboxes (lines, patches, collections) ----
    data_bboxes = []

    for line in ax.lines:
        data_bboxes.append(line.get_window_extent(renderer))

    for p in ax.patches:
        data_bboxes.append(p.get_window_extent(renderer))

    for c in ax.collections:
        data_bboxes.append(c.get_window_extent(renderer))

    best_loc = None
    best_cost = np.inf

    for loc in candidates:
        leg = ax.legend(loc=loc, **legend_kwargs)
        fig.canvas.draw()
        leg_bbox = leg.get_window_extent(renderer)

        text_overlap = sum(leg_bbox.overlaps(bb) for bb in text_bboxes)
        data_overlap = sum(leg_bbox.overlaps(bb) for bb in data_bboxes)

        cost = w_text * text_overlap + w_data * data_overlap

        if cost < best_cost:
            best_cost = cost
            best_loc = loc

        leg.remove()

    return ax.legend(loc=best_loc, **legend_kwargs)

###########################################################################################
###########################################################################################
def plot_spec_region(ax, zeta, wn, wd, color='m', highlight_color='r', linestyle='--'):
    """
    Draws the damping/angle spec lines used in your plots, but using the
    current axis limits (no magic 20). Works for both full and zoomed axes.
    """
    # angle from geometry
    th = np.arctan2(wd, (zeta*wn))

    # current axis bounds
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # vertical damping line at real = -zeta*wn, extend it to the full vertical axis
    x_vert = -zeta * wn
    ax.plot([x_vert, x_vert], [ymin, ymax], color=color, linestyle=linestyle)

    # Angled lines: these were originally y = -x * tan(th) (so they pass through origin).
    # Build a line from left axis limit to the damping vertical line.
    x_line = np.array([xmin, x_vert])
    y_line = -x_line * np.tan(th)   # y = -x * tan(th)
    ax.plot(x_line, y_line, color=color, linestyle=linestyle)
    ax.plot(x_line, -y_line, color=color, linestyle=linestyle)  # symmetric lower branch

    # Short connectors from origin to damping line (0 -> -zeta*wn)
    x_conn = np.array([0.0, x_vert])
    y_conn = -x_conn * np.tan(th)
    ax.plot(x_conn, y_conn, color=color, linestyle=linestyle)
    ax.plot(x_conn, -y_conn, color=color, linestyle=linestyle)

    # Now draw the solid highlight (same geometry as above but solid and only the local segment)
    # Use the mph segment limited to the local spec (i.e., -wd..wd around the damping vertical)
    ax.plot([x_vert, x_vert], [-wd, wd], color=highlight_color, linestyle='-')

    # solid angled highlight between xmin and x_vert, but clip to visible y-range so we don't overdraw
    y_line_high = -x_line * np.tan(th)
    ax.plot(x_line, y_line_high, color=highlight_color, linestyle='-')
    ax.plot(x_line, -y_line_high, color=highlight_color, linestyle='-')

###########################################################################################
###########################################################################################
def clone_blank_axis(ax):
    # handle array of axes
    if isinstance(ax, np.ndarray):
        fig = ax.flat[0].figure
        shape = ax.shape
    else:
        fig = ax.figure
        shape = (1,)

    # new figure
    new_fig, new_ax = plt.subplots(*shape, figsize=fig.get_size_inches(), dpi=fig.dpi)

    # flatten for easy looping
    ax_list = ax.flat if isinstance(ax, np.ndarray) else [ax]
    new_ax_list = new_ax.flat if isinstance(new_ax, np.ndarray) else [new_ax]

    for a, na in zip(ax_list, new_ax_list):
        na.set_xlim(a.get_xlim())
        na.set_ylim(a.get_ylim())

        na.set_xscale(a.get_xscale())
        na.set_yscale(a.get_yscale())

        na.set_xlabel(a.get_xlabel())
        na.set_ylabel(a.get_ylabel())
        na.set_title(a.get_title())

        na.set_xticks(a.get_xticks())
        na.set_yticks(a.get_yticks())

        bm.nicegrid(na)

    return new_fig, new_ax

###########################################################################################
###########################################################################################
def compute_Nbar(G, K):
    ''' Find Nbar so that the steady state response to a step input is 1.0 
    for the SISO closed loop system using feedback gain u=-Kx.'''
    A, B, C = G.A, G.B, G.C
    X = np.linalg.solve(A - B @ K, B)
    Nbar = 1.0 / (-C @ X).item()
    return Nbar

###########################################################################################
###########################################################################################
class Step_info:
    def __init__(self, t, y, method=0, t0=0, SettlingTimeLimits=None, RiseTimeLimits=(0.1, 0.9)):
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.Yss = self.y[-1] # assumes Tf is large enough that this is true

        if SettlingTimeLimits is None:
            self.SettlingTimeLimits = [0.02]
        elif np.isscalar(SettlingTimeLimits):
            self.SettlingTimeLimits = [SettlingTimeLimits]
        else:
            self.SettlingTimeLimits = list(SettlingTimeLimits)        

        self.RiseTimeLimits = RiseTimeLimits
        sgnYss = np.sign(self.Yss.real) if np.isreal(self.Yss) else np.sign(self.Yss)

        self.Tr, self.Tr_values = rise_time(self.t, self.y, yss=self.Yss, limits=RiseTimeLimits, t0=t0)
        self.Ts = settling_time(self.t, self.y, tol=self.SettlingTimeLimits[0], t0=t0) 
        self.Mp, self.Tp = max_overshoot(self.t, self.y, self.Yss)

        # different assumptions can be used here to estimate these response parameters
        if method == 0:             # using Tp
            if self.Mp <= 0:
                self.zeta = np.nan
                self.wn = np.nan
            else:
                self.zeta = 1.0 / np.sqrt(1.0 + (np.pi / np.log(self.Mp)) ** 2)
                self.wn = np.pi / self.Tp / np.sqrt(1.0 - self.zeta ** 2)
        else: # using Ts
            q = self.Tp / np.pi / self.Ts if self.Ts != 0 else np.nan
            if self.SettlingTimeLimits[0] == 0.01:
                q *= 4.6
            else:
                q *= 4.0
            self.zeta = q / np.sqrt(1.0 + q ** 2) if not np.isnan(q) else np.nan
            self.wn = 4.0 / self.Ts / self.zeta if self.zeta != 0 else np.nan

    def printout(self, verbose=False):
        print(f"omega_n:\t{self.wn:.3f}")
        print(f"zeta   :\t{self.zeta:.3f}")
        print(f"Tr     :\t{self.Tr:.2f}s")
        print(f"Ts     :\t{self.Ts:.2f}s")
        print(f"Mp     :\t{self.Mp:.2f}")
        print(f"Tp     :\t{self.Tp:.2f}s")
        print(f"Yss    :\t{self.Yss:.2f}")
        if verbose:
            return SimpleNamespace(**{
            "Mp": self.Mp,
            "Tr": self.Tr,
            "Ts": self.Ts,
            "Tp": self.Tp,
            "Yss": self.Yss,
            })
        else:
            pass

    def nice_plot(self, ax=None, Tmax=None, Ymax=None, label=None, lc='b'):
        if Ymax is None:
            ylim = (np.floor(np.min(self.y)), np.ceil(10.0 * np.max(self.y)) / 10.0)
            Ymax = np.max(ylim)
        if Tmax is None:
            Tmax = np.max(self.t)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.t, self.y, color=lc, label=label)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Response")
        ax.set_title("Step Response")
        ax.set_ylim(0, Ymax)
        ax.set_xlim(0, Tmax)
        #ax.set_aspect('equal', adjustable='box')

        ax.axvline(x=self.Tr_values[0], ymax=0.1 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Tr_values[1], ymax=0.9 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Ts, ymax=self.Yss / Ymax * (1-self.SettlingTimeLimits[0]), c="grey", ls="dashed")
        ax.axvline(ymax=min((self.Yss * (1 + self.Mp)) / Ymax, 1.0), x=self.Tp, c="m", ls="dashed", lw=2)
        ax.axhline(y=(1 + self.SettlingTimeLimits[0]) * self.Yss, xmin=self.Ts / Tmax, c="grey", ls="dashed", lw=1)
        ax.axhline(y=(1 - self.SettlingTimeLimits[0]) * self.Yss, xmin=self.Ts / Tmax, c="grey", ls="dashed", lw=1)
        ax.plot((0, self.Tp), (self.Yss * (1 + self.Mp), self.Yss * (1 + self.Mp)), c="green", ls="dashed", lw=2)
        ax.text(min(self.Tr / 2, Tmax/2), 0.25 * self.Yss, f"Tr = {self.Tr:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Tp, Tmax/2), 0.75 * self.Yss, f"Tp = {self.Tp:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Ts, Tmax/2), 0.5 * self.Yss, f"Ts = {self.Ts:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Tp * 1.1, Tmax/2), min(self.Yss * (1 + self.Mp),0.8*Ymax), f"Mp = {self.Mp:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Ts, Tmax/2), min(self.Yss * 1.1,0.9*Ymax), rf"$e_{{ss}}$ = {1 - self.Yss:.3f}", fontsize=SMALL_SIZE, color="purple")

###########################################################################################
###########################################################################################
def max_overshoot(t, y, yss=None):
    """
    Compute maximum overshoot Mp and peak time Tp from step response.

    Returns
    -------
    Mp : float
        Maximum overshoot as a fraction (e.g. 0.15 for 15%)
    Tp : float
        Peak time (time at maximum overshoot)
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if yss is None:
        yss = y[-1]

    if yss == 0:
        return np.nan, np.nan

    # work in the direction of the step
    sgn = np.sign(yss)
    y_adj = sgn * y
    yss_adj = abs(yss)

    # peak relative to steady state
    idx_peak = np.argmax(y_adj)
    ymax = y_adj[idx_peak]

    Mp = (ymax - yss_adj) / yss_adj
    Mp = max(0.0, Mp)   # clip if no overshoot

    Tp = t[idx_peak] if Mp > 0 else np.nan

    return Mp, Tp

###########################################################################################
###########################################################################################
def settling_time(t, y, tol=0.02, t0=0):
    """
    Compute Tol=2% settling time.
    Returns np.nan if never settles.
    """
    y = np.asarray(y)
    t = np.asarray(t)
    yss = y[-1]
    if yss == 0:
        return np.nan
    band = tol * abs(yss)
    err = np.abs(y - yss)
    outside = np.where(err > band)[0]     # indices where response is OUTSIDE the band
    if len(outside) == 0:
        return t[0]   # already settled
    last_outside = outside[-1]
    if last_outside == len(t) - 1:
        return np.nan  # never settles within simulation time
    return t[last_outside + 1]

###########################################################################################
###########################################################################################
def rise_time(t, y, yss=None, limits=(0.1, 0.9), t0=0.0):
    """
    Robust rise time computation using linear interpolation.

    Parameters
    ----------
    t : array_like
        Time vector
    y : array_like
        Response vector
    yss : float or None
        Steady-state value (default: y[-1])
    limits : tuple
        Fractional rise limits, e.g. (0.1, 0.9)
    t0 : float
        Time offset to subtract (default: 0)

    Returns
    -------
    Tr : float
        Rise time (NaN if undefined)
    (t_lo, t_hi) : tuple
        Times at lower and upper crossings
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if yss is None:
        yss = y[-1]

    if yss == 0:
        return np.nan, (np.nan, np.nan)

    sgn = np.sign(yss)
    y_adj = sgn * y
    yss_adj = abs(yss)

    y_lo = limits[0] * yss_adj
    y_hi = limits[1] * yss_adj

    # Find first crossing indices
    def crossing_time(level):
        idx = np.where(y_adj >= level)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        i = idx[0]
        # linear interpolation
        t1, t2 = t[i-1], t[i]
        y1, y2 = y_adj[i-1], y_adj[i]
        return t1 + (level - y1) * (t2 - t1) / (y2 - y1)

    t_lo = crossing_time(y_lo)
    t_hi = crossing_time(y_hi)

    if np.isnan(t_lo) or np.isnan(t_hi):
        return np.nan, (t_lo, t_hi)

    Tr = t_hi - t_lo
    return Tr, (t_lo - t0, t_hi - t0)