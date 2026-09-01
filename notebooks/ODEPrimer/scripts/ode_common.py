from pathlib import Path

import numpy as np
import sympy as sym


t = sym.symbols("t")
y = sym.symbols("y", cls=sym.Function)(t)


def solve_ode(A=None, w=0, Y0=None):
    """Return a callable solution for first- or second-order linear ODEs."""
    if A is None:
        A = [1, 1, 1]
    if Y0 is None:
        Y0 = [0, 0]

    if len(A) == 3:
        ode = A[0] * y.diff(t, t) + A[1] * y.diff(t) + A[2] * y - w
        sol = sym.dsolve(
            ode,
            ics={y.subs(t, 0): Y0[0], sym.diff(y, t).subs(t, 0): Y0[1]},
        )
    else:
        ode = A[0] * y.diff(t) + A[1] * y - w
        sol = sym.dsolve(ode, ics={y.subs(t, 0): Y0[0]})

    return sym.lambdify([t], sol.rhs, modules=["numpy"])


def default_inputs(phi=0.0):
    we = 3 * sym.exp(-3 * t)
    wc = 3 * sym.cos(3 * t + phi)
    return we, wc


def output_dir():
    out = Path(__file__).resolve().parent.parent / "figs"
    out.mkdir(parents=True, exist_ok=True)
    return out
