import sympy as sym

from ode_common import t, y


def main():
    A = [1, 3, 2]

    g0 = A[0] * y.diff(t, t) + A[1] * y.diff(t) + A[2] * y - sym.DiracDelta(t)
    impulse = sym.dsolve(g0, ics={y.subs(t, 0): 0, sym.diff(y, t).subs(t, 0): 0})

    g1 = A[0] * y.diff(t, t) + A[1] * y.diff(t) + A[2] * y - sym.Heaviside(t)
    step = sym.dsolve(g1, ics={y.subs(t, 0): 0, sym.diff(y, t).subs(t, 0): 0})

    print("Impulse response solution:")
    sym.pprint(impulse)
    print("\nStep response solution:")
    sym.pprint(step)


if __name__ == "__main__":
    main()
