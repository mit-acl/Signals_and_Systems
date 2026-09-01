import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ode_common import default_inputs, output_dir, solve_ode


def main():
    phi = 0.0
    we, wc = default_inputs(phi)

    Y0 = [1, 0]
    A = [1, 3, 2]

    W0 = lambda t: 0.0
    W1 = lambda t: 1.0
    We = lambda t: 3.0 * np.exp(-3.0 * t)
    Wc = lambda t: 3.0 * np.cos(3.0 * t)

    def odes(t, Y, W):
        y, v = Y
        dvdt = W(t) - A[1] / A[0] * v - A[2] / A[0] * y
        return [v, dvdt]

    t_span = [0, 10]
    t_eval = np.linspace(t_span[0], t_span[1], 100)

    solution_0 = solve_ivp(odes, t_span, Y0, t_eval=t_eval, args=(W0,))
    solution_1 = solve_ivp(odes, t_span, Y0, t_eval=t_eval, args=(W1,))
    solution_e = solve_ivp(odes, t_span, Y0, t_eval=t_eval, args=(We,))
    solution_c = solve_ivp(odes, t_span, Y0, t_eval=t_eval, args=(Wc,))

    fun_0 = solve_ode(A=A, w=0, Y0=Y0)
    fun_1 = solve_ode(A=A, w=1, Y0=Y0)
    fun_2 = solve_ode(A=A, w=we, Y0=Y0)
    fun_3 = solve_ode(A=A, w=wc, Y0=Y0)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.plot(solution_0.t, solution_0.y[0], "bs", label="Numerical homogeneous", ms=4, markevery=4)
    ax.plot(solution_1.t, solution_1.y[0], "rs", label="Numerical w(t)=1", ms=4, markevery=4)
    ax.plot(solution_e.t, solution_e.y[0], "gs", label="Numerical 3exp(-3t)", ms=4, markevery=4)
    ax.plot(solution_c.t, solution_c.y[0], "ms", label="Numerical 3cos(3t)", ms=4, markevery=4)

    ax.plot(t_eval, fun_0(t_eval), "b-", label="Analytical homogeneous", lw=3)
    ax.plot(t_eval, fun_1(t_eval), "r-", label=r"Analytical $w(t)=1$")
    ax.plot(t_eval, fun_2(t_eval), "g--", label=r"Analytical $3exp^{-3t}$", lw=3)
    ax.plot(t_eval, fun_3(t_eval), "m", label=r"Analytical $3cos(3t)$")

    ax.text(4, 0.75, "y(0) = " + str(Y0[0]) + r", $\dot y(0) =$ " + str(Y0[1]))
    ax.set_xlabel("Time t")
    ax.set_ylabel("Solution")
    ax.set_title("Second Order: $\\ddot y(t) + 3\\dot y(t) + 2y(t) = w(t)$")
    ax.grid(True)
    ax.legend()

    out = output_dir()
    fig.savefig(out / "second_order_numerical_compare.png", dpi=300)
    fig.savefig(out / "second_order_numerical_compare.pdf")
    plt.show()


if __name__ == "__main__":
    main()
