import numpy as np
import matplotlib.pyplot as plt

from ode_common import default_inputs, output_dir, solve_ode


def main():
    phi = 0.0
    we, wc = default_inputs(phi)
    Y0 = [1, 1]
    A = [1, 2, 1]

    fun_0 = solve_ode(A=A, w=0, Y0=Y0)
    fun_1 = solve_ode(A=A, w=1, Y0=Y0)
    fun_2 = solve_ode(A=A, w=we, Y0=Y0)
    fun_3 = solve_ode(A=A, w=wc, Y0=Y0)

    tt = np.arange(0, 10, 0.01)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.plot(tt, fun_0(tt), "b-", label="homogeneous", lw=3)
    ax.plot(tt, fun_1(tt), "r-", label=r"$w(t)=1$")
    ax.plot(tt, fun_2(tt), "g--", label=r"$3exp^{-3t}$", lw=3)
    ax.plot(tt, fun_3(tt), "m", label=r"$3cos(3t)$")
    ax.text(4, 0.75, "y(0) = " + str(Y0[0]) + r", $\dot y(0) =$ " + str(Y0[1]))
    ax.set_xlabel("Time")
    ax.legend()
    ax.set_title("Second Order/Repeated: $\\ddot y(t) + 2\\dot y(t) + y(t) = w(t)$")

    out = output_dir()
    fig.savefig(out / "second_order_repeated.png", dpi=300)
    fig.savefig(out / "second_order_repeated.pdf")
    plt.show()


if __name__ == "__main__":
    main()
