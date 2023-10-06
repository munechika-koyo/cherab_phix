import numpy as np
from matplotlib import pyplot as plt

from cherab.phix.inversion import GCV, Lcurve

if __name__ == "__main__":
    # define true solution
    def x_0_func(t):
        return 2.0 * np.exp(-6.0 * (t - 0.8) ** 2) + np.exp(-2.0 * (t + 0.5) ** 2)

    # define Kernel of Fredholm integral equation
    def kernel(s, t):
        u = np.pi * (np.sin(s) + np.sin(t))
        if u == 0:
            return np.cos(s) + np.cos(t)
        else:
            return (np.cos(s) + np.cos(t)) * (np.sin(u) / u) ** 2

    # data set
    t = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=100)
    s = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=100)
    x_0 = x_0_func(t)

    # Operator matrix
    A = np.zeros((s.size, t.size))
    A = np.array([[kernel(i, j) for j in t] for i in s])
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5
    A *= np.abs(t[1] - t[0])
    # print(f"cond(A) = {np.linalg.cond(A)}")
    u, sigma, vh = np.linalg.svd(A, full_matrices=False)

    # mesured exact unperturbed data and added white noise
    b_0 = A.dot(x_0)
    rng = np.random.default_rng()
    b_noise = rng.normal(0, 1.0e-4, b_0.size)
    b = b_0 + b_noise

    # inversion process
    bounds = (-20.0, 2.0)

    print("L-curve method")
    lcurve = Lcurve(sigma, u, vh.T, data=b)

    # solve inverse problem using `optimize` method
    sol1 = lcurve.optimize(itemax=4, bounds=bounds)
    lambda1 = lcurve.lambda_opt
    print(f"`optimize` method result :{lcurve.lambda_opt = :.3e}")

    # solve inverse problem using `solve` method
    sol2, status = lcurve.solve(bounds=bounds, niter=10, disp=True)
    print(status)
    print(f"`solve` method result :{lcurve.lambda_opt = :.3e}")

    # plot L-curve
    _, ax = lcurve.plot_L_curve(bounds=bounds, n_beta=500)
    ax.scatter(lcurve.residual_norm(lambda1), lcurve.regularization_norm(lambda1), c="g", marker="x", label=f"$\\lambda_1$ = {lambda1:.2e}")
    ax.legend()

    # plot curvature
    _, ax = lcurve.plot_curvature(bounds=(-9, -7), n_beta=500)
    ax.scatter(lambda1, lcurve.curvature(lambda1), c="g", marker="x", label=f"$\\lambda_1$ = {lambda1:.2e}")
    print("----------------------------------------")

    print("GCV method")
    gcv = GCV(sigma, u, vh.T, data=b)

    # solve inverse problem using `optimize` method
    sol3 = gcv.optimize(itemax=4, bounds=bounds)

    # solve inverse problem using `solve` method
    sol4 = gcv.solve(bounds=bounds, disp=True)

    # plot GCV
    gcv.plot_gcv(bounds=bounds, n_beta=500)

    plt.show()
