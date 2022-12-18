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
    b_noise = np.random.normal(0, 1.0e-4, b_0.size)
    b = b_0 + b_noise

    # === inversion process
    lcurve = Lcurve(sigma, u, vh, b)
    lcurve.lambdas = 10 ** np.linspace(-20, 2, 100)
    lcurve.optimize(4)
    lcurve.plot_L_curve()

    gcv = GCV(sigma, u, vh, b)
    gcv.lambdas = 10 ** np.linspace(-20, 2, 100)
    gcv.optimize()
    gcv.plot_gcv()

    plt.show()
