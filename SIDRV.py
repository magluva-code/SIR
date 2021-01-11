#
# Equations: S' = beta(t)*S*I - p(t)*S
#            I' = beta(t)*S*I - nu(t)*I - ro(t) * I
#            D' = ro(t)*I
#            R' = nu(t)*I
#            V' = p(t)*S
#

import SIR as parent
import numpy as np


class ProblemSIDRV(parent.ProblemSIR):
    def __init__(self, beta, nu, S0, I0, R0, T, ro, D0, p, V0):
        super(ProblemSIDRV, self).__init__(beta, nu, S0, I0, R0, T)
        if isinstance(ro, (float, int)):
            self.ro = lambda t: ro
        elif callable(ro):
            self.ro = ro
        else: raise TypeError("Type {0} for ro is not supported".format(type(ro)))
        if isinstance(p, (float, int)):
            self.p = lambda t: p
        elif callable(p):
            self.p = p
        else: raise TypeError("Type {0} for ro is not supported".format(type(p)))
        self.D0 = D0
        self.V0 = V0
    
    def __call__(self, u, t):
        S, I , D, R, V = u
        return np.array([-self.beta(t)*S*I - self.p(t)*S,
                        self.beta(t)*S*I - self.nu(t)*I - self.ro(t)*I,
                        self.ro(t)*I,
                        self.nu(t)*I,
                        self.p(t)*S])


class SolverSIDRV(parent.SolverSIR):
    def __init__(self, problem, dt):
        self.problem = problem
        self.dt = dt

    def solve(self, method=parent.RK4):
        n = int(self.problem.T/self.dt)
        t = np.linspace(0, self.problem.T, n, True)
        solver = method(self.problem)
        solver.set_init_cnd([self.problem.S0, self.problem.I0, self.problem.D0, self.problem.R0, self.problem.V0])
        u, self.t = solver.solve(t)
        self.S, self.I, self.D, self.R, self.V = u[:, 0], u[:, 1], u[:, 2], u[:, 3], u[:, 4]

    def show_plot(self, style="bmh"):
        import matplotlib.pyplot as plt
        import matplotlib.style as plt_style
        # Setting up figure
        plt_style.use(style)
        fig = plt.figure()
        # Setting up axes object
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("S.I.D.R.V Plot")
        ax.set_xlabel("days")
        ax.set_ylabel("population")
        # Ploting SIDRV against time.
        ax.plot(self.t, self.S)
        ax.plot(self.t, self.I)
        ax.plot(self.t, self.D)
        ax.plot(self.t, self.R)
        ax.plot(self.t, self.V)
        # Max infected:
        for i, x in enumerate(self.I):
            if x == np.max(self.I):
                idx = i; break
        ax.plot(self.t, [self.I[idx] for i in self.t], color="green", linestyle=':')
        ax.legend(["Suseptable", "Infectious", "Deceased", "Recovered", "Vaccinated"])
        # Visualizing
        return plt.show()

