#
# Equations: S' = beta(t)*S*I 
#            I' = beta(t)*S*I - nu(t)*I - ro(t) * I
#            D' = ro(t)*I
#            R' = nu(t)*I
#

import SIR as parent
import numpy as np

class ProblemSIDR(parent.ProblemSIR):
    def __init__(self, beta, nu, S0, I0, R0, T, ro, D0):
        super(ProblemSIDR, self).__init__(beta, nu, S0, I0, R0, T)
        if isinstance(ro, (float, int)):
            self.ro = lambda t: ro
        elif callable(ro):
            self.ro = ro
        else: raise TypeError("Type {0} for ro is not supported".format(type(ro)))
        self.D0 = D0

    def __call__(self, u, t):
        S, I, D, R = u
        return np.array([-self.beta(t)*S*I,
                         self.beta(t)*S*I - self.nu(t)*I - self.ro(t) * I,
                         self.ro(t)*I,
                         self.nu(t)*I])


class SolverSIDR(parent.SolverSIR):
    def __init__(self, problem, dt):
        self.problem = problem
        self.dt = dt

    def solve(self, method=parent.RK4):
        n = int(self.problem.T/self.dt)
        t = np.linspace(0, self.problem.T, n, True)
        solver = method(self.problem)
        solver.set_init_cnd([self.problem.S0, self.problem.I0, self.problem.D0, self.problem.R0])
        u, self.t = solver.solve(t)
        self.S, self.I, self.D, self.R = u[:,0], u[:,1], u[:,2], u[:,3]

    def show_plot(self, betavalue=None, nuvalue=None, rovalue=None):
        import matplotlib.pyplot as plt
        import matplotlib.style as style
        # Setting up figure
        style.use("bmh")
        fig = plt.figure()
        # Setting up axes object
        ax = fig.add_subplot(1, 1, 1)
        if betavalue and nuvalue and rovalue is not None:
            ax.set_title("S.I.D.R Plot\nbeta: {}, nu: {}, ro {}".format(betavalue, nuvalue, rovalue))
        else:
            ax.set_title("S.I.D.R Plot")
        ax.set_xlabel("days")
        ax.set_ylabel("population")
        ax.plot(self.t, self.S)
        ax.plot(self.t, self.I)
        ax.plot(self.t, self.D)
        ax.plot(self.t, self.R)
         # Max infected:
        for i, x in enumerate(self.I):
            if x == np.max(self.I):
                idx = i; break
        ax.plot(self.t, [self.I[idx] for i in self.t], color="green", linestyle=':')
        ax.legend(["Suseptable", "Infectious", "Deceased", "Recovered"])
        # Visualizing
        return plt.show()


if __name__ == "__main__":
    # Example variables
    S0 = 1500
    I0 = 1.4
    R0 = 0
    D0 = 0
    nu = 0.1
    # Making beta callable so we get desired change in value for t>12
    beta = lambda t: 0.0005 if t <= 12 else 0.0001 # beta = 0.0005
    ro = 0.04
    T = 60
    dt = 0.5
    # Creating instance of ProblemSIR
    problem = ProblemSIDR(beta, nu, S0, I0, R0, T, ro, D0)
    # Passing object of ProblemSir to instance of SolverSIR
    solver = SolverSIDR(problem, dt)
    # Solving
    solver.solve()
    # Plotting
    solver.show_plot(betavalue="1.4 to 1 where t=12", nuvalue=nu, rovalue=ro)