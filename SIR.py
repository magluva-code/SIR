#
# Equations: S' = beta(t)*S*I 
#            I' = beta(t)*S*I - nu(t)*I 
#            R' = nu(t)*I
#

import numpy as np
import matplotlib.pyplot as plt


class ODE(object):
    def __init__(self, f):
        if not callable(f): raise TypeError("f is not callable.")
        self.f = lambda u, t: np.asarray(f(u, t), np.float64)

    def set_init_cnd(self, u0):
        if isinstance(u0, (float, int)):
            u0 = np.float64(u0)
            self.size_u = 1
        else:
            u0 = np.asarray(u0)
            self.size_u = np.shape(u0)[0]
        self.u0 = u0
    
    def advance(self):
        raise NotImplementedError("This class cannot be used directly")
    
    def solve(self, tp):
        self.t = np.asarray(tp)
        n = self.t.size
        if self.size_u == 1:
            self.u = np.zeros(n)
        else:
            self.u = np.zeros((n, self.size_u))
        self.u[0] = self.u0
        for k in range(n-1):
            self.k = k
            self.u[k+1] = self.advance()
        return self.u[:k+2], self.t[:k+2]


class RK4(ODE):
    def advance(self):
        u, f, k ,t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]
        K1 = dt*f(u[k], t[k])
        K2 = dt*f(u[k] + (1/2)*K1, t[k] + (1/2)*dt)
        K3 = dt*f(u[k] + (1/2)*K2, t[k] + (1/2)*dt)
        K4 = dt*f(u[k] + K3, t[k] + dt)
        _u = u[k] + (1/6)*(K1 + 2*K2 + 2*K3 + K4)
        return _u



class ProblemSIR(object):
    def __init__(self, beta, nu, S0, I0, R0, T):
        if isinstance(beta, (float, int)):
            self.beta = lambda t: beta
        elif callable(beta):
            self.beta = beta
        else: raise TypeError("Type {0} for beta is not supported".format(type(beta)))
        if isinstance(nu, (float, int)):
            self.nu = lambda t: nu
        elif callable(nu):
            self.nu = nu
        else: raise TypeError("Type {0} for nu is not supported".format(type(nu)))
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.T = T

    def __call__(self, u, t):
        S, I, R = u
        return np.array([-self.beta(t)*S*I, self.beta(t)*S*I - self.nu(t)*I, self.nu(t)*I])


class SolverSIR(object):
    def __init__(self, problem, dt):
        self.problem = problem
        self.dt = dt
    
    def solve(self, method=RK4):
        n = int(self.problem.T/self.dt)
        t = np.linspace(0, self.problem.T, n, True)
        solver = method(self.problem)
        solver.set_init_cnd([self.problem.S0, self.problem.I0, self.problem.R0])
        u, self.t = solver.solve(t)
        self.S, self.I, self.R = u[:, 0], u[:, 1], u[:, 2]
    
    def show_plot(self, betavalue=None, nuvalue=None):
        import matplotlib.style as style
        # Setting up figure
        style.use("bmh")
        fig = plt.figure()
        # Setting up axes object
        ax = fig.add_subplot(1, 1, 1)
        if betavalue and nuvalue is not None:
            ax.set_title("S.I.R Plot\nbeta: {}, v: {}".format(betavalue, nuvalue))
        else:
            ax.set_title("S.I.R Plot")
        ax.set_xlabel("days")
        ax.set_ylabel("population")
        ax.plot(self.t, self.S)
        ax.plot(self.t, self.I)
        #print(self.I[idx], self.t[idx])
        ax.plot(self.t, self.R)
         # Max infected:
        for i, x in enumerate(self.I):
            if x == np.max(self.I):
                idx = i; break
        ax.plot(self.t, [self.I[idx] for i in self.t], color="green", linestyle=':')
        ax.legend(["Suseptable", "Infectious", "Recovered", "maximum infectious"])
        # Visualizing
        return plt.show()


if __name__ == "__main__":
    # Example variables
    S0 = 1500
    I0 = 1
    R0 = 0
    v = 0.1
    # Making beta callable so we get desired change in value for t>12
    beta = lambda t: 0.0005 if t <= 12 else 0.0001 # beta = 0.0005
    T = 60
    dt = 0.5
    # Creating instance of ProblemSIR
    problem = ProblemSIR(beta, v, S0, I0, R0, T)
    # Passing object of ProblemSir to instance of SolverSIR
    solver = SolverSIR(problem, dt)
    # Solving
    solver.solve()
    # Plotting
    solver.show_plot(betavalue="0.0005 to 0,0001 where t=12", nuvalue=v)



