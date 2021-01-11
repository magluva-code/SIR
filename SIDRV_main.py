import SIDRV as model

def test_plot():
    # Example variables
    S0 = 1500
    I0 = 1.4
    D0 = 0
    R0 = 0
    V0 = 0
    p = 0.01
    nu = 0.1
    # Making beta callable so we get desired change in value for t>12
    beta = lambda t: 0.0005 if t <= 12 else 0.0001 # beta = 0.0005
    ro = 0.04
    T = 60
    dt = 0.5
    # Creating instance of ProblemSIR
    problem = model.ProblemSIDRV(beta, nu, S0, I0, R0, T, ro, D0, p, V0)
    # Passing object of ProblemSir to instance of SolverSIR
    solver = model.SolverSIDRV(problem, dt)
    # Solving
    solver.solve()
    # Plotting
    solver.show_plot()

def main():
    test_plot()

if __name__ == "__main__":
    main()
