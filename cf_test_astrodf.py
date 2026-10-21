from simopt.experiment_base import ProblemSolver

exp = ProblemSolver(solver_name="ASTRODF", problem_name="CNTNEWS-1")
exp.run(n_macroreps=2)
exp.post_replicate(n_postreps=50)
print("done")