"""This code doesn't use python logging, instead it uses an empty list called LOG
and appends output to it"""

import numpy as np
import pandas as pd

from simopt.solvers.astrodf import ASTRODF

LOG = []
STATE = {"mrep": -1}


def reset_log():
    LOG.clear()
    STATE["mrep"] = -1


def log_dataframe():
    return pd.DataFrame(LOG)


class LoggingASTRODF(ASTRODF):

    def _initialize_solving(self, ctx):
        super()._initialize_solving(ctx)
        STATE["mrep"] += 1

    def construct_model(self, ctx):
        delta_in = self.delta_k
        out = super().construct_model(ctx)
        fval, y_var, q, grad, hessian, solns = out

        m = 2 * self.problem.dim + 1

        for i, s in enumerate(solns[-m:]):
            LOG.append({
                "mrep": STATE["mrep"],
                "iter": self.iteration_count,
                "delta_in": delta_in,
                "incumbent_x": tuple(np.ravel(self.incumbent_x)),
                "point_idx": i,
                "x": tuple(np.ravel(s.x)),
                "F_hat": float(np.ravel(fval[i])[0]),
                "sample_size": int(s.n_reps),
                "var_F": float(np.ravel(s.objectives_var)[0]),
                "model_grad": tuple(np.ravel(grad)),
                "model_hessian_diag": tuple(np.ravel(hessian)),
            })
        return out