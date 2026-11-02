from __future__ import annotations

from typing import Annotated, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt import dsl
from simopt.base import (
    ConstraintType,
    Model,
    Problem,
    VariableType,
)
from simopt.input_models import Normal
from simopt.utils import override

# Like in simulation 
class QuadNoiseConfig(BaseModel):
    x: Annotated[
        tuple[float, ...],
        Field(default=(5.0,), description="decision variable"),
    ]
    noise_std: Annotated[
        float,
        Field(default=1e-8, ge=0.0, description="tiny noise standard deviation"),
    ]

class QuadNoiseModel(Model[QuadNoiseConfig]):
    """x^2 + 1 evaluated with tiny stochastic noise."""

    class_name_abbr: ClassVar[str] = "QUADNOISE"
    class_name: ClassVar[str] = "x^2 + 1 with tiny noise"
    config_class: ClassVar[type[QuadNoiseConfig]] = QuadNoiseConfig
    n_rngs: ClassVar[int] = 1
    n_responses: ClassVar[int] = 1
    
    def __init__(self, fixed_factors: dict | None = None) -> None:
        super().__init__(fixed_factors)
        self.noise_model = Normal()
        
    def replicate(self, factors: QuadNoiseConfig, rngs: list[MRG32k3a]) -> tuple[dict, dict]:
        x = np.asarray(factors.x, dtype=float)
        f = float(np.sum(x**2) + 1.0)

        # Tiny noise: set noise_std = 0.0 for exact deterministic behavior.
        noise = self.noise_model.random(rngs[0], mu=0.0, sigma=factors.noise_std)
        noisy_value = f + noise

        responses = {"est_f(x)": noisy_value}
        gradients = {"est_f(x)": {"x": 2.0 * x}}
        return responses, gradients
    
    #Similar to example.py in models.   
class QuadNoiseProblemConfig(BaseModel):
    initial_solution: Annotated[
        tuple[float, ...],
        Field(default=(5.0,), description="initial point"),
    ]
    budget: Annotated[
        int,
        Field(default=1000, gt=0, description="max simulation budget"),
    ]
    
class QuadNoiseProblem(Problem):
    """Minimize x^2 + 1 under noisy stochastic replications."""

    class_name_abbr: ClassVar[str] = "QUADNOISE"
    class_name: ClassVar[str] = "Minimize x^2 + 1 with zero noise"
    config_class: ClassVar[type[BaseModel]] = QuadNoiseProblemConfig
    model_class: ClassVar[type[Model]] = QuadNoiseModel
    n_objectives: ClassVar[int] = 1
    n_stochastic_constraints: ClassVar[int] = 0
    minmax: ClassVar[tuple[int, ...]] = (-1,)
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_available: ClassVar[bool] = True
    model_default_factors: ClassVar[dict] = {}
    model_decision_factors: ClassVar[set[str]] = {"x"}
    
    @property
    def optimal_value(self) -> float | None:
        return 1.0
    
    @property
    def optimal_solution(self) -> tuple | None:
        return (0.0,)
    
    @override
    def build(self) -> dsl.Model:
        problem = dsl.Model()
        x0 = self.factors["initial_solution"]
        x = problem.add_continuous_vector(
            lb=-10.0, ub=10.0, shape=(len(x0),), initial=x0,
        )
        simulation = self.add_simulation(problem, {"x": x})
        problem.minimize(dsl.mean(simulation.metric("est_f(x)")))
        return problem
    
    def vector_to_factor_dict(self, vector: tuple) -> dict:
        return {"x": vector[:]}
    
    def get_random_solution(self, rand_sol_rng: MRG32k3a) -> tuple:
        # unchanged from the example
        return tuple(
            rand_sol_rng.mvnormalvariate(
                mean_vec=[0.0] * self.dim,
                cov=np.eye(self.dim).tolist(),
                factorized=False,
            )
        )