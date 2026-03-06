from typing import Optional, Callable, Union

import numpy as np
import cantera as ct
import pandas as pd

from scipy.integrate import solve_ivp


def hr_isobaric_ode(t, y, gas, p):

    # update current gas state according to solution state
    gas.TPY = y[0], p, y[1:]

    # get new dT/dt and dY/dt values
    ydot = np.zeros(gas.n_species + 1)
    ydot[0] = gas.heat_release_rate / (gas.density_mass * gas.cp_mass)
    ydot[1:] = gas.net_production_rates * gas.molecular_weights / gas.density_mass

    return ydot

# def hr_isobaric_ode_ml(y, model):

#     return model(y.reshape(-1, 1))

    
def solve_hr(
        gas: ct.Solution,
        ic: dict[str, float],
        t_span: tuple[float, float],
        solver: str,
        dt_max: float = np.inf,
        events: Optional[Union[Callable, list[Callable]]] = None,
        dt_eval: Optional[float] = None, 
    ) -> tuple[np.ndarray]:

    if dt_eval is not None:
        t_eval = np.array(np.arange(0, t_span[1], dt_eval))
    else:
        t_eval = None

    y0 = np.concatenate((np.array(ic["T"]).reshape(-1), np.array(ic["Y"])))

    rhs = hr_isobaric_ode

    sol = solve_ivp(rhs, t_span, y0, args=(gas, ic["p"]),
                    events=events, method=solver, max_step=dt_max, t_eval=t_eval, rtol=1e-8, atol=1e-12)

    return (1e3*sol.t.transpose().reshape(-1, 1),
            sol.y[0, :].transpose().reshape(-1, 1),
            sol.y[1:, :].transpose())

# def solve_hr_ml(
#         model,
#         gas: ct.Solution,
#         ic: dict[str, float],
#         t_span: tuple[float, float],
#         dt: float,
#     ):

#     rhs = hr_isobaric_ode_ml

#     n_steps = int((t_span[1] - t_span[0]) / dt) + 1

#     n_vars = 1 + ic["Y"].shape[0]

#     t = np.zeros((n_steps,))
#     y = np.zeros((n_steps, n_vars))

#     y[0,0] = ic["T"]
#     y[0,1:] = ic["Y"]

#     for j in range(n_steps):

#         t[j] = t[j-1] + dt
#         y[j, :] = y[j-1, :] + rhs(y[j-1, :], model) * dt
    
#     return y