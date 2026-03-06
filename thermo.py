import cantera as ct
import torch


def calculate_dT(TY, dY, gas):
    
    solArr = ct.SolutionArray(gas, shape=(TY.shape[0],))

    T = TY[:,0]
    Y = TY[:,1:]

    solArr.TPY = T, ct.one_atm, Y

    molar_cp_sp = torch.tensor(solArr.partial_molar_cp)
    molar_h_sp = torch.tensor(solArr.partial_molar_enthalpies)

    M = torch.tensor(gas.molecular_weights)

    return (torch.sum(- molar_h_sp * 1/M * dY, dim=1)
            / torch.sum(molar_cp_sp * 1/M * Y, dim=1))