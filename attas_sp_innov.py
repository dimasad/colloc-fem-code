"""Tests with the ATTAS aircraft short-period mode estimation."""


import importlib
import os

import numpy as np
import scipy.io
import sympy
import sym2num.model

import fem
import symfem


# Reload modules for testing
for m in (fem, symfem):
    importlib.reload(m)


def load_data():
    # Retrieve data
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv1.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv1'][30:-30]
    t = data[:, 0] - data[0, 0]
    u = data[:, [21]] * d2r
    y = data[:, [7, 12]] * d2r
    
    # Add artificial noise
    np.random.seed(0)
    N = len(y)
    y_peak_to_peak = y.max(0) - y.min(0)
    y[:, 0] += y_peak_to_peak[0] * 1e-3 * np.random.randn(N)
    
    return t, u, y


if __name__ == '__main__':
    # Load experiment data
    t, u, y = load_data()
    symmodel = symfem.InnovationDTModel(nx=2, nu=1, ny=2)
    model = symmodel.compile_class()()
    problem = fem.InnovationDTProblem(model, y, u)

    Aoem = np.array([[0.92092623, -0.18707263], [0.04193602, 0.97679607]])
    Boem = np.array([[-0.27215983], [0.01984017]])

    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = np.eye(model.nx)
    #var0['B'][:] = Boem
    var0['C'][:] = np.eye(2)
    var0['D'][:] = np.zeros((2,1))
    var0['L'][:] = np.eye(model.nx) * 0.1
    var0['x'][:] = y - y[0]
    var0['ybias'][:] = y[0]
    var0['isRp_tril'][symfem.tril_diag(2)] = 1e3
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    #var_U['isRp_tril'][symfem.tril_diag(2)] = 1e5
    var_L['C'][:] = np.eye(2)
    var_U['C'][:] = np.eye(2)
    var_L['D'][:] = np.zeros((2,1))
    var_U['D'][:] = np.zeros((2,1))
    #var_L['L'][:] = np.zeros((2,2))
    #var_U['L'][:] = np.zeros((2,2))
    #var_L['isRp_tril'][~symfem.tril_diag(2)] = 0
    #var_U['isRp_tril'][~symfem.tril_diag(2)] = 0
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    var_constr_scale['innovation'][:] = 100
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['isRp_tril'][:] = 1e-2
    var_scale['e'][:] = 100
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)    
    xopt = opt['x']
    Aopt = opt['A']
    Bopt = opt['B']
    Copt = opt['C']
    Dopt = opt['D']
    Lopt = opt['L']
    ybiasopt = opt['ybias']
    isRpopt = symfem.tril_mat(model.ny, opt['isRp_tril'])
    yopt = xopt @ Copt.T + u @ Dopt.T + ybiasopt
    eopt = opt['e']
