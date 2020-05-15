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

    # Shift and rescale
    y = (y - [0.003, 0.04]) * [10, 20]
    u = (u - 0.04) * 25
    
    # Add artificial noise
    np.random.seed(0)
    N = len(y)
    y_peak_to_peak = y.max(0) - y.min(0)
    y[:, 0] += y_peak_to_peak[0] * 1e-4 * np.random.randn(N)
    
    return t, u, y


if __name__ == '__main__':
    # Load experiment data
    t, u, y = load_data()
    symmodel = symfem.InnovationBalDTModel(nx=2, nu=1, ny=2)
    model = symmodel.compile_class()()
    problem = fem.InnovationBalDTProblem(model, y, u)
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = np.eye(2)
    var0['C'][:] = np.eye(2)
    var0['D'][:] = np.zeros((2,1))
    var0['Kp'][:] = np.eye(2)
    var0['x'][:] = y
    var0['isRp_tril'][symfem.tril_diag(2)] = 100
    var0['W_diag'][:] = 1
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['isRp_tril'][symfem.tril_diag(2)] = 0
    var_L['W_diag'][:] = 0
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    var_constr_L = problem.unpack_constraints(constr_L)
    var_constr_U = problem.unpack_constraints(constr_U)
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['isRp_tril'][:] = 1e-2
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-5)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)    
    xopt = opt['x']
    A = opt['A']
    B = opt['B']
    C = opt['C']
    D = opt['D']
    Kp = opt['Kp']
    W = np.diag(opt['W_diag'])
    ybias = opt['ybias']
    isRp = symfem.tril_mat(model.ny, opt['isRp_tril'])
    yopt = model.output(xopt, u, C, D, ybias)
    eopt = y - yopt
