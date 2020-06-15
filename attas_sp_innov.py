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
    yshift = np.r_[-0.003, -0.04]
    yscale = np.r_[10.0, 20.0]
    ushift = np.r_[0.04]
    uscale = np.r_[25.0]
    
    y = (y + yshift) * yscale
    u = (u + ushift) * uscale
    
    # Add artificial noise
    np.random.seed(0)
    N = len(y)
    y_peak_to_peak = y.max(0) - y.min(0)
    y[:, 0] += y_peak_to_peak[0] * 1e-3 * np.random.randn(N)
    
    return t, u, y, yshift, yscale, ushift, uscale


def dt_eem(x, u, y):
    N, nx = x.shape
    _, nu = u.shape
    _, ny = y.shape
    
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    C = np.zeros((ny, nx))
    D = np.zeros((ny, nu))
    
    for i in range(nx):
        psi = np.zeros((N-1, nx+nu))
        psi[:, :nx] = x[:-1]
        psi[:, nx:] = u[:-1]
        est = np.linalg.lstsq(psi, x[1:, i], rcond=None)
        A[i, :] = est[0][:nx]
        B[i, :] = est[0][nx:]

    for i in range(ny):
        psi = np.zeros((N, nx+nu))
        psi[:, :nx] = x
        psi[:, nx:] = u
        est = np.linalg.lstsq(psi, y[:, i], rcond=None)
        C[i, :] = est[0][:nx]
        D[i, :] = est[0][nx:]
    
    return A, B, C, D


if __name__ == '__main__':
    nx = 2
    nu = 1
    ny = 2
    
    # Load experiment data
    t, u, y, yshift, yscale, ushift, uscale = load_data()
    symmodel = symfem.InnovationDTModel(nx=nx, nu=nu, ny=ny)
    model = symmodel.compile_class()()
    problem = fem.InnovationDTProblem(model, y, u)

    # Equation error method initial guess
    A0, B0, C0, D0 = dt_eem(y, u, y)
    
    # Previous sample predictor guess
    x0 = np.vstack((np.zeros(nx), y[:-1]))
    Rp0 = np.cov(y - x0, rowvar=0)
    sRp0 = np.linalg.cholesky(Rp0)
    e0 = np.linalg.solve(sRp0, (y - x0).T).T
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = np.eye(nx, nx)
    var0['B'][:] = np.zeros((2,1))
    var0['C'][:] = np.eye(2)
    var0['D'][:] = np.zeros((2,1))
    # var0['L'][:] = sRp0
    var0['x'][:] = x0
    var0['e'][:] = e0
    var0['sRp_tril'][:] = sRp0[np.tril_indices(nx)]
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['sRp_tril'][symfem.tril_diag(2)] = 1e-7
    #var_U['sRp_tril'][symfem.tril_diag(2)] = 1e5
    #var_L['sRp_tril'][~symfem.tril_diag(2)] = 0
    #var_U['sRp_tril'][~symfem.tril_diag(2)] = 0
    var_L['C'][:] = np.eye(2)
    var_U['C'][:] = np.eye(2)
    var_L['D'][:] = np.zeros((2,1))
    var_U['D'][:] = np.zeros((2,1))
    #var_L['L'][:] = np.zeros((2,2))
    #var_U['L'][:] = np.zeros((2,2))
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    var_constr_scale['innovation'][:] = 1e2
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['sRp_tril'][:] = 1e2
    var_scale['L'][:] = 1e2
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 25.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)    
    xopt = opt['x']
    A = opt['A']
    B = opt['B']
    C = opt['C']
    D = opt['D']
    L = opt['L']
    ybias = opt['ybias']
    sRp = symfem.tril_mat(opt['sRp_tril'])
    yopt = xopt @ C.T + u @ D.T + ybias
    eopt = opt['e']
