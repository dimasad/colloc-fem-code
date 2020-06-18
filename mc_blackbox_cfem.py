#!/usr/bin/env python3

"""Monte Carlo blackbox collocation-based FEM system id."""


import argparse
import importlib
import os
import pathlib

import numpy as np
import scipy.io
import sympy
import sym2num.model

import fem
import symfem


# Reload modules for testing
for m in (fem, symfem):
    importlib.reload(m)


class Model(symfem.MaximumLikelihoodDTModel, symfem.BalancedDTModel):
    generated_name = 'GeneratedBalancedMaximumLikelihoodModel'


class MLProblem(fem.MaximumLikelihoodDTProblem, fem.BalancedDTProblem):
    pass


def load_data(datafile):
    data = scipy.io.loadmat(datafile)
    
    # Retrieve data
    u = data['u']
    y = data['y']
    
    N = len(y)
    ue = u[N//2:]
    ye = y[N//2:]
    uv = u[:N//2]
    yv = y[:N//2]
    return uv, yv, ue, ye


def load_matlab_estimates(datafile):
    estfile = datafile.parent / ('estim_' + datafile.name)
    return scipy.io.loadmat(estfile)


def predict(mdl, y, u):
    A = mdl['A']
    B = mdl['B']
    C = mdl['C']
    D = mdl['D']
    try:
        L = mdl['L']
    except (KeyError, ValueError):
        sRp = symfem.tril_mat(mdl['sRp_tril'])
        Ln = mdl['Ln']
        L = Ln @ np.linalg.inv(sRp)
    
    nx = len(A)
    N = len(y)
    try:
        x0 = mdl['x0'].ravel()
    except (KeyError, ValueError):
        x0 = np.zeros(nx)
    
    x = np.tile(x0, (N, 1))
    e = np.empty_like(y)
    
    for k in range(N):
        e[k] = y[k] - C @ x[k] - D @ u[k]
        if k+1 < N:
            x[k+1] = A @ x[k] + B @ u[k] + L @ e[k]
    return x, e


def estimate(model, datafile, prob_type='bal', matlab_est=None):
    uv, yv, ue, ye = load_data(datafile)
    if prob_type == 'ml':
        problem = MLProblem(model, ye, ue)
    elif prob_type == 'bal':
        problem = fem.BalancedDTProblem(model, ye, ue)
    else:
        raise ValueError('Unknown prob_type')
    
    matlab_est = matlab_est or load_matlab_estimates(datafile)
    guess = matlab_est['guess'][0,0]
    A0 = guess['A']
    B0 = guess['B']
    C0 = guess['C']
    D0 = guess['D']
    x0, e0 = predict(guess, ye, ue)
    Rp0 = 1/len(e0) * e0.T @ e0
    sRp0 = np.linalg.cholesky(Rp0)
    en0 = np.linalg.solve(sRp0, e0.T).T
    W0 = np.diag(guess['gram'].ravel())
    sW0 = np.sqrt(W0)
    Ln0 = guess['L'] @ sRp0

    
    assert np.all(np.isfinite(W0))
    
    ctrl_bmat = np.c_[A0 @ sW0, B0]
    q, r = np.linalg.qr(ctrl_bmat.T)
    ctrl_orth0 = (q * np.diag(np.sign(r))).T

    obs_bmat = np.c_[A0.T @ sW0, C0.T]
    q, r = np.linalg.qr(obs_bmat.T)
    obs_orth0 = (q * np.diag(np.sign(r))).T
    
    nx = model.nx
    nu = model.nu
    ny = model.ny
    
    if prob_type == 'ml':
        sQ0 = np.eye(nx) * 1e-2
        Q0 = sQ0 ** 2
        sR0 = sRp0
        R0 = sR0 @ sR0.T
        Pp0 = scipy.linalg.solve_discrete_are(A0.T, C0.T, Q0, R0)
        sPp0 = np.linalg.cholesky(Pp0)
        
        corr_mat = np.block([[sR0, C0 @ sPp0], [np.zeros((nx,ny)), sPp0]])
        q,r = np.linalg.qr(corr_mat.T)
        s = np.sign(np.diag(r))
        corr_orth0 = (q * s).T
        sRp0 = (r.T * s)[:ny, :ny]
        Kn0 = (r.T * s)[ny:, :ny]
        sPc0 = (r.T * s)[ny:, ny:]
        Ln0 = A0 @ Kn0
        L0 = Ln0 @ np.linalg.inv(sRp0)
        
        pred_mat = np.block([[A0 @ sPc0, sQ0]])
        q,r = np.linalg.qr(pred_mat.T)
        s = np.sign(np.diag(r))
        pred_orth0 = (q*s).T
        
        mdl0 = dict(A=A0, B=B0, C=C0, D=D0, L=L0, x0=x0[0])
        x0, e0 = predict(mdl0, ye, ue)
        en0 = np.linalg.solve(sRp0, e0.T).T
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['A'][:] = A0
    var0['B'][:] = B0
    var0['C'][:] = C0
    var0['D'][:] = D0
    var0['Ln'][:] = Ln0
    var0['x'][:] = x0
    var0['en'][:] = en0
    var0['sRp_tril'][:] = sRp0[np.tril_indices(ny)]
    var0['sW_diag'][:] = np.diag(sW0)
    var0['ctrl_orth'][:] = ctrl_orth0
    var0['obs_orth'][:] = obs_orth0
    if prob_type == 'ml':
        var0['sQ_tril'][:] = sQ0[np.tril_indices(nx)]
        var0['sR_tril'][:] = sR0[np.tril_indices(ny)]
        var0['sPp_tril'][:] = sPp0[np.tril_indices(nx)]
        var0['sPc_tril'][:] = sPc0[np.tril_indices(nx)]
        var0['pred_orth'][:] = pred_orth0
        var0['corr_orth'][:] = corr_orth0
        var0['Kn'][:] = Kn0
    
    # Define bounds for decision variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['sRp_tril'][symfem.tril_diag(ny)] = 1e-7
    var_L['sW_diag'][:] = 0
    var_L['ybias'][:] = 0
    var_U['ybias'][:] = 0
    if prob_type == 'ml':
        var_L['sPp_tril'][symfem.tril_diag(nx)] = 0
        var_L['sPc_tril'][symfem.tril_diag(nx)] = 0
        var_L['sQ_tril'][symfem.tril_diag(nx)] = 1e-5
        var_L['sR_tril'][symfem.tril_diag(ny)] = 0
    
    # Define bounds for constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    var_constr_L = problem.unpack_constraints(constr_L)
    var_constr_U = problem.unpack_constraints(constr_U)
    
    # Define problem scaling
    obj_scale = -1.0
    constr_scale = np.ones(problem.ncons)
    var_constr_scale = problem.unpack_constraints(constr_scale)
    var_constr_scale['innovation'][:] = 1e2
    if prob_type == 'ml':
        var_constr_scale['pred_cov'][:] = 10
        var_constr_scale['corr_cov'][:] = 10
        var_constr_scale['kalman_gain'][:] = 10
    
    dec_scale = np.ones(problem.ndec)
    var_scale = problem.variables(dec_scale)
    var_scale['sRp_tril'][:] = 1e1
    var_scale['Ln'][:] = 1e1
    if prob_type == 'ml':
        var_scale['sPp_tril'][:] = 1e1
        var_scale['sPc_tril'][:] = 1e1
        var_scale['sQ_tril'][:] = 1e1
        var_scale['sR_tril'][:] = 1e1
        var_scale['Kn'][:] = 1e1
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 25.0)
        nlp.add_num_option('tol', 1e-5)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    opt['status'] = info['status']
    return opt
    

def get_model(config):
    nx = config['nx']
    nu = config['nu']
    ny = config['ny']
    
    modname = f'{Model.generated_name}_nx{nx}_nu{nu}_ny{ny}'
    try:
        mod = importlib.import_module(modname)        
        cls = getattr(mod, Model.generated_name)
        return cls()
    except ImportError:
        symmodel = Model(nx=nx, nu=nu, ny=ny)
        with open(f'{modname}.py', mode='w') as f:
            print(symmodel.print_code(), file=f)
        return get_model(config)


def cmdline_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'edir', nargs='?', default='data/mc_experim', 
        help='experiment directory',
    )
    return parser.parse_args()


def fit(e, y):
    from numpy.linalg import norm
    ymean = np.mean(y, axis=0)
    nrmse = norm(e, axis=0) / norm(y - ymean, axis=0)
    return 1 - nrmse


if __name__ == '__main__':
    args = cmdline_args()
    edir = pathlib.Path(args.edir)
    config = scipy.io.loadmat(edir / 'config.mat', squeeze_me=True)
    
    model = get_model(config)
    datafiles = sorted(edir.glob('exp*.mat'))
    
    msefile = edir / 'val_mse.txt'
    open(msefile, 'w').close()

    for datafile in datafiles:
        i = int(datafile.stem[3:])
        print('_' * 80)
        print('Experiment #', i, sep='')
        print('=' * 80)
        
        uv, yv, ue, ye = load_data(datafile)
        matlab_est = load_matlab_estimates(datafile)
        
        optbal = estimate(model, datafile, 'bal', matlab_est)
        optml = estimate(model, datafile, 'ml', matlab_est)
        
        savekeys = {
            'A', 'B', 'C', 'D', 'Ln', 'sW_diag',
            'sRp_tril', 'sQ_tril', 'sR_tril', 'sPp_tril', 'sPc_tril',
            'status',
        }
        balsave = {k:v for k,v in optbal.items() if k in savekeys}
        mlsave = {k:v for k,v in optml.items() if k in savekeys}
        np.savez(edir / ('bal_' + datafile.stem), **balsave)
        np.savez(edir / ('ml_' + datafile.stem), **mlsave)
        
        xbal, ebal = predict(optbal, yv, uv)
        xml, eml = predict(optml, yv, uv)
        esys = [predict(mdl, yv, uv)[1] for mdl in matlab_est['sys'].flat]
        
        yvdev = yv - np.mean(yv, 0)
        mse = [np.mean(e**2) for e in (yv, ebal, eml, *esys)]
        with open(msefile, 'a') as f:
            print(i, *mse, sep=', ', file=f)
        
        #raise SystemExit
       
 
