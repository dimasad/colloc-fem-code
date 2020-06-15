"""Symbolic filter-error method estimation models."""


import numpy as np

import scipy.special
import sympy
from ceacoest.modelling import symoptim


class InnovationDTModel(symoptim.Model):
    def __init__(self, nx, nu, ny):
        super().__init__()

        self.nx = nx
        """Number of states."""

        self.nu = nu
        """Number of inputs."""
        
        self.ny = ny
        """Number of outputs."""
        
        # Define decision variables
        v = self.variables
        v['x'] = [f'x{i}' for i in range(nx)]
        v['e'] = [f'e{i}' for i in range(ny)]
        v['xnext'] = [f'xnext{i}' for i in range(nx)]
        v['xprev'] = [f'xprev{i}' for i in range(nx)]
        v['eprev'] = [f'eprev{i}' for i in range(ny)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]        
        v['L'] = [[f'L{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['sRp_tril'] = [f'sRp{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({k for k in v if k != 'self'})

        # Define auxiliary variables
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['uprev'] = [f'uprev{i}' for i in range(nu)]
        
        # Register optimization functions
        self.add_constraint('dynamics')
        self.add_constraint('innovation')
        self.add_objective('loglikelihood')
    
    def dynamics(self, xnext, xprev, uprev, eprev, A, B, L):
        """Model dynamics defects."""
        xpred = A @ xprev + B @ uprev + L @ eprev
        return xnext - xpred
    
    def innovation(self, y, e, x, u, C, D, ybias, sRp_tril):
        """Model normalized innovation constraint."""
        sRp = tril_mat(sRp_tril)
        ymodel = C @ x + D @ u + ybias
        return y - ymodel - sRp @ e
    
    def loglikelihood(self, e, sRp_tril):
        """Log-likelihood function."""
        sRp = tril_mat(sRp_tril)
        log_det_sRp = sum(sympy.log(d) for d in sRp.diagonal())
        return -0.5 * (e ** 2).sum() - log_det_sRp
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'nty': len(self.variables['sRp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class BalancedDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['sW_diag'] = [f'sW{i}' for i in range(nx)]
        co = [[f'ctrl_orth{i}_{j}' for j in range(nx+nu)] for i in range(nx)]
        oo = [[f'obs_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx)]
        v['ctrl_orth'] = co
        v['obs_orth'] = oo
        self.decision.update({'sW_diag', 'ctrl_orth', 'obs_orth'})

        # Register additional constraints
        self.add_constraint('ctrl_gram')
        self.add_constraint('obs_gram')
        self.add_constraint('ctrl_orthogonality')
        self.add_constraint('obs_orthogonality')
    
    def ctrl_gram(self, sW_diag, A, B, ctrl_orth):
        bmat = np.block([A * sW_diag, B])
        return sW_diag[:, None] * ctrl_orth - bmat
    
    def obs_gram(self, sW_diag, A, C, obs_orth):
        bmat = np.block([A.T * sW_diag, C.T])
        return sW_diag[:, None] * obs_orth - bmat
    
    def ctrl_orthogonality(self, ctrl_orth):
        resid = 0.5 * (ctrl_orth @ ctrl_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def obs_orthogonality(self, obs_orth):
        resid = 0.5 * (obs_orth @ obs_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]


class MaximumLikelihoodDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['Kn'] = [[f'Kn{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['sQ_tril'] = [f'sQ{i}_{j}' for i,j in tril_ind(nx)]
        v['sR_tril'] = [f'sR{i}_{j}' for i,j in tril_ind(ny)]
        v['sPp_tril'] = [f'sPp{i}_{j}' for i,j in tril_ind(nx)]
        v['sPc_tril'] = [f'sPc{i}_{j}' for i,j in tril_ind(nx)]
        po = [[f'pred_orth{i}_{j}' for j in range(2*nx)] for i in range(nx)]
        co = [[f'corr_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx+ny)]
        v['pred_orth'] = po
        v['corr_orth'] = co
        self.decision.update({'sPp_tril', 'sPc_tril', 'sQ_tril'})
        self.decision.update({'sRp_tril', 'sR_tril', 'Kn'})
        self.decision.update({'pred_orth', 'corr_orth'})
        
        # Register additional constraints
        self.add_constraint('pred_orthogonality')
        self.add_constraint('corr_orthogonality')
        self.add_constraint('pred_cov')
        self.add_constraint('corr_cov')
        self.add_constraint('kalman_gain')
    
    def pred_orthogonality(self, pred_orth):
        resid = 0.5 * (pred_orth @ pred_orth.T - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def corr_orthogonality(self, corr_orth):
        resid = 0.5 * (corr_orth @ corr_orth.T - np.eye(self.nx + self.ny))
        return [resid[i] for i in tril_ind(self.nx + self.ny)]
    
    def pred_cov(self, A, sPp_tril, sPc_tril, sQ_tril, pred_orth):
        sPp = tril_mat(sPp_tril)
        sPc = tril_mat(sPc_tril)
        sQ = tril_mat(sQ_tril)
        bmat = np.block([[A @ sPc, sQ]])
        return sPp @ pred_orth - bmat
    
    def corr_cov(self, C, sR_tril, sRp_tril, sPp_tril, sPc_tril, Kn, 
                 corr_orth):
        sPp = tril_mat(sPp_tril)
        sPc = tril_mat(sPc_tril)
        sRp = tril_mat(sRp_tril)
        sR = tril_mat(sR_tril)
        
        zeros = np.zeros((self.nx, self.ny))
        M1 = np.block([[sRp,  zeros], 
                       [Kn, sPc]])
        M2 = np.block([[sR,    C @ sPp], 
                       [zeros, sPp]])
        return M1 @ corr_orth - M2
    
    def kalman_gain(self, L, Kn, A):
        return L - A @ Kn
        
    @property
    def generate_assignments(self):
        gen = {'ntx': len(self.variables['sPp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class ZOHDynamicsModel(InnovationDTModel):
    
    expm_order = 3
    
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['dt'] = 'dt'
        v['Ac'] = [[f'Ac{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Bc'] = [[f'Bc{i}_{j}' for j in range(nu)] for i in range(nx)]
        self.decision.update({'Ac', 'Bc'})
        
        # Register additional constraints
        self.add_constraint('discretize_AB')
    
    def discretize_AB(self, A, B, Ac, Bc, dt):
        z = np.zeros((self.nu, self.nx + self.nu))
        F = np.block([[Ac, Bc], [z]]) * dt
        eF = expm_taylor(F, self.expm_order)
        return eF[:self.nx] - np.c_[A, B]


class DiscretizedNoiseModel(MaximumLikelihoodDTModel):
    
    expm_order = 3
    
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['dt'] = 'dt'
        v['Ac'] = [[f'Ac{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Bc'] = [[f'Bc{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['Qc'] = [[f'Qc{i}_{j}' for j in range(nx)] for i in range(nx)]
        self.decision.update({'Ac', 'Bc', 'Qc'})
        
        # Register additional constraints        
        self.add_constraint('discretize_Q')
    
    def discretize_Q(self, A, Ac, sQ_tril, Qc, dt):
        sQ = tril_mat(sQ_tril)
        z = np.zeros((self.nx, self.nx))
        F = np.block([[-Ac, Qc], [z, Ac.T]]) * dt
        eF = expm_taylor(F, self.expm_order)
        return A @ eF[:self.nx, self.nx:] - sQ @ sQ.T


def tril_ind(n):
    yield from ((i,j) for (i,j) in np.ndindex(n, n) if i>=j)


def tril_diag(n):
    return np.array([i==j for (i,j) in tril_ind(n)])


def tril_mat(elem):
    ntril = len(elem)
    n = int(round(0.5*(np.sqrt(8*ntril + 1) - 1)))
    mat = np.zeros((n, n), dtype=elem.dtype)
    for ind, val in zip(tril_ind(n), elem):
        mat[ind] = val
    return mat


def symmetry_constraint(M):
    n = len(M)
    return [M[i,j] - M[j,i] for (i,j) in np.ndindex(n, n) if i < j]


def psd_constraint(mat, sqrt_tril_elem):
    sqrt = tril_mat(len(mat), sqrt_tril_elem)
    return mat - sqrt @ sqrt.T


def expm_taylor(a, order):
    """Taylor approximation for matrix exponential."""
    a = np.asarray(a)
    n = len(a)
    assert a.shape == (n, n)
    
    term = np.eye(n)
    expm = term
    for i in range(1, order + 1):
        term = 1 / i * term @ a
        expm = expm + term
    return expm
