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
        v['xn'] = [f'xn{i}' for i in range(nx)]
        v['xp'] = [f'xp{i}' for i in range(nx)]
        v['ybias'] = [f'ybias{i}' for i in range(ny)]
        v['A'] = [[f'A{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['B'] = [[f'B{i}_{j}' for j in range(nu)] for i in range(nx)]
        v['C'] = [[f'C{i}_{j}' for j in range(nx)] for i in range(ny)]
        v['D'] = [[f'D{i}_{j}' for j in range(nu)] for i in range(ny)]        
        v['K'] = [[f'K{i}_{j}' for j in range(ny)] for i in range(nx)]
        v['isRp_tril'] = [f'isRp{i}_{j}' for i,j in tril_ind(ny)]
        self.decision.update({k for k in v if k != 'self'})

        # Define auxiliary variables
        v['u'] = [f'u{i}' for i in range(nu)]
        v['y'] = [f'y{i}' for i in range(ny)]
        v['up'] = [f'up{i}' for i in range(nu)]
        v['yp'] = [f'yp{i}' for i in range(ny)]
        
        # Register optimization functions
        self.add_constraint('defects')
        self.add_objective('L')

        # Mark extra functions to generate code
        self.generate_functions.update({'output', 'propagate'})
        
    def defects(self, xn, xp, up, yp, A, B, C, D, K, ybias):
        """Model dynamics defects."""
        return xn - self.propagate(xp, up, yp, A, B, C, D, K, ybias)
    
    def L(self, y, x, u, C, D, isRp_tril, ybias):
        """Measurement log-likelihood."""
        isRp = tril_mat(self.ny, isRp_tril)
        
        ymodel = self.output(x, u, C, D, ybias)
        e = y - ymodel
        
        log_det_isRp = sum(sympy.log(d) for d in isRp.diagonal())
        L = -0.5 * e.T @ isRp.T @ isRp @ e + log_det_isRp
        return L
    
    def output(self, x, u, C, D, ybias):
        return C @ x + D @ u + ybias
    
    def propagate(self, x, u, y, A, B, C, D, K, ybias):
        e = y - self.output(x, u, C, D, ybias)
        return A @ x + B @ u + K @ e
    
    @property
    def generate_assignments(self):
        gen = {'nx': self.nx, 'nu': self.nu, 'ny': self.ny, 
               'n_tril_y': len(self.variables['isRp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class NaturalDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['Pp'] = [[f'Pp{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Pc'] = [[f'Pc{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['Q'] = [[f'Q{i}_{j}' for j in range(nx)] for i in range(nx)]
        v['R'] = [[f'R{i}_{j}' for j in range(ny)] for i in range(ny)]
        v['sRp_tril'] = [f'sRp{i}_{j}' for i,j in tril_ind(ny)]
        v['sQ_tril'] = [f'sQ{i}_{j}' for i,j in tril_ind(nx)]
        v['sR_tril'] = [f'sR{i}_{j}' for i,j in tril_ind(ny)]
        v['sPp_tril'] = [f'sPp{i}_{j}' for i,j in tril_ind(nx)]
        v['sPc_tril'] = [f'sPc{i}_{j}' for i,j in tril_ind(nx)]
        self.decision.update({'Pp', 'Pc', 'Q', 'R'})
        self.decision.update({'sPp_tril', 'sPc_tril', 'sQ_tril'})
        self.decision.update({'sRp_tril', 'sR_tril'})
        
        # Register additional constraints
        self.add_constraint('Rp_inverse')
        self.add_constraint('output_cov')
        self.add_constraint('x_pred_cov')
        self.add_constraint('x_corr_cov')
        self.add_constraint('kalman_gain')
        self.add_constraint('Pp_psd')
        self.add_constraint('Pc_psd')
        self.add_constraint('Q_psd')
        self.add_constraint('R_psd')
        
    def Pp_psd(self, Pp, sPp_tril):
        return psd_constraint(Pp, sPp_tril)
    
    def Pc_psd(self, Pc, sPc_tril):
        return psd_constraint(Pc, sPc_tril)
    
    def Q_psd(self, Q, sQ_tril):
        return psd_constraint(Q, sQ_tril)
    
    def R_psd(self, R, sR_tril):
        return psd_constraint(R, sR_tril)
        
    def Rp_inverse(self, isRp_tril, sRp_tril):
        isRp = tril_mat(self.ny, isRp_tril)
        sRp = tril_mat(self.ny, sRp_tril)
        I = np.eye(self.ny)
        resid = sRp @ isRp - I
        return [resid[i] for i in tril_ind(self.ny)]
    
    def output_cov(self, C, Pp, R, sRp_tril):
        sRp = tril_mat(self.ny, sRp_tril)
        resid = C @ Pp @ C.T + R - sRp @ sRp.T
        return [resid[i] for i in tril_ind(self.ny)]
    
    def x_pred_cov(self, A, Pp, Pc, Q):
        resid = A @ Pc @ A.T + Q - Pc
        return [resid[i] for i in tril_ind(self.nx)]
    
    def x_corr_cov(self, C, K, Pp, Pc):
        resid = Pp - K @ C @ Pp - Pc
        return [resid[i] for i in tril_ind(self.nx)]
    
    def kalman_gain(self, C, K,  Pp, sRp_tril, isRp_tril):
        sRp = tril_mat(self.ny, sRp_tril)
        isRp = tril_mat(self.ny, isRp_tril)
        return Pp @ C.T @ isRp.T - K @ sRp
    
    @property
    def generate_assignments(self):
        gen = {'n_tril_x': len(self.variables['sPp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class NaturalSqrtDTModel(InnovationDTModel):
    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny)
        
        # Define additional decision variables
        v = self.variables
        v['sQ_tril'] = [f'sQ{i}_{j}' for i,j in tril_ind(nx)]
        v['sR_tril'] = [f'sR{i}_{j}' for i,j in tril_ind(ny)]
        v['sRp_tril'] = [f'sRp{i}_{j}' for i,j in tril_ind(ny)]
        v['sPp_tril'] = [f'sPp{i}_{j}' for i,j in tril_ind(nx)]
        v['sPc_tril'] = [f'sPc{i}_{j}' for i,j in tril_ind(nx)]
        po = [[f'pred_orth{i}_{j}' for j in range(nx)] for i in range(2*nx)]
        co = [[f'corr_orth{i}_{j}' for j in range(nx+ny)] for i in range(nx+ny)]
        v['pred_orth'] = po
        v['corr_orth'] = co
        self.decision.update({'sPp_tril', 'sPc_tril', 'sQ_tril'})
        self.decision.update({'sRp_tril', 'sR_tril'})
        self.decision.update({'pred_orth', 'corr_orth'})
        
        # Register additional constraints
        self.add_constraint('pred_orthogonality')
        self.add_constraint('corr_orthogonality')
        self.add_constraint('pred_cov')
        self.add_constraint('corr_cov')
        self.add_constraint('Rp_inverse')
    
    def pred_orthogonality(self, pred_orth):
        resid = 0.5 * (pred_orth.T @ pred_orth - np.eye(self.nx))
        return [resid[i] for i in tril_ind(self.nx)]
    
    def corr_orthogonality(self, corr_orth):
        resid = 0.5 * (corr_orth.T @ corr_orth - np.eye(self.nx + self.ny))
        return [resid[i] for i in tril_ind(self.nx + self.ny)]
    
    def pred_cov(self, A, sPp_tril, sPc_tril, sQ_tril, pred_orth):
        sPp = tril_mat(self.nx, sPp_tril)
        sPc = tril_mat(self.nx, sPc_tril)
        sQ = tril_mat(self.nx, sQ_tril)
        bmat = np.bmat([[sPc.T @ A.T], [sQ.T]])
        return pred_orth @ sPp - bmat

    def corr_cov(self, C, sR_tril, sRp_tril, sPp_tril, sPc_tril, K, corr_orth):
        sPp = tril_mat(self.nx, sPp_tril)
        sPc = tril_mat(self.nx, sPc_tril)
        sRp = tril_mat(self.ny, sRp_tril)
        sR = tril_mat(self.ny, sR_tril)
        
        zeros = np.zeros((self.nx, self.ny))
        M1 = np.bmat([[sRp.T, sRp @ K.T], 
                      [zeros, sPc.T]])
        M2 = np.bmat([[sR.T, zeros.T], 
                      [sPp.T @ C.T, sPp.T]])
        return corr_orth @ M1 - M2
    
    def Rp_inverse(self, isRp_tril, sRp_tril):
        isRp = tril_mat(self.ny, isRp_tril)
        sRp = tril_mat(self.ny, sRp_tril)
        I = np.eye(self.ny)
        resid = sRp @ isRp - I
        return [resid[i] for i in tril_ind(self.ny)]
    
    @property
    def generate_assignments(self):
        gen = {'n_tril_x': len(self.variables['sPp_tril']),
               **getattr(super(), 'generate_assignments', {})}
        return gen


class NaturalSqrtZOHModel(NaturalSqrtDTModel):

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
        self.add_constraint('discretize_AB')
        self.add_constraint('discretize_Q')
    
    def discretize_AB(self, A, B, Ac, Bc, dt):
        z = np.zeros((self.nu, self.nx + self.nu))
        F = np.block([[Ac, Bc], [z]]) * dt
        eF = expm_taylor(F, self.expm_order)
        return eF[:self.nx] - np.c_[A, B]

    def discretize_Q(self, A, Ac, sQ_tril, Qc, dt):
        sQ = tril_mat(self.nx, sQ_tril)
        z = np.zeros((self.nx, self.nx))
        F = np.block([[-Ac, Qc], [z, Ac.T]]) * dt
        eF = expm_taylor(F, self.expm_order)
        return A @ eF[:self.nx, self.nx:] - sQ @ sQ.T


def tril_ind(n):
    yield from ((i,j) for (i,j) in np.ndindex(n, n) if i>=j)


def tril_diag(n):
    return np.array([i==j for (i,j) in tril_ind(n)])


def tril_mat(n, elem):
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
