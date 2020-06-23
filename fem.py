"""Filter-error method estimation module."""


import numpy as np

from ceacoest import optim


class InnovationDTProblem(optim.Problem):

    def __init__(self, model, y, u):
        super().__init__()

        self.model = model
        """Underlying model."""
        
        self.y = np.asarray(y)
        """Measurements."""
        
        self.u = np.asarray(u)
        """Inputs."""

        self.uprev = self.u[:-1]
        """Previous inputs."""
        
        N = len(y)
        self.N = N
        """Number of measurement instants."""
        
        assert y.ndim == 2
        assert y.shape[1] == model.ny
        assert u.shape == (N, model.nu)
        assert N > 1
        
        # Register decision variables
        self.add_decision('ybias', model.ny)
        self.add_decision('isRp_tril', model.nty)
        self.add_decision('A', (model.nx, model.nx))
        self.add_decision('B', (model.nx, model.nu))
        self.add_decision('C', (model.ny, model.nx))
        self.add_decision('D', (model.ny, model.nu))
        self.add_decision('Ln', (model.nx, model.ny))
        self.add_decision('yp', (N, model.ny))
        x = self.add_decision('x', (N, model.nx))
        en = self.add_decision('en', (N, model.ny))
        
        # Define and register dependent variables
        xprev = optim.Decision((N-1, model.nx), x.offset)
        enprev = optim.Decision((N-1, model.ny), en.offset)
        xnext = optim.Decision((N-1, model.nx), x.offset + model.nx)
        self.add_dependent_variable('xprev', xprev)
        self.add_dependent_variable('enprev', enprev)
        self.add_dependent_variable('xnext', xnext)
    
        # Register problem functions
        self.add_objective(model.loglikelihood, N)
        self.add_constraint(model.dynamics, (N - 1, model.nx))
        self.add_constraint(model.output, (N, model.ny))
        self.add_constraint(model.innovation, (N, model.ny))
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'y': self.y, 'u': self.u, 'uprev': self.uprev,
                **super().variables(dvec)}


class BalancedDTProblem(InnovationDTProblem):
    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nx = model.nx
        nu = model.nu
        ny = model.ny
        ntx = nx * (nx + 1) // 2
        
        # Register decision variables
        self.add_decision('sW_diag', nx)
        self.add_decision('ctrl_orth', (nx, nx + nu))
        self.add_decision('obs_orth', (nx, nx + ny))
        
        # Register constraint functions
        self.add_constraint(model.ctrl_gram, (nx, nx + nu))
        self.add_constraint(model.obs_gram, (nx, nx + ny))
        self.add_constraint(model.ctrl_orthogonality, ntx)
        self.add_constraint(model.obs_orthogonality, ntx)


class StableDTProblem(InnovationDTProblem):
    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nx = model.nx
        nu = model.nu
        ny = model.ny
        ntx = nx * (nx + 1) // 2
        
        # Register decision variables
        self.add_decision('Apred', (nx, nx))
        self.add_decision('sM_tril', ntx)
        self.add_decision('stab_orth', (nx, 2*nx))
        
        # Register constraint functions
        self.add_constraint(model.predictor_stability, (nx, 2*nx))
        self.add_constraint(model.predictor_A, (nx, nx))
        self.add_constraint(model.stab_orthogonality, ntx)


class MaximumLikelihoodDTProblem(InnovationDTProblem):

    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nx = model.nx
        nu = model.nu
        ny = model.ny
        nty = ny * (ny + 1) // 2
        ntx = nx * (nx + 1) // 2
        nxy = nx + ny
        ntxy = nxy * (nxy + 1) // 2
                
        # Register decision variables
        self.add_decision('S', (nx, ny))
        self.add_decision('sQ_tril', ntx)
        self.add_decision('sR_tril', nty)
        self.add_decision('sPp_tril', ntx)
        self.add_decision('sRp_tril', nty)
        self.add_decision('isR_tril', nty)
        self.add_decision('sQd_tril', ntx)
        self.add_decision('Kn', (nx, ny))
        self.add_decision('pred_orth', (nxy, nxy + nx))
        self.add_decision('decorr_orth', (nx, nxy))
        
        # Register constraint functions
        self.add_constraint(model.decorr_orthogonality, ntx)
        self.add_constraint(model.pred_orthogonality, ntxy)
        self.add_constraint(model.decorr_cov, (nx, nxy))
        self.add_constraint(model.pred_cov, (nxy, ny + 2*nx))
        self.add_constraint(model.sRp_inv, nty)
        self.add_constraint(model.sR_inv, nty)


class ZOHDynamicsProblem(InnovationDTProblem):
    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nx = model.nx
        nu = model.nu
        
        # Register decision variables
        self.add_decision('Ac', (nx, nx))
        self.add_decision('Bc', (nx, nu))
        
        # Register constraint functions
        self.add_constraint(model.discretize_AB, (nx, nx + nu))
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'dt': self.model.dt, **super().variables(dvec)}


class DiscretizedNoiseProblem(MaximumLikelihoodDTProblem):
    def __init__(self, model, y, u):
        super().__init__(model, y, u)
        
        nx = model.nx
        nu = model.nu
        ntx = nx * (nx + 1) // 2
        
        # Register decision variables
        self.add_decision('sQc_tril', model.ntx)
        if 'Ac' not in self.decision:
            self.add_decision('Ac', (nx, nx))
        
        # Register constraint functions
        self.add_constraint(model.discretize_Q, ntx)
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'dt': self.model.dt, **super().variables(dvec)}

