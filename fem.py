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
        
        N = len(y)
        self.N = N
        """Number of measurement instants."""
        
        assert y.ndim == 2
        assert y.shape[1] == model.ny
        assert u.shape == (N, model.nu)
        
        
