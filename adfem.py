"""Automatic differentiation filter-error method estimation models."""


import collections
import functools
import inspect
import itertools

import jax
from jax import numpy as jnp
from jax import scipy as jscipy

import numpy as onp


### Enable 64-bit in jax ###
jax.config.update("jax_enable_x64", True)


class BoundADFunction:
    def __init__(self, adfun, model):
        self.adfun = adfun
        self.model = model
    
    @property
    def __signature__(self):
        return bound_signature(self.adfun)
    
    @property
    def __name__(self):
        return self.adfun.__name__
    
    def __repr__(self):
        cls = type(self)
        name = getattr(self, '__name__', '*no name*')
        return f"<{cls.__module__}.{cls.__name__} '{name}'>"
    
    def __call__(self, *args, **kwargs):
        return self.adfun(self.model, *args, **kwargs)
    
    def hess_nnz(self, dec_shapes, out_shape):
        adfun = self.adfun
        return adfun._sparse_deriv_nnz(adfun.hessian, dec_shapes, out_shape)
    
    def hess_ind(self, dec_shapes, out_shape):
        adfun = self.adfun
        return adfun._sparse_deriv_ind(adfun.hessian, dec_shapes, out_shape)

    def hess_val(self, *args, **kwargs):
        adfun = self.adfun
        wrt_seq = adfun.hessian
        return adfun._sparse_deriv_val(wrt_seq, self.model, *args, **kwargs)


class BoundADConstraint(BoundADFunction):
    def jac_nnz(self, dec_shapes, out_shape):
        adfun = self.adfun
        d1 = adfun.first_derivatives
        return adfun._sparse_deriv_nnz(d1, dec_shapes, out_shape)
    
    def jac_ind(self, dec_shapes, out_shape):
        adfun = self.adfun
        d1 = adfun.first_derivatives
        return adfun._sparse_deriv_ind(d1, dec_shapes, out_shape)

    def jac_val(self, *args, **kwargs):
        adfun = self.adfun
        d1 = adfun.first_derivatives
        return adfun._sparse_deriv_val(d1, *args, **kwargs)


class BoundADObjective(BoundADFunction):
    def grad(self, *args, **kwargs):
        adfun = self.adfun
        ret = collections.OrderedDict()
        for wrt in adfun.first_derivatives:
            # Calculate the gradient
            grad_fun = adfun.derivatives[wrt,]
            grad_val = grad_fun(self.model, *args, **kwargs)
            
            # skip empty gradients
            if not grad_val.size:
                continue
            
            # Get the shape of the wrt argument
            try:
                wrt_shape = onp.shape(kwargs[wrt])
            except KeyError:
                wrt_shape = onp.shape(args[adfun.argnum(wrt) - 1])
            
            # Accumulate so the gradient has the same shape as the variable
            ret[wrt] = grad_val.reshape(-1, *wrt_shape).sum(0)
        return ret


class ADFunction:
    """Helper for optimization function automatic differentiation."""

    BoundClass = BoundADFunction
    
    def __init__(self, fun, core_shape=''):
        self.fun = fun
        """Underlying function."""
        
        self.core_shape = core_shape
        """The shape of the output of the elementary function, if vectorized."""
        
        self.hessian = None
        """Hessian elements."""
        
        self.__signature__ = inspect.signature(self.fun)
        """The underlying function's signature."""
        
        self.args = list(self.__signature__.parameters)
        """The underlying function argument names."""
        
        self.derivatives = {}
        """Dictionary of function derivatives."""

        self.first_derivatives = []
        """Sequence of registered first derivatives."""
        
        self.isvectorized = False
        """Whether the function is vectorized"""
    
    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            return self.BoundClass(self, instance)

    def __set_name__(self, owner, name):
        self.__name__ = name
    
    def __repr__(self):
        cls = type(self)
        name = getattr(self, '__name__', '*no name*')
        return f"<{cls.__module__}.{cls.__name__} '{name}'>"
    
    def argnum(self, argname):
        return self.args.index(argname)
    
    def derivative(self, wrt):
        if isinstance(wrt, str):
            wrt_tuple = wrt,
            return self.derivative(wrt_tuple)
        
        # Trivial case, i.e., 0-th derivative
        if wrt == ():
            return self.fun
        
        # Return the registered derivative, if it exists
        try:
            return self.derivatives[wrt]
        except KeyError:
            pass
        
        # Compute the derivative
        assert len(wrt) >= 1
        fun = self.derivative(wrt[1:])
        argnum = self.argnum(wrt[0])
        deriv = jax.jacrev(fun, argnum)
        
        # Save it and return
        self.derivatives[wrt] = deriv
        return deriv
    
    def prepare_derivatives(self, decision):
        # Compute and save the first derivatives
        for d in self.args:
            if d in decision:
                self.derivative(d)
                self.first_derivatives.append(d)
        
        # Define default Hessian, if unset
        if self.hessian is None:
            first_deriv = self.first_derivatives
            hess_gen = itertools.combinations_with_replacement(first_deriv, 2)
            self.hessian = list(hess_gen)
        
        # Compute second derivatives
        for wrt_pair in self.hessian:
            self.derivative(wrt_pair)
    
    def vectorize(self, vectorized):
        vec_args = [a for a in self.args if a in vectorized]
        excluded = [i for i,a in enumerate(self.args) if a not in vectorized]
        
        if not vec_args:
            return
        
        arg_sig = ",".join(f'({vectorized[a]})' for a in vec_args)
        sig = f"{arg_sig}->({self.core_shape})"
        
        self.fun = jnp.vectorize(self.fun, excluded=excluded, signature=sig)
        for wrt, d in self.derivatives.items():
            wrtsig = (vectorized[var] for var in reversed(wrt))
            if self.core_shape:
                outsig = ','.join((self.core_shape, *wrtsig))
            else:
                outsig = ','.join(wrtsig)
            dsig = f"{arg_sig}->({outsig})"
            vecd = jnp.vectorize(d, excluded=excluded, signature=dsig)
            self.derivatives[wrt] = vecd
        
        core_shape = self.core_shape
        out_core_ndim = len(core_shape.split(',')) if core_shape else 0
        self.core_ndim = {a: len(vectorized[a].split(',')) for a in vec_args}
        self.core_ndim[None] = out_core_ndim
        self.isvectorized = True
    
    def _split_shape(self, shape, varname=None):
        """Split a variable's shape into extension and core."""
        try:
            core_ndim = self.core_ndim[varname]
        except KeyError:
            return (), shape #This variable is not vectorized
        
        # Test whether the core element is scalar (ndim==0)
        if core_ndim:
            return shape[:-core_ndim], shape[-core_ndim:]
        else:
            return shape, ()        
    
    def _ext_shape(self, shape, varname=None):
        """Return a variable's shape extension."""
        return self._split_shape(shape, varname)[0]
    
    def _core_shape(self, shape, varname=None):
        """Return a variable's core shape."""
        return self._split_shape(shape, varname)[1]
        
    def _deriv_core_shape(self, wrt, dec_shapes, out_shape):
        out_ext, out_core = self._split_shape(out_shape)
        if len(wrt) == 0:
            return out_core
        else:
            wrt0, *wrt_rem = wrt
            wrt0_shape = dec_shapes[wrt0]
            wrt0_ext, wrt0_core = self._split_shape(wrt0_shape, wrt0)
            rem_core = self._deriv_core_shape(wrt_rem, dec_shapes, out_shape)
            return wrt0_core + rem_core
    
    def _deriv_core_ind(self, wrt, dec_shapes, out_shape):
        if len(wrt) == 0:
            out_ext, out_core = self._split_shape(out_shape)
            return [onp.arange(shape_size(out_core))]
        else:
            wrt0, *wrt_rem = wrt
            wrt0_shape = dec_shapes[wrt0]
            wrt0_ext, wrt0_core = self._split_shape(wrt0_shape, wrt0)
            rem_ind = self._deriv_core_ind(wrt_rem, dec_shapes, out_shape)
            
            wrt0_core_size = shape_size(wrt0_core)
            wrt0_tile = rem_ind[0].size
            wrt0_ind = onp.tile(onp.arange(wrt0_core_size), wrt0_tile)
            core_ind = [onp.repeat(i, wrt0_core_size) for i in rem_ind]
            core_ind.insert(0, wrt0_ind)
            return core_ind
    
    def _sparse_deriv_nnz(self, wrt_seq, dec_shapes, out_shape):
        nnz = 0
        out_ext, out_core = self._split_shape(out_shape)
        ext_sz = shape_size(out_ext)
        for wrt in wrt_seq:
            deriv_core = self._deriv_core_shape(wrt, dec_shapes, out_shape)
            nnz += shape_size(deriv_core) * ext_sz
        return nnz
    
    def _sparse_deriv_ind(self, wrt_seq, dec_shapes, out_shape):
        out_ext, out_core = self._split_shape(out_shape)
        core_out_sz = shape_size(out_core)
        
        ret = collections.OrderedDict()
        for wrt in wrt_seq:
            ind = []
            base_ind = self._deriv_core_ind(wrt, dec_shapes, out_shape)
            for wrt_name, wrt_ind in zip(wrt, base_ind):
                wrt_shape = dec_shapes[wrt_name]
                wrt_ext, wrt_core = self._split_shape(wrt_shape, wrt_name)
                wrt_core_sz = shape_size(wrt_core)
                
                wrt_offs = ndim_range(wrt_ext) * wrt_core_sz
                wrt_offs = onp.broadcast_to(wrt_offs, out_ext)
                ind.append(wrt_ind + wrt_offs[..., None])
            
            # Extend the output indices
            out_ind = base_ind[-1]
            out_offs = ndim_range(out_ext) * core_out_sz
            ind.append(out_ind + out_offs[..., None])
            
            # Save in dictionary
            ret[wrt] = onp.array(ind)
        return ret

    def _sparse_deriv_val(self, wrt_seq, *args, **kwargs):
        ret = collections.OrderedDict()
        for wrt in wrt_seq:
            deriv = self.derivatives[wrt]
            ret[wrt] = deriv(*args, **kwargs)
        return ret


class ADConstraint(ADFunction):
    """Helper for constraint function automatic differentiation."""

    BoundClass = BoundADConstraint
    

class ADObjective(ADFunction):
    """Helper for objective function automatic differentiation."""

    BoundClass = BoundADObjective


def constraint(core_shape_or_fun=None):
    if callable(core_shape_or_fun):
        fun = core_shape_or_fun
        return constraint()(fun)
    else:
        core_shape = core_shape_or_fun
        def decorator(fun):
            return ADConstraint(fun, core_shape)
        return decorator


def objective(fun):
    return ADObjective(fun)


def hessian(*args):
    def decorator(obj):
        obj.hessian = args
        return obj
    return decorator


class ADModel:
    def __init_subclass__(cls):
        base_dec = getattr(super(), 'decision', set())
        cls_dec = getattr(cls, 'decision', set())  
        cls.decision = set.union(base_dec, cls_dec)
        """The decision variables of this model."""
        
        base_vec = getattr(super(), 'vectorized', {})
        cls_vec = getattr(cls, 'vectorized', {})  
        cls.vectorized = {**base_vec, **cls_vec}
        """The core shape of vectorized variables."""
        
        cls_items = cls.__dict__.items()
        adfuns = {k:v for k,v in cls_items if isinstance(v, ADFunction)}
        for name, adfun in adfuns.items():
            adfun.prepare_derivatives(cls.decision)
            adfun.vectorize(cls.vectorized)


class InnovationDTModel(ADModel):
    
    decision = {
        'x', 'e', 'xnext', 'xprev', 'eprev', 'ybias', 
        'A', 'B', 'C', 'D', 'L', 'isRp_tril'
    }
    """Decision variables of the optimization problem."""
    
    vectorized = dict(
        x='nx', e='ny', xnext='nx', xprev='nx', eprev='ny',
        u='nu', y='ny', uprev='nu',
        A='nx,nx', B='nx,nu', C='ny,nx', D='ny,nu', L='nx,ny', ybias='ny',
        isRp_tril='nty',
    )
    """Decision variables of the optimization problem."""
    
    def __init__(self, nx, nu, ny):
        self.nx = nx
        """Number of states."""
        
        self.nu = nu
        """Number of inputs."""
        
        self.ny = ny
        """Number of outputs."""
    
    @hessian(('xprev', 'A'), ('eprev', 'L'))
    @constraint('nx')
    def defects(self, xnext, xprev, uprev, eprev, A, B, L):
        """Model dynamics defects."""
        xpred = A @ xprev + B @ uprev + L @ eprev
        return xnext - xpred
    
    @hessian(('x', 'C'), ('x', 'isRp_tril'), ('C', 'isRp_tril'), 
             ('D', 'isRp_tril'), ('ybias', 'isRp_tril'))
    @constraint('ny')
    def innovation(self, y, e, x, u, C, D, ybias, isRp_tril):
        """Model dynamics defects."""
        ymodel = C @ x + D @ u + ybias
        isRp = tril_mat(isRp_tril)
        return isRp @ (y - ymodel) - e
    
    @hessian(('e', 'e'), ('isRp_tril', 'isRp_tril'))
    @objective
    def L(self, e, isRp_tril):
        """Measurement log-likelihood."""
        isRp = tril_mat(isRp_tril)
        log_det_isRp = jnp.log(isRp.diagonal()).sum()
        return -0.5 * (e ** 2).sum() + log_det_isRp


def tril_mat(tril_elem):
    """Build a matrix from its lower-triangular elements."""
    ntril = len(tril_elem)
    n = int(round(0.5*(onp.sqrt(8*ntril + 1) - 1)))
    tril_ind = onp.tril_indices(n)
    M = jnp.zeros((n, n))
    return M.at[tril_ind].set(tril_elem)


def shape_size(shape):
    return onp.prod(shape, dtype=int)


def ndim_range(shape):
    assert isinstance(shape, tuple)
    return onp.arange(shape_size(shape)).reshape(shape)


def bound_signature(method):
    """Return the signature of a method when bound."""
    sig = inspect.signature(method)
    param = list(sig.parameters.values())[1:]
    return inspect.Signature(param, return_annotation=sig.return_annotation)
