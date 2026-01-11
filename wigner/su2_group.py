from collections.abc import Callable
import logging
from typing import Any, Concatenate, Optional, Union

import numpy as np

import sympy as sym
import sympy.physics.quantum.spin as spin
import sympy.physics.quantum.cg as cg
from tqdm import tqdm

from wigner.utils import trigsimp

import atexit
from pathlib import Path
import pickle


logger = logging.getLogger(__name__)


def _gell_mann_dict(a: int, b: int) -> dict[tuple[int, int], sym.Expr]:
    gell_mann_dict: dict[tuple[int, int], sym.Expr] = {}

    if a == b:
        for n in range(a):
            gell_mann_dict[n, n] = 1
        gell_mann_dict[a, a] = -a
    else:
        if a < b:
            _ab = 1
        else:
            _ab = sym.I

        gell_mann_dict[a, b] = _ab
        gell_mann_dict[b, a] = sym.conjugate(_ab)

    return gell_mann_dict


def _gell_mann_x(d: int) -> list[sym.ImmutableSparseMatrix]:
    return [
        sym.ImmutableSparseMatrix(d, d, _gell_mann_dict(a, b))
        for b in range(d)
        for a in range(b)
    ]


def _gell_mann_y(d: int) -> list[sym.ImmutableSparseMatrix]:
    return [
        sym.ImmutableSparseMatrix(d, d, _gell_mann_dict(a, b))
        for a in range(d)
        for b in range(a)
    ]


def _gell_mann_z(d: int) -> list[sym.ImmutableSparseMatrix]:
    return [sym.ImmutableSparseMatrix(d, d, _gell_mann_dict(a, a)) for a in range(1, d)]


def _gell_mann_basis(d: int) -> list[sym.ImmutableSparseMatrix]:
    return _gell_mann_z(d) + _gell_mann_x(d) + _gell_mann_y(d)


class Su2Group:
    _cache_dir: Optional[Path] = Path(__file__).parent / ".cache"
    _cache: Optional[dict[int, Su2Group]] = None
    _cached_attrs: list[str] = [
        "_jx",
        "_jy",
        "_jz",
        "_disp",
        "_parity",
        "_kernel",
        "_gm_basis",
        "_wg_basis",
        "_wg_expr",
    ]

    _dim: int
    _j: Union[sym.Integer, sym.Rational]

    _jx: sym.ImmutableSparseMatrix
    _jy: sym.ImmutableSparseMatrix
    _jz: sym.ImmutableSparseMatrix

    _disp: sym.ImmutableDenseMatrix
    _parity: sym.ImmutableSparseMatrix
    _kernel: sym.ImmutableDenseMatrix

    _gm_basis: list[sym.ImmutableSparseMatrix]
    _wg_basis: list[sym.Expr]

    _wg_expr: sym.Expr
    _wg_transform: Callable[
        Concatenate[np.typing.ArrayLike, np.typing.ArrayLike, ...],
        np.typing.ArrayLike,
    ]

    @classmethod
    def _new(cls, dim: int, **cached_obj: Optional[dict[str, Any]]) -> Su2Group:
        group = object.__new__(Su2Group)

        group._dim = dim
        group._j = (sym.Integer(group._dim) - 1) / 2

        if cached_obj is not None:
            for key in cls._cached_attrs:
                if key in cached_obj:
                    setattr(group, key, cached_obj[key])

        return group

    def _get_obj(self) -> dict[str, Any]:
        cached_obj: dict[str, Any] = {"dim": self.dim}

        for key in self._cached_attrs:
            if hasattr(self, key):
                cached_obj[key] = getattr(self, key)

        return cached_obj

    @classmethod
    def _get_cache(cls) -> dict[int, Su2Group]:
        if cls._cache is None:

            cached_list: list[dict[str, Any]] = []

            if cls._cache_dir is not None:
                try:
                    with (cls._cache_dir / f"{cls.__name__}.pkl").open("rb") as f:
                        cached_list = pickle.load(f)
                except FileNotFoundError:
                    pass

            cls._cache = {
                cached_obj["dim"]: Su2Group._new(**cached_obj)
                for cached_obj in cached_list
            }

        return cls._cache

    @classmethod
    def _dump_cache(cls) -> None:
        if cls._cache is not None:
            cache = cls._get_cache()
            cached_list = [cache[dim]._get_obj() for dim in sorted(cache)]

            if cls._cache_dir is not None:
                cls._cache_dir.mkdir(exist_ok=True)

                with (cls._cache_dir / f"{cls.__name__}.pkl").open("wb") as f:
                    pickle.dump(cached_list, f)

    def __new__(cls, dim: int) -> Su2Group:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Value of `dim` should be positive integer")

        _cache = cls._get_cache()
        if dim not in _cache:
            _cache[dim] = Su2Group._new(dim)

        return _cache[dim]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def j(self) -> Union[sym.Integer, sym.Rational]:
        return self._j

    @property
    def gell_mann_basis(self) -> list[sym.ImmutableSparseMatrix]:
        if not hasattr(self, "_gm_basis"):
            self._gm_basis = list(_gell_mann_basis(self.dim))

        return self._gm_basis

    def ket(self, n: sym.Basic) -> spin.JzKet:
        return spin.JzKet(self.j, self.j - n)

    def bra(self, n: sym.Basic) -> spin.JzBra:
        return spin.JzBra(self.j, self.j - n)

    def rep_matrix(self, op: spin.SpinOpBase) -> sym.ImmutableSparseMatrix:
        return sym.ImmutableSparseMatrix(
            self.dim,
            self.dim,
            lambda i, j: spin.qapply(self.bra(i) * op * self.ket(j)),
        )

    @property
    def jx(self) -> sym.ImmutableSparseMatrix:
        if not hasattr(self, "_jx"):
            self._jx = self.rep_matrix(spin.Jx / spin.hbar)

        return self._jx

    @property
    def jy(self) -> sym.ImmutableSparseMatrix:
        if not hasattr(self, "_jy"):
            self._jy = self.rep_matrix(spin.Jy / spin.hbar)

        return self._jy

    @property
    def jz(self) -> sym.ImmutableSparseMatrix:
        if not hasattr(self, "_jz"):
            self._jz = self.rep_matrix(spin.Jz / spin.hbar)

        return self._jz

    @property
    def disp(self) -> sym.ImmutableDenseMatrix:
        if not hasattr(self, "_disp"):
            _theta, _phi = sym.symbols("theta, phi", real=True)
            self._disp = sym.ImmutableDenseMatrix(
                sym.exp(-sym.I * _phi * self.jz) @ sym.exp(-sym.I * _theta * self.jy)
            )

        return self._disp

    def _parity_diag(self, n: int) -> sym.Expr:
        j = self.j
        return sym.simplify(
            sum(
                (2 * l + 1) * cg.CG(j, j - n, l, 0, j, j - n).doit()
                for l in range(2 * j + 1)
            )
            / (2 * j + 1)
        )

    @property
    def parity(self) -> sym.ImmutableSparseMatrix:
        if not hasattr(self, "_parity"):
            d = self.dim
            self._parity = sym.ImmutableSparseMatrix(
                d, d, {(n, n): self._parity_diag(n) for n in range(d)}
            )

        return self._parity

    @property
    def kernel(self) -> sym.ImmutableDenseMatrix:
        if not hasattr(self, "_kernel"):
            self._kernel = sym.ImmutableDenseMatrix(
                trigsimp(self.disp @ self.parity @ self.disp.H)
            )

        return self._kernel

    def _wigner_pure(self, psi: sym.NDimArray | sym.MatrixBase) -> sym.Expr:
        d = self.dim
        psi = sym.ImmutableDenseMatrix(d, 1, psi).normalized()

        return (psi.H @ self.kernel @ psi)[0]

    def _wigner_op(self, op: sym.NDimArray | sym.MatrixBase) -> sym.Expr:
        op = sym.ImmutableDenseMatrix(op)

        return trigsimp(
            sum(
                ein_val * sum(self._wigner_pure(ein_vec) for ein_vec in ein_vecs)
                for ein_val, _, ein_vecs in op.eigenvects()
                if not ein_val.is_zero
            )
        )

    def _wg_simp(self, x: sym.Expr) -> sym.Expr:
        def _gens(var: sym.Expr, dim: int) -> list[sym.Expr]:
            return [h(k * var) for h in [sym.cos, sym.sin] for k in range(1, dim)]

        def _decomp(
            x: sym.Expr, gens: list[sym.Expr]
        ) -> tuple[sym.Expr, list[sym.Expr]]:
            xcs = [x.coeff(gen) for gen in gens]
            xc = sum(xc * gen for xc, gen in zip(xcs, gens))

            return sym.simplify(x - xc), xcs

        def _comp(xc0: sym.Expr, xcs: list[sym.Expr], gens: list[sym.Expr]) -> sym.Expr:
            return sum([xc * gen for xc, gen in zip(xcs, gens)], start=xc0)

        x = sym.expand(x)

        theta, phi = sym.symbols("theta, phi", real=True)

        gens_theta = _gens(theta, self.dim)
        xc0, xcs = _decomp(x, gens_theta)

        gens_phi = _gens(phi, self.dim)
        xc0 = _comp(*_decomp(sym.re(xc0), gens_phi), gens_phi)
        xcs = [_comp(*_decomp(sym.re(xc), gens_phi), gens_phi) for xc in xcs]

        return _comp(xc0, xcs, gens_theta)

    @property
    def wigner_basis(self) -> list[sym.Expr]:
        if not hasattr(self, "_wg_basis"):
            _wg_basis = []
            for gm in tqdm(
                self.gell_mann_basis, desc=f"Wigner basis for Su2Group({self.dim})"
            ):
                _wg_basis += [self._wg_simp(self._wigner_op(gm))]

            self._wg_basis = _wg_basis

        return self._wg_basis

    @property
    def wigner_transform_expr(self) -> sym.Expr:
        d = self.dim

        if not hasattr(self, "_wg_expr"):
            _op_elts = sym.symarray("o", (d, d), complex=True)
            _op = sym.Matrix.zeros(d, d)
            for m in range(d):
                _op[m, m] = sym.re(_op_elts[m, m])
                for n in range(m):
                    _elt_re = sym.re(_op_elts[n, m])
                    _elt_im = sym.im(_op_elts[n, m])

                    _op[m, n] = _elt_re + sym.I * _elt_im
                    _op[n, m] = _elt_re - sym.I * _elt_im

            assert _op.is_hermitian

            _op_coeff = [
                sym.simplify(sym.trace(_op @ gm) / sym.trace(gm**2))
                for gm in self.gell_mann_basis
            ]

            self._wg_expr = sym.simplify(sym.trace(_op) / self.dim) + self._wg_simp(
                sym.expand(sum((o * w) for o, w in zip(_op_coeff, self.wigner_basis)))
            )

        return self._wg_expr

    def wigner_transform(
        self,
        theta: np.typing.ArrayLike,
        phi: np.typing.ArrayLike,
        op: np.typing.ArrayLike,
    ) -> np.ndarray | np.floating:
        d = self.dim
        rdx, cdx = np.triu_indices(d)

        if not hasattr(self, "_wg_transform"):
            _theta, _phi = sym.symbols("theta, phi", real=True)
            _op_triu = sym.symarray("o", (d, d), complex=True)[rdx, cdx]

            _args = (_theta, _phi, *_op_triu)
            assert not self.wigner_transform_expr.free_symbols.difference(_args)
            assert self.wigner_transform_expr.is_real

            self._wg_transform = sym.lambdify(
                _args,
                self.wigner_transform_expr,
                modules=["numpy"],
                cse=True,
                docstring_limit=None,
            )

        op_triu = np.moveaxis(op[..., rdx, cdx], -1, 0)
        theta, phi, *op_triu = np.broadcast_arrays(theta, phi, *op_triu)

        result = np.asarray(self._wg_transform(theta, phi, *op_triu))
        return result if result.shape else result[()]


atexit.register(Su2Group._dump_cache)
