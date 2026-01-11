import atexit
from itertools import product
import logging
import pytest

import sympy as sym

from wigner import Su2Group

Su2Group._cache_dir = None

logger = logging.getLogger(__name__)

DIMS = [2, 3, 4, 5]


@pytest.mark.parametrize("dim", DIMS)
def test_gell_mann(dim: int):
    group = Su2Group(dim)
    logger.debug("%s", group)

    assert group.dim == dim
    assert 2 * group.j + 1 == dim

    su2_basis = group.gell_mann_basis
    assert len(su2_basis) == dim**2

    for i, j in product(range(dim**2), repeat=2):
        su2_i = su2_basis[i]
        su2_j = su2_basis[j]

        assert sym.trace(su2_i @ su2_j) == int(i == j)


@pytest.mark.parametrize("dim", DIMS)
def test_group_j(dim: int):
    group = Su2Group(dim)

    assert group.jx.is_hermitian
    assert group.jx.shape == (dim, dim)

    assert group.jy.is_hermitian
    assert group.jy.shape == (dim, dim)

    assert group.jz.is_hermitian
    assert group.jz.shape == (dim, dim)

    J2 = group.jx**2 + group.jy**2 + group.jz**2
    assert J2 == group.j * (group.j + 1) * sym.eye(dim, dim)


def simp(expr: sym.Expr) -> sym.Expr:
    return sym.simplify(sym.expand_complex(expr))


@pytest.mark.parametrize("dim", DIMS)
def test_wigner_displacement(dim: int):
    group = Su2Group(dim)

    assert simp(group.disp.H @ group.disp) == sym.eye(dim)
    assert group.disp.shape == (dim, dim)

    theta, phi = sym.symbols("theta, phi", real=True)
    assert not group.disp.free_symbols.difference([theta, phi])

    disp = group.disp
    disp_theta = disp.diff(theta)
    disp_phi = disp.diff(phi)

    assert sym.ImmutableDenseMatrix.zeros(dim, dim).is_zero_matrix

    assert simp(disp_theta + sym.I * disp @ group.jy).is_zero_matrix
    assert simp(disp_phi + sym.I * group.jz @ disp).is_zero_matrix

    logger.debug("%s", disp)


@pytest.mark.parametrize("dim", DIMS)
def test_wigner_kernel(dim: int):
    group = Su2Group(dim)

    assert sym.trace(group.parity) == 1
    assert sym.trace(group.kernel) == 1

    assert sym.simplify(sym.expand(sym.trace(group.parity**2))) == dim
    assert sym.simplify(sym.expand(sym.trace(group.kernel**2))) == dim


@pytest.mark.parametrize("dim", DIMS)
def test_wigner_pure_state(dim: int):
    group = Su2Group(dim)

    for i in range(dim):
        psi = sym.Matrix.zeros(dim, 1)
        psi[i] = 1

        wg_i = group._wigner_pure(psi)
        assert sym.im(wg_i).is_zero

        for j in range(i):
            psi = sym.Matrix.zeros(dim, 1)
            psi[i] = psi[j] = 1 / sym.sqrt(2)

            wg_ij = group._wigner_pure(psi)
            assert sym.im(wg_ij).is_zero


@pytest.mark.parametrize("dim", DIMS)
def test_wigner_operator(dim: int):
    group = Su2Group(dim)

    for gm in group.gell_mann_basis:
        wg_gm = group._wigner_op(gm)
        assert sym.im(wg_gm).is_zero


@pytest.mark.parametrize("dim", DIMS)
def test_wigner_basis(dim: int):
    group = Su2Group(dim)

    theta, phi = sym.symbols("theta, phi", real=True)

    for wg in group.wigner_basis:
        logger.debug("%s", wg)

        for k in range(1, dim):
            assert wg.count(sym.cos(k * theta) ** 2) == 0
            assert wg.count(sym.sin(k * theta) ** 2) == 0

            for l in range(1, dim):
                assert wg.count(sym.cos(k * theta) * sym.sin(l * theta)) == 0


@pytest.mark.parametrize("dim", DIMS)
def test_wigner_transform(dim: int):
    group = Su2Group(dim)

    theta, phi = sym.symbols("theta, phi", real=True)

    wg = group.wigner_transform_expr
    logger.debug("%s", wg)

    for k in range(1, dim):
        assert wg.count(sym.cos(k * theta) ** 2) == 0
        assert wg.count(sym.sin(k * theta) ** 2) == 0

        for l in range(1, dim):
            assert wg.count(sym.cos(k * theta) * sym.sin(l * theta)) == 0

        assert wg.count(sym.cos(k * theta)) <= 1
        assert wg.count(sym.sin(k * theta)) <= 1
