from itertools import product
from typing import Optional, overload

import sympy as sym


@overload
def flat_add(x: sym.MatrixExpr) -> list[sym.MatrixExpr]: ...


def flat_add(x: sym.Expr) -> list[sym.Expr]:
    if isinstance(x, sym.MatrixExpr):
        shape = x.shape
        cls = type(x)

        adds: list[sym.MatrixExpr] = []

        for i in product(*map(range, shape)):
            adds_ij = flat_add(x[i])

            if len(adds_ij) > len(adds):
                adds += [
                    cls.zeros(*shape).as_mutable()
                    for _ in range(len(adds), len(adds_ij))
                ]

            for n, add_ij in enumerate(adds_ij):
                adds[n][i] = add_ij

        return [cls(add) for add in adds]

    x = sym.expand(x)

    if isinstance(x, sym.Add):
        return list(x.args)

    if x.is_zero:
        return []

    return [x]


w0, w1, C = sym.symbols("w0, w1, C", cls=sym.Wild, real=True)

_REPLACE_SIN_COS: list[tuple[sym.Expr, sym.Expr]] = [
    (C * sym.cos(w0) * sym.cos(w1), C * (sym.cos(w0 - w1) + sym.cos(w0 + w1)) / 2),
    (C * sym.sin(w0) * sym.sin(w1), C * (sym.cos(w0 - w1) - sym.cos(w0 + w1)) / 2),
    (C * sym.sin(w0) * sym.cos(w1), C * (sym.sin(w0 + w1) + sym.sin(w0 - w1)) / 2),
]


@overload
def trigsimp(x: sym.MatrixExpr) -> sym.MatrixExpr: ...


def trigsimp(x: sym.Expr) -> sym.Expr:
    def sin_cos_pow(x: sym.Expr) -> int:
        if isinstance(x, sym.Pow):
            x, n = x.args
            return n * sin_cos_pow(x)

        if isinstance(x, sym.Mul):
            return sum(sin_cos_pow(y) for y in x.args)

        if isinstance(x, sym.Add):
            return max(sin_cos_pow(y) for y in x.args)

        return int(isinstance(x, (sym.cos, sym.sin)))

    if isinstance(x, sym.MatrixExpr):
        r, c = x.shape
        cls = type(x)
        x = x.as_mutable()

        ij = list(product(range(r), range(c)))
        for i, j in ij:
            x[i, j] = trigsimp(x[i, j])

        return cls(x)

    adds: list[sym.Expr] = flat_add(x)

    n = 0
    while True:
        adds = flat_add(sum(x.replace(*_REPLACE_SIN_COS[n]) for x in adds))
        n = (n + 1) % len(_REPLACE_SIN_COS)

        if all(sin_cos_pow(x) <= 1 for x in adds):
            break

    return sum(adds)
