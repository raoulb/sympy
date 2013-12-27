"""Gosper's algorithm for hypergeometric summation. """
from __future__ import print_function, division

from sympy.core import S, Dummy, symbols
from sympy.core.compatibility import is_sequence, xrange
from sympy.polys import Poly, parallel_poly_from_expr, factor
from sympy.concrete.dispersion import dispersionset
from sympy.concrete.compare_coefficients import build_matrix
from sympy.solvers.linear import solve_general_linear
from sympy.utilities import flatten
from sympy.simplify import hypersimp


def gosper_normalize(f, g):
    lcf = f.LC()
    lcg = g.LC()
    F = f.monic()
    G = g.monic()
    Z = lcf / lcg

    p = F.one
    q = F.shift(-1)
    r = G.shift(-1)

    J = dispersionset(F, G)

    for j in sorted(J):
        gj = q.gcd(r.shift(j))
        # Update
        q = q.quo(gj)
        r = r.quo(gj.shift(-j))
        for i in xrange(0, j):
            p *= gj.shift(-i)

    q = q.mul_ground(Z)

    return (q, r, p)


def degree_bound(p, q, r):
    r"""Gosper degree bound for :math:`f_k`.
    """
    qpr = q.shift(1) + r
    qmr = q.shift(1) - r
    dqpr = qpr.degree()
    dqmr = qmr.degree()
    dp = S(p.degree())
    if dqpr <= dqmr:
        df = dp - dqmr
    else:
        k = p.gen
        a = S(qpr.coeff_monomial(k**dqpr))
        if dqpr > 0: # TODO: Recheck
            b = S(qmr.coeff_monomial(k**(dqpr-1)))
        else:
            b = 0
        if -2*b/a < 0:
            df = dp - dqpr + 1
        else:
            df = max(-2*b/a, dp-dqpr+1)
    return df


def rational_certificate(p, q, r):
    r"""
    """
    df = degree_bound(p, q, r)

    if df < 0:
        return None
        raise ValueError("Not Gosper summable")

    coeffs = symbols('c:%s' % (df + 1), cls=Dummy)
    domain = q.get_domain().inject(*coeffs)
    f = Poly(coeffs, p.gen, domain=domain)

    # Correct:
    eqn1 = q.shift(1)*f.shift(1) - r*f.shift(0) - p # OK
    #eqn1 = q.shift(1)*f - r*f.shift(-1) - p
    eqn = eqn1.as_poly(p.gen)

    M, v = build_matrix(eqn, coeffs)

    try:
        sol, params = solve_general_linear(M, v)
        sol, params = flatten(sol.tolist()), flatten(params.tolist())
    except:
        return None

    # Construct the solution g
    vals = [(var,val) for var, val in zip(coeffs, sol)]
    params = [(pa,0) for pa in params]
    fsol = f.shift(0).subs(vals+params) # OK
    #fsol = fsol.as_poly(p.gen, extension=True).shift(-1)

    return r/p * fsol


def gosper_main(a, n):
    r = hypersimp(a, n)
    if r is None:
        return None
        #raise ValueError("Could not simplify input into hypergeometric term")

    u, v = r.as_numer_denom()
    (u, v), opt = parallel_poly_from_expr((u, v), n, field=True, extension=True)

    q, r, p = gosper_normalize(u, v)

    R = rational_certificate(p, q, r)

    if R is None:
        return None

    return R * a


def gosper_sum(f, k):
    r"""
    Gosper's hypergeometric summation algorithm.

    Given a hypergeometric term ``f`` such that:

    .. math ::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f(n)` doesn't depend on `n`, returns `g_{n} - g(0)` where
    `g_{n+1} - g_n = f_n`, or ``None`` if `s_n` can not be expressed
    in closed form as a sum of hypergeometric terms.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_sum
    >>> from sympy.functions import factorial
    >>> from sympy.abc import n, k

    >>> f = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    >>> F = gosper_sum(f, (k, 0, n))
    >>> F
    (-factorial(n) + 2*factorial(2*n + 1))/factorial(2*n + 1)
    >>> F.subs(n, 2) == sum(f.subs(k, i) for i in [0, 1, 2])
    True
    >>> F = gosper_sum(f, (k, 3, n))
    >>> F
    (-60*factorial(n) + factorial(2*n + 1))/(60*factorial(2*n + 1))
    >>> F.subs(n, 5) == sum(f.subs(k, i) for i in [3, 4, 5])
    True

    References
    ==========

    .. [1] Marko Petkovsek, Herbert S. Wilf, Doron Zeilberger, A = B,
           AK Peters, Ltd., Wellesley, MA, USA, 1997, pp. 73--100
    """
    indefinite = False

    if is_sequence(k):
        k, a, b = k
    else:
        indefinite = True

    g = gosper_main(f, k)

    if g is None:
        return None

    if indefinite:
        result = g
    else:
        result = g.subs(k, b + 1) - g.subs(k, a)
        if result is S.NaN:
            try:
                result = g.limit(k, b + 1) - g.limit(k, a)
            except NotImplementedError:
                result = None

    return factor(result)
