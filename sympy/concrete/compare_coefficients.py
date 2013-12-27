from sympy.polys import Poly
from sympy.matrices import zeros


def build_matrix(P, coefficients):
    r"""Build a linear system.
    """
    M = zeros(P.degree()+1, len(coefficients))
    rhs = zeros(P.degree()+1, 1)
    n = P.gen

    for d in xrange(P.degree()+1):
        cmon = P.coeff_monomial(n**d)
        cmonp = Poly(cmon, gens=coefficients)
        # Fill in matrix elements
        for c, coe in enumerate(coefficients):
            xi = cmonp.coeff_monomial(coe)
            M[d,c] = xi
        # Put constants in the RHS
        chi = cmonp.coeff_monomial(1)
        rhs[d,0] = -chi

    return M, rhs
