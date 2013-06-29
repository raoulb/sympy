from sympy.core.basic import C
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Derivative
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Wild)
from sympy.core.sympify import sympify
from sympy.concrete.gosper import gosper_sum
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.polys import apart, PolynomialError
from sympy.solvers import solve


class Sum(Expr):
    """Represents unevaluated summation.

    Sum represents a finite or infinite series, with the first argument being
    the general form of terms in the series, and the second argument being
    (dummy_variable, start, end), with dummy_variable taking all integer values
    from start to end.  In accordance with long-standing mathematical
    convention, the end term is included in the summation.

    >>> from sympy.abc import k, m, n, x
    >>> from sympy import Sum, factorial, oo
    >>> Sum(k,(k,1,m))
    Sum(k, (k, 1, m))
    >>> Sum(k,(k,1,m)).doit()
    m**2/2 + m/2
    >>> Sum(k**2,(k,1,m))
    Sum(k**2, (k, 1, m))
    >>> Sum(k**2,(k,1,m)).doit()
    m**3/3 + m**2/2 + m/6
    >>> Sum(x**k,(k,0,oo))
    Sum(x**k, (k, 0, oo))
    >>> Sum(x**k,(k,0,oo)).doit()
    Piecewise((1/(-x + 1), Abs(x) < 1), (Sum(x**k, (k, 0, oo)), True))
    >>> Sum(x**k/factorial(k),(k,0,oo)).doit()
    exp(x)

    """

    __slots__ = ['is_commutative']

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.integrals.integrals import _process_limits

        # Any embedded piecewise functions need to be brought out to the
        # top level so that integration can go into piecewise mode at the
        # earliest possible moment.
        function = piecewise_fold(sympify(function))

        if function is S.NaN:
            return S.NaN

        if not symbols:
            raise ValueError("Summation variables must be given")

        limits, sign = _process_limits(*symbols)

        # Only limits with lower and upper bounds are supported; the indefinite Sum
        # is not supported
        if any(len(l) != 3 or None in l for l in limits):
            raise ValueError('Sum requires values for lower and upper bounds.')

        obj = Expr.__new__(cls, **assumptions)
        arglist = [sign*function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative  # limits already checked

        return obj

    @property
    def function(self):
        return self._args[0]

    @property
    def limits(self):
        return self._args[1:]

    @property
    def variables(self):
        """Return a list of the summation variables

        >>> from sympy import Sum
        >>> from sympy.abc import x, i
        >>> Sum(x**i, (i, 1, 3)).variables
        [i]
        """
        return [l[0] for l in self.limits]

    @property
    def free_symbols(self):
        from sympy.integrals.integrals import _free_symbols
        if self.function.is_zero:
            return set()
        return _free_symbols(self)

    @property
    def is_zero(self):
        """A Sum is only zero if its function is zero or if all terms
        cancel out. This only answers whether the summand zero."""

        return self.function.is_zero

    @property
    def is_number(self):
        """
        Return True if the Sum will result in a number, else False.

        Sums are a special case since they contain symbols that can
        be replaced with numbers. Whether the sum can be done or not in
        closed form is another issue. But answering whether the final
        result is a number is not difficult.

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y
        >>> Sum(x, (y, 1, x)).is_number
        False
        >>> Sum(1, (y, 1, x)).is_number
        False
        >>> Sum(0, (y, 1, x)).is_number
        True
        >>> Sum(x, (y, 1, 2)).is_number
        False
        >>> Sum(x, (y, 1, 1)).is_number
        False
        >>> Sum(x, (x, 1, 2)).is_number
        True
        >>> Sum(x*y, (x, 1, 2), (y, 1, 3)).is_number
        True
        """

        return self.function.is_zero or not self.free_symbols

    def as_dummy(self):
        from sympy.integrals.integrals import _as_dummy
        return _as_dummy(self)

    def doit(self, **hints):
        if hints.get('deep', True):
            f = self.function.doit(**hints)
        else:
            f = self.function

        for limit in self.limits:
            i, a, b = limit
            dif = b - a
            if dif.is_Integer and dif < 0:
                a, b = b + 1, a - 1
                f = -f

            f = eval_sum(f, (i, a, b))
            if f is None:
                return self

        if hints.get('deep', True):
            # eval_sum could return partially unevaluated
            # result with Piecewise.  In this case we won't
            # doit() recursively.
            if not isinstance(f, Piecewise):
                return f.doit(**hints)

        return f

    def _eval_adjoint(self):
        return Sum(self.function.adjoint(), *self.limits)

    def _eval_conjugate(self):
        return Sum(self.function.conjugate(), *self.limits)

    def _eval_derivative(self, x):
        """
        Differentiate wrt x as long as x is not in the free symbols of any of
        the upper or lower limits.

        Sum(a*b*x, (x, 1, a)) can be differentiated wrt x or b but not `a`
        since the value of the sum is discontinuous in `a`. In a case
        involving a limit variable, the unevaluated derivative is returned.
        """

        # diff already confirmed that x is in the free symbols of self, but we
        # don't want to differentiate wrt any free symbol in the upper or lower
        # limits
        # XXX remove this test for free_symbols when the default _eval_derivative is in
        if x not in self.free_symbols:
            return S.Zero

        # get limits and the function
        f, limits = self.function, list(self.limits)

        limit = limits.pop(-1)

        if limits:  # f is the argument to a Sum
            f = Sum(f, *limits)

        if len(limit) == 3:
            _, a, b = limit
            if x in a.free_symbols or x in b.free_symbols:
                return None
            df = Derivative(f, x, **{'evaluate': True})
            rv = Sum(df, limit)
            if limit[0] not in df.free_symbols:
                rv = rv.doit()
            return rv
        else:
            return NotImplementedError('Lower and upper bound expected.')

    def _eval_summation(self, f, x):
        return None

    def _eval_transpose(self):
        return Sum(self.function.transpose(), *self.limits)

    def euler_maclaurin(self, m=0, n=0, eps=0, eval_integral=True):
        """
        Return an Euler-Maclaurin approximation of self, where m is the
        number of leading terms to sum directly and n is the number of
        terms in the tail.

        With m = n = 0, this is simply the corresponding integral
        plus a first-order endpoint correction.

        Returns (s, e) where s is the Euler-Maclaurin approximation
        and e is the estimated error (taken to be the magnitude of
        the first omitted term in the tail):

            >>> from sympy.abc import k, a, b
            >>> from sympy import Sum
            >>> Sum(1/k, (k, 2, 5)).doit().evalf()
            1.28333333333333
            >>> s, e = Sum(1/k, (k, 2, 5)).euler_maclaurin()
            >>> s
            -log(2) + 7/20 + log(5)
            >>> from sympy import sstr
            >>> print sstr((s.evalf(), e.evalf()), full_prec=True)
            (1.26629073187415, 0.0175000000000000)

        The endpoints may be symbolic:

            >>> s, e = Sum(1/k, (k, a, b)).euler_maclaurin()
            >>> s
            -log(a) + log(b) + 1/(2*b) + 1/(2*a)
            >>> e
            Abs(-1/(12*b**2) + 1/(12*a**2))

        If the function is a polynomial of degree at most 2n+1, the
        Euler-Maclaurin formula becomes exact (and e = 0 is returned):

            >>> Sum(k, (k, 2, b)).euler_maclaurin()
            (b**2/2 + b/2 - 1, 0)
            >>> Sum(k, (k, 2, b)).doit()
            b**2/2 + b/2 - 1

        With a nonzero eps specified, the summation is ended
        as soon as the remainder term is less than the epsilon.
        """
        m = int(m)
        n = int(n)
        f = self.function
        assert len(self.limits) == 1
        i, a, b = self.limits[0]
        if a > b:
            a, b = b + 1, a - 1
            f = -f
        s = S.Zero
        if m:
            for k in range(m):
                term = f.subs(i, a + k)
                if (eps and term and abs(term.evalf(3)) < eps):
                    return s, abs(term)
                s += term
            a += m
        x = Dummy('x')
        I = C.Integral(f.subs(i, x), (x, a, b))
        if eval_integral:
            I = I.doit()
        s += I

        def fpoint(expr):
            if b is S.Infinity:
                return expr.subs(i, a), 0
            return expr.subs(i, a), expr.subs(i, b)
        fa, fb = fpoint(f)
        iterm = (fa + fb)/2
        g = f.diff(i)
        for k in xrange(1, n + 2):
            ga, gb = fpoint(g)
            term = C.bernoulli(2*k)/C.factorial(2*k)*(gb - ga)
            if (eps and term and abs(term.evalf(3)) < eps) or (k > n):
                break
            s += term
            g = g.diff(i, 2, simplify=False)
        return s + iterm, abs(term)

    def _eval_subs(self, old, new):
        from sympy.polys.polytools import Poly
        # from sympy.integrals.integrals import _eval_subs
        # return _eval_subs(self, old, new)
        summand, limits = self.function, list(self.limits)
        limits.reverse() # so that scoping matches standard mathematical practice for scoping

        # If one of the expressions we are replacing is used as a coordinate
        # one of two things happens.
        #   - the old variable first appears as a free variable
        #       so we perform all free substitutions before it becomes
        #       a coordinate.
        #   - the old variable first appears as a coordinate, in
        #       which case we change that coordinate.
        if not isinstance(old,C.Symbol) or old.free_symbols.intersection(self.free_symbols):
            sub_into_summand = True
            for i, xab in enumerate(limits):
                assert len(xab) == 3, "undefined summation limit in substitution"
                (x,a,b) = xab
                limits[i] = (x, a._subs(old, new),b._subs(old,new))
                if 0!= len(x.free_symbols.intersection(old.free_symbols)):
                    sub_into_summand = False
                    break
            if sub_into_summand:
                summand = summand.subs(old, new)
        else:
            new_ns = new.free_symbols.difference(self.free_symbols)
            assert 1==len(new_ns), "no free symbols as dummies"
            new_n = new_ns.pop(); del new_ns
            assert new.is_polynomial(new_n) \
                and Poly(new,new_n).degree() == 1, \
                "Only linear substitutions allowed for Sum"
            assert new.coeff(new_n,1) in [1,-1], \
                "Sum substitution slope must be in [-1,1]."
            found = False
            for i, xab in enumerate(limits):
                if len(xab) != 3:
                    continue
                (x,a,b) = xab
                if not found:
                    if old == x:
                        found = True
                        assert old != new_n
                        sols = solve(new-old,new_n)
                        assert 1 == len(sols)
                        sol = sols[0]
                        limits[i] = (new_n, sol.subs(old,a) ,sol.subs(old,b))
                    else:
                        assert not old.free_symbols.intersection(a.free_symbols)
                        assert not old.free_symbols.intersection(b.free_symbols)
                else:
                    assert x != old, "repeated dummy variable in Sum"
                    limits[i] = (x,a.subs(old,new),b.subs(old,new))
            summand = summand.subs(old, new)
        return self.func(summand, *limits)

    def _eval_factor(self, **hints):
        summand = self.function.factor(**hints)
        keep_inside = []
        pull_outside = []
        if summand.is_Mul and summand.is_commutative:
            for i in summand.args:
                if not i.atoms(C.Symbol).intersection(self.variables):
                    pull_outside.append(i)
                else:
                    keep_inside.append(i)
            return C.Mul(*pull_outside) * Sum(C.Mul(*keep_inside), *self.limits)
        return self

    def _eval_expand_basic(self, **hints):
        summand = self.function.expand(**hints)
        if summand.is_Add and summand.is_commutative:
            return C.Add(*[ Sum(i, *self.limits) for i in summand.args ])
        elif summand != self.function:
            return Sum(summand, *self.limits)
        return self


def summation(f, *symbols, **kwargs):
    r"""
    Compute the summation of f with respect to symbols.

    The notation for symbols is similar to the notation used in Integral.
    summation(f, (i, a, b)) computes the sum of f with respect to i from a to b,
    i.e.,

    ::

                                    b
                                  ____
                                  \   `
        summation(f, (i, a, b)) =  )    f
                                  /___,
                                  i = a

    If it cannot compute the sum, it returns an unevaluated Sum object.
    Repeated sums can be computed by introducing additional symbols tuples::

    >>> from sympy import summation, oo, symbols, log
    >>> i, n, m = symbols('i n m', integer=True)

    >>> summation(2*i - 1, (i, 1, n))
    n**2
    >>> summation(1/2**i, (i, 0, oo))
    2
    >>> summation(1/log(n)**n, (n, 2, oo))
    Sum(log(n)**(-n), (n, 2, oo))
    >>> summation(i, (i, 0, n), (n, 0, m))
    m**3/6 + m**2/2 + m/3

    >>> from sympy.abc import x
    >>> from sympy import factorial
    >>> summation(x**n/factorial(n), (n, 0, oo))
    exp(x)

    """
    return Sum(f, *symbols, **kwargs).doit(deep=False)


def telescopic_direct(L, R, n, limits):
    """Returns the direct summation of the terms of a telescopic sum

    L is the term with lower index
    R is the term with higher index
    n difference between the indexes of L and R

    For example:

    >>> from sympy.concrete.summations import telescopic_direct
    >>> from sympy.abc import k, a, b
    >>> telescopic_direct(1/k, -1/(k+2), 2, (k, a, b))
    -1/(b + 2) - 1/(b + 1) + 1/(a + 1) + 1/a

    """
    (i, a, b) = limits
    s = 0
    for m in xrange(n):
        s += L.subs(i, a + m) + R.subs(i, b - m)
    return s


def telescopic(L, R, limits):
    '''Tries to perform the summation using the telescopic property

    return None if not possible
    '''
    (i, a, b) = limits
    if L.is_Add or R.is_Add:
        return None

    # We want to solve(L.subs(i, i + m) + R, m)
    # First we try a simple match since this does things that
    # solve doesn't do, e.g. solve(f(k+m)-f(k), m) fails

    k = Wild("k")
    sol = (-R).match(L.subs(i, i + k))
    s = None
    if sol and k in sol:
        s = sol[k]
        if not (s.is_Integer and L.subs(i, i + s) == -R):
            #sometimes match fail(f(x+2).match(-f(x+k))->{k: -2 - 2x}))
            s = None

    # But there are things that match doesn't do that solve
    # can do, e.g. determine that 1/(x + m) = 1/(1 - x) when m = 1

    if s is None:
        m = Dummy('m')
        try:
            sol = solve(L.subs(i, i + m) + R, m) or []
        except NotImplementedError:
            return None
        sol = [si for si in sol if si.is_Integer and
               (L.subs(i, i + si) + R).expand().is_zero]
        if len(sol) != 1:
            return None
        s = sol[0]

    if s < 0:
        return telescopic_direct(R, L, abs(s), (i, a, b))
    elif s > 0:
        return telescopic_direct(L, R, s, (i, a, b))


def eval_sum(f, limits):
    from sympy.concrete.delta import deltasummation, _has_simple_delta
    from sympy.functions import KroneckerDelta

    (i, a, b) = limits
    if f is S.Zero:
        return S.Zero
    if i not in f.free_symbols:
        return f*(b - a + 1)
    if a == b:
        return f.subs(i, a)

    if f.has(KroneckerDelta) and _has_simple_delta(f, limits[0]):
        return deltasummation(f, limits)

    dif = b - a
    definite = dif.is_Integer
    # Doing it directly may be faster if there are very few terms.
    if definite and (dif < 100):
        return eval_sum_direct(f, (i, a, b))
    # Try to do it symbolically. Even when the number of terms is known,
    # this can save time when b-a is big.
    # We should try to transform to partial fractions
    value = eval_sum_symbolic(f.expand(), (i, a, b))
    if value is not None:
        return value
    # Do it directly
    if definite:
        return eval_sum_direct(f, (i, a, b))


def eval_sum_direct(expr, limits):
    (i, a, b) = limits

    dif = b - a
    return C.Add(*[expr.subs(i, a + j) for j in xrange(dif + 1)])


def eval_sum_symbolic(f, limits):
    (i, a, b) = limits
    if not f.has(i):
        return f*(b - a + 1)

    # Linearity
    if f.is_Mul:
        L, R = f.as_two_terms()

        if not L.has(i):
            sR = eval_sum_symbolic(R, (i, a, b))
            if sR:
                return L*sR

        if not R.has(i):
            sL = eval_sum_symbolic(L, (i, a, b))
            if sL:
                return R*sL

        try:
            f = apart(f, i)  # see if it becomes an Add
        except PolynomialError:
            pass

    if f.is_Add:
        L, R = f.as_two_terms()
        lrsum = telescopic(L, R, (i, a, b))

        if lrsum:
            return lrsum

        lsum = eval_sum_symbolic(L, (i, a, b))
        rsum = eval_sum_symbolic(R, (i, a, b))

        if None not in (lsum, rsum):
            return lsum + rsum

    # Polynomial terms with Faulhaber's formula
    n = Wild('n')
    result = f.match(i**n)

    if result is not None:
        n = result[n]

        if n.is_Integer:
            if n >= 0:
                return ((C.bernoulli(n + 1, b + 1) - C.bernoulli(n + 1, a))/(n + 1)).expand()
            elif a.is_Integer and a >= 1:
                if n == -1:
                    return C.harmonic(b) - C.harmonic(a - 1)
                else:
                    return C.harmonic(b, abs(n)) - C.harmonic(a - 1, abs(n))

    if not (a.has(S.Infinity, S.NegativeInfinity) or
            b.has(S.Infinity, S.NegativeInfinity)):
        # Geometric terms
        c1 = C.Wild('c1', exclude=[i])
        c2 = C.Wild('c2', exclude=[i])
        c3 = C.Wild('c3', exclude=[i])

        e = f.match(c1**(c2*i + c3))

        if e is not None:
            p = (c1**c3).subs(e)
            q = (c1**c2).subs(e)

            r = p*(q**a - q**(b + 1))/(1 - q)
            l = p*(b - a + 1)

            return Piecewise((l, Eq(q, S.One)), (r, True))

        r = gosper_sum(f, (i, a, b))

        if not r in (None, S.NaN):
            return r

    return eval_sum_hyper(f, (i, a, b))


def _eval_sum_hyper(f, i, a):
    """ Returns (res, cond). Sums from a to oo. """
    from sympy.functions import hyper
    from sympy.simplify import hyperexpand, hypersimp, fraction, simplify
    from sympy.polys.polytools import Poly, factor

    if a != 0:
        return _eval_sum_hyper(f.subs(i, i + a), i, 0)

    if f.subs(i, 0) == 0:
        if simplify(f.subs(i, Dummy('i', integer=True, positive=True))) == 0:
            return S(0), True
        return _eval_sum_hyper(f.subs(i, i + 1), i, 0)

    hs = hypersimp(f, i)
    if hs is None:
        return None

    numer, denom = fraction(factor(hs))
    top, topl = numer.as_coeff_mul(i)
    bot, botl = denom.as_coeff_mul(i)
    ab = [top, bot]
    factors = [topl, botl]
    params = [[], []]
    for k in range(2):
        for fac in factors[k]:
            mul = 1
            if fac.is_Pow:
                mul = fac.exp
                fac = fac.base
                if not mul.is_Integer:
                    return None
            p = Poly(fac, i)
            if p.degree() != 1:
                return None
            m, n = p.all_coeffs()
            ab[k] *= m**mul
            params[k] += [n/m]*mul

    # Add "1" to numerator parameters, to account for implicit n! in
    # hypergeometric series.
    ap = params[0] + [1]
    bq = params[1]
    x = ab[0]/ab[1]
    h = hyper(ap, bq, x)

    return f.subs(i, 0)*hyperexpand(h), h.convergence_statement


def eval_sum_hyper(f, (i, a, b)):
    from sympy import oo, And

    if b != oo:
        if a == -oo:
            res = _eval_sum_hyper(f.subs(i, -i), i, -b)
            if res is not None:
                return Piecewise(res, (Sum(f, (i, a, b)), True))
        else:
            res1 = _eval_sum_hyper(f, i, a)
            res2 = _eval_sum_hyper(f, i, b + 1)
            if res1 is None or res2 is None:
                return None
            (res1, cond1), (res2, cond2) = res1, res2
            cond = And(cond1, cond2)
            if cond is False:
                return None
        return Piecewise((res1 - res2, cond), (Sum(f, (i, a, b)), True))

    if a == -oo:
        res1 = _eval_sum_hyper(f.subs(i, -i), i, 1)
        res2 = _eval_sum_hyper(f, i, 0)
        if res1 is None or res2 is None:
            return None
        res1, cond1 = res1
        res2, cond2 = res2
        cond = And(cond1, cond2)
        if cond is False:
            return None
        return Piecewise((res1 + res2, cond), (Sum(f, (i, a, b)), True))

    # Now b == oo, a != -oo
    res = _eval_sum_hyper(f, i, a)
    if res is not None:
        r, c = res
        if c is False:
            if r.is_number:
                if f.is_positive or f.is_zero:
                    return S.Infinity
                elif f.is_negative:
                    return S.NegativeInfinity
            return None
        return Piecewise(res, (Sum(f, (i, a, b)), True))
