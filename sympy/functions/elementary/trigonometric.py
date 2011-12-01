from sympy.core.add import Add
from sympy.core.numbers import Rational, Float
from sympy.core.basic import C, sympify, cacheit
from sympy.core.singleton import S
from sympy.core.function import Function, ArgumentIndexError
from miscellaneous import sqrt
from hyperbolic import sinh, csch, cosh, sech, tanh, coth

###############################################################################
########################## TRIGONOMETRIC FUNCTIONS ############################
###############################################################################

class TrigonometricFunction(Function):
    """Base class for trigonometric functions. """

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit


def _peeloff_pi(arg):
    """
    Split ARG into two parts, a "rest" and a multiple of pi/2.
    This assumes ARG to be an Add.
    The multiple of pi returned in the second position is always a Rational.

    Examples:
    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi as peel
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> peel(x + pi/2)
    (x, pi/2)
    >>> peel(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, pi/2)
    """
    for a in Add.make_args(arg):
        if a is S.Pi:
            K = S.One
            break
        elif a.is_Mul:
            K, p = a.as_two_terms()
            if p is S.Pi and K.is_Rational:
                break
    else:
        return arg, S.Zero

    m1 = (K % S.Half) * S.Pi
    m2 = K*S.Pi - m1
    return arg - m2, m2

def _pi_coeff(arg, cycles=1):
    """
    When arg is a Number times pi (e.g. 3*pi/2) then return the Number
    normalized to be in the range [0, 2], else None.

    When an even multiple of pi is encountered, if it is multiplying
    something with known parity then the multiple is returned as 0 otherwise
    as 2.

    Examples:
    >>> from sympy.functions.elementary.trigonometric import _pi_coeff as coeff
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> coeff(3*x*pi)
    3*x
    >>> coeff(11*pi/7)
    11/7
    >>> coeff(-11*pi/7)
    3/7
    >>> coeff(4*pi)
    0
    >>> coeff(5*pi)
    1
    >>> coeff(5.0*pi)
    1
    >>> coeff(5.5*pi)
    3/2
    >>> coeff(2 + pi)

    """
    arg = sympify(arg)
    if arg is S.Pi:
        return S.One
    elif not arg:
        return S.Zero
    elif arg.is_Mul:
        cx = arg.coeff(S.Pi)
        if cx:
            c, x = cx.as_coeff_Mul() # pi is not included as coeff
            if c.is_Float:
                # recast exact binary fractions to Rationals
                m = int(c*2)
                if Float(float(m)/2) == c:
                    c = Rational(m, 2)
            if x is not S.One or not (c.is_Rational and c.q != 1):
                if x.is_integer:
                    c2 = c % 2
                    if c2 == 1:
                        return x
                    elif not c2:
                        if x.is_even is not None: # known parity
                            return S.Zero
                        return 2*x
                    else:
                        return c2*x
                return cx
            else:
                return Rational(c.p % (2*c.q), c.q)


class sin(TrigonometricFunction):
    """
    Usage
    =====
      sin(x) -> Returns the sine of x (measured in radians)

    Notes
    =====
        sin(x) will evaluate automatically in the case x
        is a multiple of pi, pi/2, pi/3, pi/4 and pi/6.

    Examples
    ========
        >>> from sympy import sin, pi
        >>> from sympy.abc import x
        >>> sin(x**2).diff(x)
        2*x*cos(x**2)
        >>> sin(1).diff(x)
        0
        >>> sin(pi)
        0
        >>> sin(pi/2)
        1
        >>> sin(pi/6)
        1/2

    See also
    ========
       L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

       External links
       --------------

         U{Definitions in trigonometry<http://planetmath.org/encyclopedia/DefinitionsInTrigonometry.html>}
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.Zero
            elif arg is S.Infinity:
                return

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return S.ImaginaryUnit * C.sinh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*S.Pi
                if narg != arg:
                    return cls(narg)
                return None

            cst_table_some = {
                2 : S.One,
                3 : S.Half*sqrt(3),
                4 : S.Half*sqrt(2),
                6 : S.Half,
            }

            cst_table_more = {
                (1, 5) : sqrt((5 - sqrt(5)) / 8),
                (2, 5) : sqrt((5 + sqrt(5)) / 8)
            }

            p = pi_coeff.p
            q = pi_coeff.q

            Q, P = p // q, p % q

            try:
                result = cst_table_some[q]
            except KeyError:
                if abs(P) > q // 2:
                    P = q - P

                try:
                    result = cst_table_more[(P, q)]
                except KeyError:
                    if P != p:
                        result = cls(C.Rational(P, q)*S.Pi)
                    else:
                        return None

            if Q % 2 == 1:
                return -result
            else:
                return result

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                return sin(m)*cos(x)+cos(m)*sin(x)

        if arg.func is asin:
            return arg.args[0]

        if arg.func is atan:
            x = arg.args[0]
            return x / sqrt(1 + x**2)

        if arg.func is acos:
            x = arg.args[0]
            return sqrt(1 - x**2)

        if arg.func is acot:
            x = arg.args[0];
            return 1 / (sqrt(1 + 1 / x**2) * x)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2
            return (-1)**k*x**(2*k+1)/C.factorial(2*k+1)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return cos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return asin


    def _eval_rewrite_as_exp(self, arg):
        exp, I = C.exp, S.ImaginaryUnit
        return (exp(arg*I) - exp(-arg*I)) / (2*I)

    def _eval_rewrite_as_cos(self, arg):
        return -cos(arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        tan_half = tan(S.Half*arg)
        return 2*tan_half/(1 + tan_half**2)

    def _eval_rewrite_as_cot(self, arg):
        cot_half = cot(S.Half*arg)
        return 2*cot_half/(1 + cot_half**2)

    def _eval_rewrite_as_sec(self, arg):
        return -1 / sec(arg + S.Pi/2)

    def _eval_rewrite_as_csc(self, arg):
        return 1 / csc(arg)

    def _eval_rewrite_as_sinh(self, arg):
        return -S.ImaginaryUnit*sinh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_csch(self, arg):
        return -S.ImaginaryUnit / csch(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cosh(self, arg):
        return -cosh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_sech(self, arg):
        return -1 / sech(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half = tanh(S.ImaginaryUnit*S.Half*arg)
        return 2*S.ImaginaryUnit*tanh_half / (tanh_half**2 - 1)

    def _eval_rewrite_as_coth(self, arg):
        coth_half = coth(S.ImaginaryUnit*S.Half*arg)
        return 2*S.ImaginaryUnit*coth_half / (1 - coth_half**2)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (sin(re)*cosh(im), cos(re)*sinh(im))


    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_mul()
            if not (coeff is S.One) and coeff.is_Integer and terms:
                x = arg._new_rawargs(*terms)
                y = (coeff-1)*x
        if x is not None:
            return (sin(x)*cos(y) + sin(y)*cos(x)).expand(trig=True)
        return sin(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_real:
            return True


    def _sage_(self):
        import sage.all as sage
        return sage.sin(self.args[0]._sage_())


class cos(TrigonometricFunction):
    """
    Usage
    =====
      cos(x) -> Returns the cosine of x (measured in radians)

    Notes
    =====
        cos(x) will evaluate automatically in the case x
        is a multiple of pi, pi/2, pi/3, pi/4 and pi/6.

    Examples
    ========
        >>> from sympy import cos, pi
        >>> from sympy.abc import x
        >>> cos(x**2).diff(x)
        -2*x*sin(x**2)
        >>> cos(1).diff(x)
        0
        >>> cos(pi)
        -1
        >>> cos(pi/2)
        0
        >>> cos(2*pi/3)
        -1/2

    See also
    ========
       L{sin}, L{csc}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

       External links
       --------------

         U{Definitions in trigonometry<http://planetmath.org/encyclopedia/DefinitionsInTrigonometry.html>}
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.One
            elif arg is S.Infinity:
                return

        if arg.could_extract_minus_sign():
            return cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return C.cosh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if not pi_coeff.is_Rational:
                if pi_coeff.is_integer:
                    return (S.NegativeOne)**pi_coeff
                narg = pi_coeff*S.Pi
                if narg != arg:
                    return cls(narg)
                return None

            cst_table_some = {
                1 : S.One,
                2 : S.Zero,
                3 : S.Half,
                4 : S.Half*sqrt(2),
                6 : S.Half*sqrt(3),
            }

            cst_table_more = {
                (1, 5) : (sqrt(5) + 1)/4,
                (2, 5) : (sqrt(5) - 1)/4
            }

            p = pi_coeff.p
            q = pi_coeff.q

            Q, P = 2*p // q, p % q

            try:
                result = cst_table_some[q]
            except KeyError:
                if abs(P) > q // 2:
                    P = q - P

                try:
                    result = cst_table_more[(P, q)]
                except KeyError:
                    if P != p:
                        result = cls(C.Rational(P, q)*S.Pi)
                    else:
                        return None

            if Q % 4 in (1, 2):
                return -result
            else:
                return result

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                return cos(m)*cos(x)-sin(m)*sin(x)

        if arg.func is acos:
            return arg.args[0]

        if arg.func is atan:
            x = arg.args[0]
            return 1 / sqrt(1 + x**2)

        if arg.func is asin:
            x = arg.args[0]
            return sqrt(1 - x ** 2)

        if arg.func is acot:
            x = arg.args[0]
            return 1 / sqrt(1 + 1 / x**2)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2
            return (-1)**k*x**(2*k)/C.factorial(2*k)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -sin(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return acos


    def _eval_rewrite_as_exp(self, arg):
        exp, I = C.exp, S.ImaginaryUnit
        return (exp(arg*I) + exp(-arg*I)) / 2

    def _eval_rewrite_as_sin(self, arg):
        return sin(arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        tan_half_sq = tan(S.Half*arg)**2
        return (1-tan_half_sq) / (1+tan_half_sq)

    def _eval_rewrite_as_cot(self, arg):
        cot_half_sq = cot(S.Half*arg)**2
        return (cot_half_sq-1) / (cot_half_sq+1)

    def _eval_rewrite_as_sec(self, arg):
        return 1 / sec(arg)

    def _eval_rewrite_as_csc(self, arg):
        return 1 / csc(arg + s.Pi/2)

    def _eval_rewrite_as_sinh(self, arg):
        return -S.ImaginaryUnit*sinh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_csch(self, arg):
        return -S.ImaginaryUnit / csch(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_cosh(self, arg):
        return cosh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_sech(self, arg):
        return 1 / sech(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half_sq = tanh(S.ImaginaryUnit*S.Half*arg)**2
        return (1 + tanh_half_sq) / (1 - tanh_half_sq)

    def _eval_rewrite_as_coth(self, arg):
        coth_half_sq = coth(S.ImaginaryUnit*S.Half*arg)**2
        return (coth_half_sq + 1) / (coth_half_sq - 1)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (cos(re)*cosh(im), -sin(re)*sinh(im))


    def _eval_expand_trig(self, deep=True, **hints):
        # RECHECK
        if deep:
            arg = self.args[0].expand()
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
            return (cos(x)*cos(y) - sin(y)*sin(x)).expand(trig=True)
        else:
            coeff, terms = arg.as_coeff_mul()
            if not (coeff is S.One) and coeff.is_Integer and terms:
                x = arg._new_rawargs(*terms)
                return C.chebyshevt(coeff, cos(x))
        return cos(arg)


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return S.One
        else:
            return self.func(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_real:
            return True


    def _sage_(self):
        import sage.all as sage
        return sage.cos(self.args[0]._sage_())


class tan(TrigonometricFunction):
    """
    Usage
    =====
      tan(x) -> Returns the tangent of x (measured in radians)

    Notes
    =====
        tan(x) will evaluate automatically in the case x is a
        multiple of pi.

    Examples
    ========
        >>> from sympy import tan
        >>> from sympy.abc import x
        >>> tan(x**2).diff(x)
        2*x*(tan(x**2)**2 + 1)
        >>> tan(1).diff(x)
        0

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

       External links
       --------------

         U{Definitions in trigonometry<http://planetmath.org/encyclopedia/DefinitionsInTrigonometry.html>}
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.Zero

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return S.ImaginaryUnit * C.tanh(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*S.Pi
                if narg != arg:
                    return cls(narg)
                return None

            cst_table = {
                2 : S.ComplexInfinity,
                3 : sqrt(3),
                4 : S.One,
                6 : 1 / sqrt(3),
            }

            try:
                result = cst_table[pi_coeff.q]

                if (2*pi_coeff.p // pi_coeff.q) % 4 in (1, 3):
                    return -result
                else:
                    return result
            except KeyError:
                if pi_coeff.p > pi_coeff.q:
                    p, q = pi_coeff.p % pi_coeff.q, pi_coeff.q
                    if 2 * p > q:
                        return -cls(Rational(q - p, q)*S.Pi)
                    return cls(Rational(p, q)*S.Pi)

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                if (m*2/S.Pi) % 2 == 0:
                    return tan(x)
                else:
                    return -cot(x)

        if arg.func is atan:
            return arg.args[0]

        if arg.func is asin:
            x = arg.args[0]
            return x / sqrt(1 - x**2)

        if arg.func is acos:
            x = arg.args[0]
            return sqrt(1 - x**2) / x

        if arg.func is acot:
            x = arg.args[0]
            return 1 / x


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2 + 1
            return (-1)**(k-1) * 2**(2*k) * (2**(2*k)-1) * C.bernoulli(2*k) / C.factorial(2*k) * x**(2*k-1)


    def fdiff(self, argindex=1):
        if argindex==1:
            return sec(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return atan


    def _eval_nseries(self, x, n, logx):
    # RECHECK
        i = self.args[0].limit(x, 0)*2/S.Pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return Function._eval_nseries(self, x, n=n, logx=logx)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = cos(2*re) + cosh(2*im)
        return (sin(2*re)/denom, sinh(2*im)/denom)


    def _eval_expand_trig(self, deep=True, **hints):
    # RECHECK
        return self


    def _eval_rewrite_as_exp(self, arg):
        exp, I = C.exp, S.ImaginaryUnit
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(neg_exp-pos_exp)/(neg_exp+pos_exp)

    def _eval_rewrite_as_sin(self, arg):
        return sin(arg) / sin(S.Pi/2 + arg)

    def _eval_rewrite_as_cos(self, arg):
        return -cos(arg + S.Pi/2)/cos(arg)

    def _eval_rewrite_as_cot(self, arg):
        return 1/cot(arg)

    def _eval_rewrite_as_sec(self, arg):
        return -sec(arg) / sec(S.Pi/2 + arg)

    def _eval_rewrite_as_csc(self, arg):
        return csc(S.Pi/2 + arg) / csc(arg)

    def _eval_rewrite_as_sinh(self, arg):
        return sinh(S.ImaginaryUnit*arg) / sinh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_csch(self, arg):
        return scsh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2) / csch(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cosh(self, arg):
        return -(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2) / cosh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_sech(self, arg):
        return -sech(S.ImaginaryUnit*arg) / (S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_tanh(self, arg):
        return -S.ImaginaryUnit*tanh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_coth(self, arg):
        return -S.ImaginaryUnit / coth(S.ImaginaryUnit*arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_imaginary:
            return True


    def _eval_subs(self, old, new):
    # RECHECK
        if self == old:
            return new
        arg = self.args[0]
        argnew = arg.subs(old, new)
        if arg != argnew and (argnew/(S.Pi/2)).is_odd:
            return S.NaN
        return tan(argnew)


    def _sage_(self):
        import sage.all as sage
        return sage.tan(self.args[0]._sage_())


class cot(TrigonometricFunction):
    """
    Usage
    =====
      cot(x) -> Returns the cotangent of x (measured in radians)


    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            if arg is S.Zero:
                return S.ComplexInfinity

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return -S.ImaginaryUnit * C.coth(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.ComplexInfinity

            if not pi_coeff.is_Rational:
                narg = pi_coeff*S.Pi
                if narg != arg:
                    return cls(narg)
                return None

            cst_table = {
                2 : S.Zero,
                3 : 1 / sqrt(3),
                4 : S.One,
                6 : sqrt(3)
            }

            try:
                result = cst_table[pi_coeff.q]

                if (2*pi_coeff.p // pi_coeff.q) % 4 in (1, 3):
                    return -result
                else:
                    return result
            except KeyError:
                if pi_coeff.p > pi_coeff.q:
                    p, q = pi_coeff.p % pi_coeff.q, pi_coeff.q
                    if 2 * p > q:
                        return -cls(Rational(q - p, q)*S.Pi)
                    return cls(Rational(p, q)*S.Pi)

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                if (m*2/S.Pi) % 2 == 0:
                    return cot(x)
                else:
                    return -tan(x)

        if arg.func is acot:
            return arg.args[0]

        if arg.func is atan:
            x = arg.args[0]
            return 1 / x

        if arg.func is asin:
            x = arg.args[0]
            return sqrt(1 - x**2) / x

        if arg.func is acos:
            x = arg.args[0]
            return x / sqrt(1 - x**2)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2 + 1
            return (-1)**k * 2**(2*k) * C.bernoulli(2*k) / C.factorial(2*k) * x**(2*k-1)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -csc(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return acot


    def _eval_nseries(self, x, n, logx):
    # RECHECK
        i = self.args[0].limit(x, 0)/S.Pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return Function._eval_nseries(self, x, n=n, logx=logx)


    def _eval_conjugate(self):
        # Why the assertion here (and only here?)
        assert len(self.args) == 1
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = cos(2*re) - cosh(2*im)
        return (-sin(2*re)/denom, sinh(2*im)/denom)


    def _eval_rewrite_as_exp(self, arg):
        exp, I = C.exp, S.ImaginaryUnit
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(pos_exp+neg_exp) / (pos_exp-neg_exp)

    def _eval_rewrite_as_sin(self, arg):
        return sin(arg + S.Pi/2) / sin(arg)

    def _eval_rewrite_as_cos(self, arg):
        return -cos(arg) / cos(arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        return 1/tan(arg)

    def _eval_rewrite_as_sec(self, arg):
        return sec(arg + S.Pi/2) / sec(arg)

    def _eval_rewrite_as_csc(self, arg):
        return csc(arg) / csc(arg + S.Pi/2)

    def _eval_rewrite_as_sinh(self, arg):
        return sinh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2) / sinh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_csch(self, arg):
        return csch(S.ImaginaryUnit*arg) / csch(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_cosh(self, arg):
        return -coth(S.ImaginaryUnit*arg) / coth(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_sech(self, arg):
        return -sech(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2) / sech(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_tanh(self, arg):
        return S.ImaginaryUnit / tanh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_coth(self, arg):
        return S.ImaginaryUnit*coth(S.ImaginaryUnit*arg)


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_subs(self, old, new):
    # RECHECK
        if self == old:
            return new
        arg = self.args[0]
        argnew = arg.subs(old, new)
        if arg != argnew and (argnew/S.Pi).is_integer:
            return S.NaN
        return cot(argnew)


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_imaginary:
            return True


    def _sage_(self):
        import sage.all as sage
        return sage.cot(self.args[0]._sage_())


class sec(TrigonometricFunction):
    """
    Usage
    =====
      sec(x) -> Returns the secant of x (measured in radians)

    Notes
    =====
        sec(x) will evaluate automatically in the case x is a
        multiple of pi.

    Examples
    ========
        >>> from sympy import sec
        >>> from sympy.abc import x
        >>> sec(x**2).diff(x)
        2*x*tan(x**2)*sec(x**2)
        >>> sec(1).diff(x)
        0

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

       External links
       --------------

         U{Definitions in trigonometry<http://planetmath.org/encyclopedia/DefinitionsInTrigonometry.html>}
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.One

        if arg.could_extract_minus_sign():
            return cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return C.sech(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        #if pi_coeff is not None:
            # if pi_coeff.is_integer:
        #         return S.Zero

        #     if not pi_coeff.is_Rational:
        #         narg = pi_coeff*S.Pi
        #         if narg != arg:
        #             return cls(narg)
        #         return None

        #     cst_table = {
        #         2 : S.ComplexInfinity,
        #         3 : sqrt(3),
        #         4 : S.One,
        #         6 : 1 / sqrt(3),
        #     }

        #     try:
        #         result = cst_table[pi_coeff.q]

        #         if (2*pi_coeff.p // pi_coeff.q) % 4 in (1, 3):
        #             return -result
        #         else:
        #             return result
        #     except KeyError:
        #         if pi_coeff.p > pi_coeff.q:
        #             p, q = pi_coeff.p % pi_coeff.q, pi_coeff.q
        #             if 2 * p > q:
        #                 return -cls(Rational(q - p, q)*S.Pi)
        #             return cls(Rational(p, q)*S.Pi)

        # if arg.is_Add:
        #     x, m = _peeloff_pi(arg)
        #     if m:
        #         if (m*2/S.Pi) % 2 == 0:
        #             return tan(x)
        #         else:
        #             return -cot(x)

        if arg.func is asec:
            return arg.args[0]

        if arg.func is asin:
            x = arg.args[0]
            return 1 / sqrt(1 - x**2)

        if arg.func is acos:
            x = arg.args[0]
            return 1 / x

        if arg.func is acot:
            x = arg.args[0]
            return sqrt(1 + x**2) / x

        # TODO
        # Other inverses


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            raise NotImplementedError("Euler numbers needed")
            #x = sympify(x)
            #k = n // 2
            #return (-1)**k * C.euler(2*k) / C.factorial(2*k) * x**(2*k)


    def fdiff(self, argindex=1):
        if argindex==1:
            return self*tan(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return asec


    def _eval_nseries(self, x, n, logx):
    # RECHECK
        pass
        # i = self.args[0].limit(x, 0)*2/S.Pi
        # if i and i.is_Integer:
        #     return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        # return Function._eval_nseries(self, x, n=n, logx=logx)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = cos(2*re) + cosh(2*im)
        return (2*cos(re)*cosh(im)/denom, 2*sin(re)*sinh(im)/denom)


    def _eval_expand_trig(self, deep=True, **hints):
    # RECHECK
        return self


    def _eval_rewrite_as_exp(self, arg):
        exp, I = C.exp, S.ImaginaryUnit
        return 2 / (exp(arg*I) + exp(-arg*I))

    def _eval_rewrite_as_sin(self, arg):
        return 1 / sin(arg + S.Pi/2)

    def _eval_rewrite_as_cos(self, arg):
        return 1 / cos(arg)

    def _eval_rewrite_as_tan(self, arg):
        tan_half_sq = tan(S.Half*arg)**2
        return (1+tan_half_sq) / (1-tan_half_sq)

    def _eval_rewrite_as_cot(self, arg):
        cot_half_sq = cot(S.Half*arg)**2
        return (cot_half_sq+1) / (cot_half_sq-1)

    def _eval_rewrite_as_csc(self, arg):
        return csc(arg + S.Pi/2)

    def _eval_rewrite_as_sinh(self, arg):
        return S.ImaginaryUnit / sinh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_csch(self, arg):
        return S.ImaginaryUnit*csch(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_cosh(self, arg):
        return 1 / cosh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_sech(self, arg):
        return sech(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half_sq = tanh(S.Half*S.ImaginaryUnit*arg)**2
        return (1 - tanh_half_sq) / (1 + tanh_half_sq)

    def _eval_rewrite_as_coth(self, arg):
        coth_half_sq = coth(S.Half*S.ImaginaryUnit*arg)**2
        return (tanh_half_sq - 1) / (tanh_half_sq + 1)


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return S.One
        else:
            return self.func(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_imaginary:
            return True


    def _eval_subs(self, old, new):
    # RECHECK
        pass
        # if self == old:
        #     return new
        # arg = self.args[0]
        # argnew = arg.subs(old, new)
        # if arg != argnew and (argnew/(S.Pi/2)).is_odd:
        #     return S.NaN
        # return tan(argnew)


    def _sage_(self):
        import sage.all as sage
        return sage.sec(self.args[0]._sage_())


class csc(TrigonometricFunction):
    """
    Usage
    =====
        csc(x) -> Returns the cosecant of x (measured in radians)

    Notes
    =====
        csc(x) will evaluate automatically in the case x is a
        multiple of pi.

    Examples
    ========
        >>> from sympy import csc
        >>> from sympy.abc import x
        >>> csc(x**2).diff(x)
        -2*x*cot(x**2)*csc(x**2)
        >>> csc(1).diff(x)
        0

    See also
    ========
       L{sin}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

       External links
       --------------

         U{Definitions in trigonometry<http://planetmath.org/encyclopedia/DefinitionsInTrigonometry.html>}
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return -S.ImaginaryUnit*C.csch(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        #if pi_coeff is not None:
            # if pi_coeff.is_integer:
        #         return S.Zero

        #     if not pi_coeff.is_Rational:
        #         narg = pi_coeff*S.Pi
        #         if narg != arg:
        #             return cls(narg)
        #         return None

        #     cst_table = {
        #         2 : S.ComplexInfinity,
        #         3 : sqrt(3),
        #         4 : S.One,
        #         6 : 1 / sqrt(3),
        #     }

        #     try:
        #         result = cst_table[pi_coeff.q]

        #         if (2*pi_coeff.p // pi_coeff.q) % 4 in (1, 3):
        #             return -result
        #         else:
        #             return result
        #     except KeyError:
        #         if pi_coeff.p > pi_coeff.q:
        #             p, q = pi_coeff.p % pi_coeff.q, pi_coeff.q
        #             if 2 * p > q:
        #                 return -cls(Rational(q - p, q)*S.Pi)
        #             return cls(Rational(p, q)*S.Pi)

        # if arg.is_Add:
        #     x, m = _peeloff_pi(arg)
        #     if m:
        #         if (m*2/S.Pi) % 2 == 0:
        #             return tan(x)
        #         else:
        #             return -cot(x)

        if arg.func is acsc:
            return arg.args[0]

        if arg.func is asin:
            x = arg.args[0]
            return 1 / x

        if arg.func is acos:
            x = arg.args[0]
            return 1 / sqrt(1 - x**2)

        if arg.func is acot:
            x = arg.args[0]
            return sqrt(1 + x**2)

        # TODO
        # Other inverses


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2 + 1
            return (-1)**(k-1) * 2 * (2**(2*k-1)-1) * C.bernoulli(2*k) * x**(2*k-1) / C.factorial(2*k)


    def fdiff(self, argindex=1):
        if argindex==1:
            return -cot(self.args[0])*self
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return acsc


    def _eval_nseries(self, x, n, logx):
    # RECHECK
        pass
        # i = self.args[0].limit(x, 0)*2/S.Pi
        # if i and i.is_Integer:
        #     return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        # return Function._eval_nseries(self, x, n=n, logx=logx)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = cos(2*re) - cosh(2*im)
        return (-2*sin(re)*cosh(im)/denom, 2*cos(re)*sinh(im)/denom)


    def _eval_expand_trig(self, deep=True, **hints):
    # RECHECK
        return self


    def _eval_rewrite_as_exp(self, arg):
        exp, I = C.exp, S.ImaginaryUnit
        return (2*I) / (exp(arg*I) - exp(-arg*I))

    def _eval_rewrite_as_sin(self, arg):
        return 1/sin(arg)

    def _eval_rewrite_as_cos(self, arg):
        return -1 / cos(arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        tan_half = tan(S.Half*arg)
        return (1+tan_half**2) / (2*tan_half)

    def _eval_rewrite_as_cot(self, arg):
        cot_half = cot(S.Half*arg)
        return (1+cot_half**2) / (2*cot_half)

    def _eval_rewrite_as_sec(self, arg):
        return -sec(arg + S.Pi/2)

    def _eval_rewrite_as_sinh(self, arg):
        return S.ImaginaryUnit / sinh(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_csch(self, arg):
        return S.ImaginaryUnit * csch(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cosh(self, arg):
        return -1 / cosh(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_sech(self, arg):
        return -sech(S.ImaginaryUnit*arg + S.ImaginaryUnit*S.Pi/2)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half = tanh(S.ImaginaryUnit*arg/2)
        return (S.ImaginaryUnit - S.ImaginaryUnit*tanh_half**2) / (2*tanh_half)

    def _eval_rewrite_as_coth(self, arg):
        coth_half = coth(S.ImaginaryUnit*arg/2)
        return  (S.ImaginaryUnit + S.ImaginaryUnit*coth_half**2) / (2*coth_half)


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_imaginary:
            return True


    def _eval_subs(self, old, new):
    # RECHECK
        pass
        # if self == old:
        #     return new
        # arg = self.args[0]
        # argnew = arg.subs(old, new)
        # if arg != argnew and (argnew/(S.Pi/2)).is_odd:
        #     return S.NaN
        # return tan(argnew)


    def _sage_(self):
        import sage.all as sage
        return sage.csc(self.args[0]._sage_())


###############################################################################
########################### TRIGONOMETRIC INVERSES ############################
###############################################################################

class InverseTrigonometricFunction(Function):
    """Base class for trigonometric functions. """
    pass


class asin(InverseTrigonometricFunction):
    """
    Usage
    =====
      asin(x) -> Returns the arc sine of x (measured in radians)

    Notes
    ====
        asin(x) will evaluate automatically in the cases
        oo, -oo, 0, 1, -1

    Examples
    ========
        >>> from sympy import asin, oo, pi
        >>> asin(1)
        pi/2
        >>> asin(-1)
        -pi/2

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.NegativeInfinity * S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.Infinity * S.ImaginaryUnit
            elif arg is S.Zero:
                return S.Zero
            elif arg is S.One:
                return S.Pi / 2
            elif arg is S.NegativeOne:
                return -S.Pi / 2

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            cst_table = {
                sqrt(3)/2  : 3,
                -sqrt(3)/2 : -3,
                sqrt(2)/2  : 4,
                -sqrt(2)/2 : -4,
                1/sqrt(2)  : 4,
                -1/sqrt(2) : -4,
                sqrt((5-sqrt(5))/8) : 5,
                -sqrt((5-sqrt(5))/8) : -5,
                S.Half     : 6,
                -S.Half    : -6,
                sqrt(2-sqrt(2))/2 : 8,
                -sqrt(2-sqrt(2))/2 : -8,
                (sqrt(5)-1)/4 : 10,
                (1-sqrt(5))/4 : -10,
                (sqrt(3)-1)/sqrt(2**3) : 12,
                (1-sqrt(3))/sqrt(2**3) : -12,
                (sqrt(5)+1)/4 : S(10)/3,
                -(sqrt(5)+1)/4 : -S(10)/3
                }

            if arg in cst_table:
                return S.Pi / cst_table[arg]

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return S.ImaginaryUnit * C.asinh(i_coeff)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n-2)**2/(n*(n-1)) * x**2
            else:
                k = (n - 1) // 2
                R = C.RisingFactorial(S.Half, k)
                F = C.factorial(k)
                return R / F * x**n / n


    def _eval_as_leading_term(self, x):
    # RECHECK
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)


    # RECHECK ALL
    def _eval_rewrite_as_log(self, arg):
        return -S.ImaginaryUnit*C.log(S.ImaginaryUnit*arg + sqrt(1-arg**2))

    def _eval_rewrite_as_acos(self, arg):
        return S.Pi/2 - acos(arg)

    def _eval_rewrite_as_atan(self, arg):
        return 2*atan(arg/(1 + sqrt(1 - arg**2)))

    def _eval_rewrite_as_acot(self, arg):
        pass

    def _eval_rewrite_as_asec(self, arg):
        pass

    def _eval_rewrite_as_acsc(self, arg):
        pass


    def _eval_is_real(self):
    # RECHECK
        return self.args[0].is_real and (self.args[0]>=-1 and self.args[0]<=1)


    def _sage_(self):
        import sage.all as sage
        return sage.arcsin(self.args[0]._sage_())


class acos(InverseTrigonometricFunction):
    """
    Usage
    =====
      acos(x) -> Returns the arc cosine of x (measured in radians)

    Notes
    =====
        acos(x) will evaluate automatically in the cases
        oo, -oo, 0, 1, -1

    Examples
    ========
        >>> from sympy import acos, oo, pi
        >>> acos(1)
        0
        >>> acos(0)
        pi/2
        >>> acos(oo)
        oo*I
    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity * S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity * S.ImaginaryUnit
            elif arg is S.Zero:
                return S.Pi / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return S.Pi

        if arg.is_number:
            cst_table = {
                S.Half     : S.Pi/3,
                -S.Half    : 2*S.Pi/3,
                sqrt(2)/2  : S.Pi/4,
                -sqrt(2)/2 : 3*S.Pi/4,
                1/sqrt(2)  : S.Pi/4,
                -1/sqrt(2) : 3*S.Pi/4,
                sqrt(3)/2  : S.Pi/6,
                -sqrt(3)/2 : 5*S.Pi/6,
                }

            if arg in cst_table:
                return cst_table[arg]


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n == 0:
            return S.Pi / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n-2)**2/(n*(n-1)) * x**2
            else:
                k = (n - 1) // 2
                R = C.RisingFactorial(S.Half, k)
                F = C.factorial(k)
                return -R / F * x**n / n


    def _eval_as_leading_term(self, x):
    # RECHECK
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
    # RECHECK
        return self.args[0].is_real and (self.args[0]>=-1 and self.args[0]<=1)


    # RECHECK ALL
    def _eval_rewrite_as_log(self, arg):
        return S.Pi/2 + S.ImaginaryUnit * C.log(S.ImaginaryUnit * arg + sqrt(1 - arg**2))

    def _eval_rewrite_as_asin(self, arg):
        return S.Pi/2 - asin(arg)

    def _eval_rewrite_as_atan(self, arg):
        if arg > -1 and arg <= 1:
            return 2 * atan(sqrt(1 - arg**2)/(1 + arg))
        else:
            raise ValueError("The argument must be bounded in the interval (-1,1]")

    def _eval_rewrite_as_acot(self, arg):
        pass

    def _eval_rewrite_as_asec(self, arg):
        pass

    def _eval_rewrite_as_acsc(self, arg):
        pass


    def _sage_(self):
        import sage.all as sage
        return sage.arccos(self.args[0]._sage_())


class atan(InverseTrigonometricFunction):
    """
    Usage
    =====
      atan(x) -> Returns the arc tangent of x (measured in radians)

    Notes
    =====
        atan(x) will evaluate automatically in the cases
        oo, -oo, 0, 1, -1

    Examples
    ========
        >>> from sympy import atan, oo, pi
        >>> atan(0)
        0
        >>> atan(1)
        pi/4
        >>> atan(oo)
        pi/2

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1+self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Pi / 2
            elif arg is S.NegativeInfinity:
                return -S.Pi / 2
            elif arg is S.Zero:
                return S.Zero
            elif arg is S.One:
                return S.Pi / 4
            elif arg is S.NegativeOne:
                return -S.Pi / 4
        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            cst_table = {
                sqrt(3)/3  : 6,
                -sqrt(3)/3 : -6,
                1/sqrt(3)  : 6,
                -1/sqrt(3) : -6,
                sqrt(3)    : 3,
                -sqrt(3)   : -3,
                (1+sqrt(2)) : S(8)/3,
                -(1+sqrt(2)) : S(8)/3,
                (sqrt(2)-1) : 8,
                (1-sqrt(2)) : -8,
                sqrt((5+2*sqrt(5))) : S(5)/2,
                -sqrt((5+2*sqrt(5))) : -S(5)/2,
                (2-sqrt(3)) : 12,
                -(2-sqrt(3)) : -12
                }

            if arg in cst_table:
                return S.Pi / cst_table[arg]

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return S.ImaginaryUnit * C.atanh(i_coeff)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return (-1)**((n-1)//2) * x**n / n


    def _eval_as_leading_term(self, x):
    # RECHECK
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
    # RECHECK
        return self.args[0].is_real


    # RECHECK ALL
    def _eval_rewrite_as_log(self, arg):
        return S.ImaginaryUnit/2 * \
               (C.log((S(1) - S.ImaginaryUnit * arg)/(S(1) + S.ImaginaryUnit * arg)))

    def _eval_rewrite_as_asin(self, arg):
        pass

    def _eval_rewrite_as_acos(self, arg):
        pass

    def _eval_rewrite_as_acot(self, arg):
        pass

    def _eval_rewrite_as_asec(self, arg):
        pass

    def _eval_rewrite_as_acsc(self, arg):
        pass


    def _eval_aseries(self, n, args0, x, logx):
    # RECHECK
        if args0[0] == S.Infinity:
            return S.Pi/2 - atan(1/self.args[0])
        elif args0[0] == S.NegativeInfinity:
            return -S.Pi/2 - atan(1/self.args[0])
        else:
            return super(atan, self)._eval_aseries(n, args0, x, logx)


    def _sage_(self):
        import sage.all as sage
        return sage.arctan(self.args[0]._sage_())


class acot(InverseTrigonometricFunction):
    """
    Usage
    =====
      acot(x) -> Returns the arc cotangent of x (measured in radians)

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / (1+self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg is S.Zero:
                return S.Pi/ 2
            elif arg is S.One:
                return S.Pi / 4
            elif arg is S.NegativeOne:
                return -S.Pi / 4

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            cst_table = {
                sqrt(3)/3  : 3,
                -sqrt(3)/3 : -3,
                1/sqrt(3)  : 3,
                -1/sqrt(3) : -3,
                sqrt(3)    : 6,
                -sqrt(3)   : -6,
                (1+sqrt(2)) : 8,
                -(1+sqrt(2)) : -8,
                (1-sqrt(2)) : -S(8)/3,
                (sqrt(2)-1) : S(8)/3,
                sqrt(5+2*sqrt(5)) : 10,
                -sqrt(5+2*sqrt(5)) : -10,
                (2+sqrt(3)) : 12,
                -(2+sqrt(3)) : -12,
                (2-sqrt(3)) : S(12)/5,
                -(2-sqrt(3)) : -S(12)/5,
                }

            if arg in cst_table:
                return S.Pi / cst_table[arg]

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)
        if i_coeff is not None:
            return -S.ImaginaryUnit * C.acoth(i_coeff)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n == 0:
            return S.Pi / 2 # FIX THIS
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return (-1)**((n+1)//2) * x**n / n


    def _eval_as_leading_term(self, x):
    # RECHECK
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)

    # RECHECK ALL
    def _eval_rewrite_as_log(self, arg):
        return S.ImaginaryUnit/2 * \
               (C.log((arg - S.ImaginaryUnit)/(arg + S.ImaginaryUnit)))

    def _eval_rewrite_as_asin(self, arg):
        pass

    def _eval_rewrite_as_acos(self, arg):
        return S.Pi/2 - acos(arg)

    def _eval_rewrite_as_atan(self, arg):
        return 2*atan(arg/(1 + sqrt(1 - arg**2)))

    def _eval_rewrite_as_asec(self, arg):
        pass

    def _eval_rewrite_as_acsc(self, arg):
        pass


    def _eval_is_real(self):
    # RECHECK
        return self.args[0].is_real


    def _eval_aseries(self, n, args0, x, logx):
    # RECHECK
        if args0[0] == S.Infinity:
            return S.Pi/2 - acot(1/self.args[0])
        elif args0[0] == S.NegativeInfinity:
            return 3*S.Pi/2 - acot(1/self.args[0])
        else:
            return super(atan, self)._eval_aseries(n, args0, x, logx)


    def _sage_(self):
        import sage.all as sage
        return sage.arccot(self.args[0]._sage_())


class asec(InverseTrigonometricFunction):
    """
    Usage
    =====
      asec(x) -> Returns the arc secant of x (measured in radians)

    Notes
    =====
        asec(x) will evaluate automatically in the cases
        oo, -oo, 0, 1, -1

    Examples
    ========
        >>> from sympy import asec, oo, pi
        >>> asec(1)
        0
        >>> asec(-1)
        pi

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / (self.args[0]**2 * sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            # elif arg is S.Infinity:
            #     return S.Infinity * S.ImaginaryUnit
            # elif arg is S.NegativeInfinity:
            #     return S.NegativeInfinity * S.ImaginaryUnit
            # elif arg is S.Zero:
            #     return S.Pi / 2
            # elif arg is S.One:
            #     return S.Zero
            # elif arg is S.NegativeOne:
            #     return S.Pi

        # if arg.is_number:
        #     cst_table = {
        #         S.Half     : S.Pi/3,
        #         -S.Half    : 2*S.Pi/3,
        #         sqrt(2)/2  : S.Pi/4,
        #         -sqrt(2)/2 : 3*S.Pi/4,
        #         1/sqrt(2)  : S.Pi/4,
        #         -1/sqrt(2) : 3*S.Pi/4,
        #         sqrt(3)/2  : S.Pi/6,
        #         -sqrt(3)/2 : 5*S.Pi/6,
        #         }

        #     if arg in cst_table:
        #         return cst_table[arg]


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n == 0:
            return S.Pi / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        # else:
        #     x = sympify(x)
        #     if len(previous_terms) >= 2 and n > 2:
        #         p = previous_terms[-2]
        #         return p * (n-2)**2/(n*(n-1)) * x**2
        #     else:
        #         k = (n - 1) // 2
        #         R = C.RisingFactorial(S.Half, k)
        #         F = C.factorial(k)
        #         return -R / F * x**n / n


    def _eval_as_leading_term(self, x):
    # RECHECK
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
    # RECHECK
        return self.args[0].is_real and (self.args[0]<=-1 and self.args[0]>=1)

    # RECHECK ALL
    def _eval_rewrite_as_log(self, x):
        return S.Pi/2 + S.ImaginaryUnit * C.log(S.ImaginaryUnit / x + sqrt(1 - 1/x**2))

    def _eval_rewrite_as_asin(self, arg):
        pass

    def _eval_rewrite_as_acos(self, arg):
        pass

    def _eval_rewrite_as_atan(self, arg):
        pass

    def _eval_rewrite_as_acot(self, arg):
        pass

    def _eval_rewrite_as_acsc(self, arg):
        pass


    def _sage_(self):
        import sage.all as sage
        return sage.arcsec(self.args[0]._sage_())


class acsc(InverseTrigonometricFunction):
    """
    Usage
    =====
      acsc(x) -> Returns the arc cosecant of x (measured in radians)

    Notes
    =====
        acsc(x) will evaluate automatically in the cases
        oo, -oo, 0, 1, -1

    Examples
    ========
        >>> from sympy import acsc, oo, pi
        >>> acsc(1)
        pi/2
        >>> acsc(-1)
        -pi/2

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / (self.args[0]**2 * sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
    # RECHECK
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            # elif arg is S.Infinity:
            #     return S.Infinity * S.ImaginaryUnit
            # elif arg is S.NegativeInfinity:
            #     return S.NegativeInfinity * S.ImaginaryUnit
            # elif arg is S.Zero:
            #     return S.Pi / 2
            # elif arg is S.One:
            #     return S.Zero
            # elif arg is S.NegativeOne:
            #     return S.Pi

        # if arg.is_number:
        #     cst_table = {
        #         S.Half     : S.Pi/3,
        #         -S.Half    : 2*S.Pi/3,
        #         sqrt(2)/2  : S.Pi/4,
        #         -sqrt(2)/2 : 3*S.Pi/4,
        #         1/sqrt(2)  : S.Pi/4,
        #         -1/sqrt(2) : 3*S.Pi/4,
        #         sqrt(3)/2  : S.Pi/6,
        #         -sqrt(3)/2 : 5*S.Pi/6,
        #         }

        #     if arg in cst_table:
        #         return cst_table[arg]


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n == 0:
            return S.Pi / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        # else:
        #     x = sympify(x)
        #     if len(previous_terms) >= 2 and n > 2:
        #         p = previous_terms[-2]
        #         return p * (n-2)**2/(n*(n-1)) * x**2
        #     else:
        #         k = (n - 1) // 2
        #         R = C.RisingFactorial(S.Half, k)
        #         F = C.factorial(k)
        #         return -R / F * x**n / n


    def _eval_as_leading_term(self, x):
    # RECHECK
        arg = self.args[0].as_leading_term(x)

        if C.Order(1,x).contains(arg):
            return arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
    # RECHECK
        return self.args[0].is_real and (self.args[0]<=-1 and self.args[0]>=1)

    # RECHECK ALL
    def _eval_rewrite_as_log(self, arg):
        return -S.ImaginaryUnit * C.log(S.ImaginaryUnit / arg + sqrt(1 - 1/arg**2))

    def _eval_rewrite_as_asin(self, arg):
        pass

    def _eval_rewrite_as_acos(self, arg):
        pass

    def _eval_rewrite_as_atan(self, arg):
        pass

    def _eval_rewrite_as_acot(self, arg):
        pass

    def _eval_rewrite_as_asec(self, arg):
        pass


    def _sage_(self):
        import sage.all as sage
        return sage.arccsc(self.args[0]._sage_())


class atan2(Function):
    """
    atan2(y,x) -> Returns the atan(y/x) taking two arguments y and x.
    Signs of both y and x are considered to determine the appropriate
    quadrant of atan(y/x). The range is (-pi, pi].


    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}

    """

    nargs = 2

    @classmethod
    def eval(cls, y, x):
        sign_y = C.sign(y)

        if y.is_zero:
            if x.is_positive:
                return S.Zero
            elif x.is_zero:
                return S.NaN
            elif x.is_negative:
                return S.Pi
        elif x.is_zero:
            if sign_y.is_Number:
                return sign_y * S.Pi/2
        elif x**2 == y**2:
            return (S.Pi*(2*sqrt(x**2)-x)) / (4*y)
        else:
            abs_yx = C.Abs(y/x)
            if sign_y.is_Number and abs_yx.is_number:
                phi = C.atan(abs_yx)
                if x.is_positive:
                    return sign_y * phi
                else:
                    return sign_y * (S.Pi - phi)


    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real


    def _eval_rewrite_as_log(self, y, x):
        return -S.ImaginaryUnit*C.log((x + S.ImaginaryUnit*y)/sqrt(x**2+y**2))


    def fdiff(self, argindex):
        y, x = self.args
        if argindex == 1:
            # Diff wrt to y
            return x/(x**2 + y**2)
        elif argindex == 2:
            # Diff wrt to x
            return -y/(x**2 + y**2)
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.atan2(self.args[0]._sage_(), self.args[1]._sage_())
