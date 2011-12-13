from sympy.core import S, C, sympify, cacheit
from sympy.core.function import Function, ArgumentIndexError

from sympy.functions.elementary.miscellaneous import sqrt

###############################################################################
########################### HYPERBOLIC FUNCTIONS ##############################
###############################################################################

class HyperbolicFunction(Function):
    """Base class for hyperbolic functions."""

    nargs = 1

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit


class sinh(HyperbolicFunction):
    """
    Usage
    =====
      sinh(x) -> Returns the hyperbolic sine of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg is S.Zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                return S.ImaginaryUnit * C.sin(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return -cls(-arg)

            if arg.func == asinh:
                return arg.args[0]

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x-1) * sqrt(x+1)

            if arg.func == atanh:
                x = arg.args[0]
                return x/sqrt(1-x**2)

            if arg.func == acoth:
                x = arg.args[0]
                return 1/(sqrt(x-1) * sqrt(x+1))


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / C.factorial(n)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return cosh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return asinh


    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] == S.Infinity:
            return C.exp(x) / 2
        elif args0[0] == S.NegativeInfinity:
            return -C.exp(-x) / 2
        elif C.re(args0[0]) == 0:
            # No asymptotic series expansion along the imaginary line
            return sinh(x)
        else:
            return super(sinh, self)._eval_aseries(n, args0, x, logx)


    def _eval_rewrite_as_exp(self, arg):
        return (C.exp(arg) - C.exp(-arg)) / 2

    def _eval_rewrite_as_csch(self, arg):
        return 1 / csch(arg)

    def _eval_rewrite_as_cosh(self, arg):
        return -S.ImaginaryUnit * cosh(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_sech(self, arg):
        return -S.ImaginaryUnit / sech(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half = tanh(S.Half*arg)
        return 2*tanh_half / (1 - tanh_half**2)

    def _eval_rewrite_as_coth(self, arg):
        coth_half = coth(S.Half*arg)
        return 2*coth_half / (coth_half**2 - 1)

    def _eval_rewrite_as_sin(self, arg):
        return -S.ImaginaryUnit * C.sin(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_csc(self, arg):
        return -S.ImaginaryUnit / C.csc(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cos(self, arg):
        return S.ImaginaryUnit * C.cos(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_sec(self, arg):
        return S.ImaginaryUnit / C.sec(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        tan_half = C.tan(S.ImaginaryUnit*S.Half*arg)
        return -2*S.ImaginaryUnit*tan_half / (tan_half**2 + 1)

    def _eval_rewrite_as_cot(self, arg):
        cot_half = C.cot(S.ImaginaryUnit*S.Half*arg)
        return -2*S.ImaginaryUnit*cot_half / (cot_half**2 + 1)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
    # RECHECK
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
        return (sinh(re)*C.cos(im), cosh(re)*C.sin(im))


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        return False


    def _sage_(self):
        import sage.all as sage
        return sage.sinh(self.args[0]._sage_())


class cosh(HyperbolicFunction):
    """
    Usage
    =====
      cosh(x) -> Returns the hyperbolic cosine of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg is S.Zero:
                return S.One
            elif arg.is_negative:
                return cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                return C.cos(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return cls(-arg)

            if arg.func == asinh:
                return sqrt(1+arg.args[0]**2)

            if arg.func == acosh:
                return arg.args[0]

            if arg.func == atanh:
                return 1/sqrt(1-arg.args[0]**2)

            if arg.func == acoth:
                x = arg.args[0]
                return x/(sqrt(x-1) * sqrt(x+1))


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / C.factorial(n)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return sinh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return acosh


    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] == S.Infinity:
            return C.exp(x) / 2
        elif args0[0] == S.NegativeInfinity:
            return C.exp(-x) / 2
        elif C.re(args0[0]) == 0:
            # No asymptotic series expansion along the imaginary line
            return cosh(x)
        else:
            return super(cosh, self)._eval_aseries(n, args0, x, logx)


    def _eval_rewrite_as_exp(self, arg):
        return (C.exp(arg) + C.exp(-arg)) / 2

    def _eval_rewrite_as_sinh(self, arg):
        return -S.ImaginaryUnit*sinh(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_csch(self, arg):
        return -S.ImaginaryUnit / csch(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_sech(self, arg):
        return 1 / sech(arg)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half = tanh(S.Half*arg)**2
        return (1 + tanh_half) / (1 - tanh_half)

    def _eval_rewrite_as_coth(self, arg):
        coth_half = coth(S.Half*arg)**2
        return (coth_half + 1) / (coth_half - 1)

    def _eval_rewrite_as_sin(self, arg):
        return C.sin(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_csc(self, arg):
        return 1 / C.csc(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_cos(self, arg):
        return C.cos(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_sec(self, arg):
        return 1 / C.sec(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_tan(self, arg):
        tan_half_sq = C.tan(S.ImaginaryUnit*S.Half*arg)**2
        return (1 - tan_half_sq) / (1 + tan_half_sq)

    def _eval_rewrite_as_cot(self, arg):
        cot_half_sq = C.cot(S.ImaginaryUnit*S.Half*arg)**2
        return (tan_half_sq - 1) / (tan_half_sq + 1)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
    # RECHECK
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

        return (cosh(re)*C.cos(im), sinh(re)*C.sin(im))


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return S.One
        else:
            return self.func(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        return False


    def _sage_(self):
        import sage.all as sage
        return sage.cosh(self.args[0]._sage_())


class tanh(HyperbolicFunction):
    """
    Usage
    =====
      tanh(x) -> Returns the hyperbolic tangent of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg is S.Zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                if i_coeff.as_coeff_mul()[0].is_negative:
                    return -S.ImaginaryUnit * C.tan(-i_coeff)
                return S.ImaginaryUnit * C.tan(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return -cls(-arg)

            if arg.func == asinh:
                x = arg.args[0]
                return x/sqrt(1+x**2)

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x-1) * sqrt(x+1) / x

            if arg.func == atanh:
                return arg.args[0]

            if arg.func == acoth:
                return 1/arg.args[0]


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2 + 1
            return 2**(2*k) * (2**(2*k)-1) * C.bernoulli(2*k) / C.factorial(2*k) * x**(2*k-1)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return sech(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return atanh


    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] == S.Infinity:
            return S.One - 2*C.exp(-2*x)*C.hyper([1],[],-C.exp(-2*x))
        elif args0[0] == S.NegativeInfinity:
            return -S.One + 2*C.exp(2*x)*C.hyper([1],[],-C.exp(2*x))
        elif C.re(args0[0]) == 0:
            # No asymptotic series expansion along the imaginary line
            return tanh(x)
        else:
            return super(tanh, self)._eval_aseries(n, args0, x, logx)


    def _eval_rewrite_as_exp(self, arg):
        neg_exp, pos_exp = C.exp(-arg), C.exp(arg)
        return (pos_exp - neg_exp) / (pos_exp + neg_exp)

    def _eval_rewrite_as_sinh(self, arg):
        return S.ImaginaryUnit*sinh(arg) / sinh(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_csch(self, arg):
        return S.ImaginaryUnit*csch(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_cosh(self, arg):
        return -S.ImaginaryUnit*cosh(arg + S.Pi*S.ImaginaryUnit/2) / cosh(arg)

    def _eval_rewrite_as_sech(self, arg):
        return -S.ImaginaryUnit*sech(arg) / sech(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_coth(self, arg):
        return 1 / coth(arg)

    def _eval_rewrite_as_sin(self, arg):
        return -S.ImaginaryUnit*C.sin(S.ImaginaryUnit*arg) / C.sin(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_csc(self, arg):
        return -S.ImaginaryUnit*C.csc(S.ImaginaryUnit*arg + S.Pi/2) / C.csc(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cos(self, arg):
        return S.ImaginaryUnit*C.cos(S.ImaginaryUnit*arg + S.Pi/2) / C.cos(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_sec(self, arg):
        return S.ImaginaryUnit*C.sec(S.ImaginaryUnit*arg) / C.sec(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        return -S.ImaginaryUnit*C.tan(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cot(self, arg):
        return S.ImaginaryUnit*C.cot(S.ImaginaryUnit*arg + S.Pi/2)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
    # RECHECK
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
        denom = sinh(re)**2 + C.cos(im)**2
        return (sinh(re)*cosh(re)/denom, C.sin(im)*C.cos(im)/denom)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        arg = self.args[0]
        if arg.is_real:
            return True


    def _sage_(self):
        import sage.all as sage
        return sage.tanh(self.args[0]._sage_())


class coth(HyperbolicFunction):
    """
    Usage
    =====
      coth(x) -> Returns the hyperbolic cotangent of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg is S.Zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                if i_coeff.as_coeff_mul()[0].is_negative:
                    return S.ImaginaryUnit * C.cot(-i_coeff)
                return -S.ImaginaryUnit * C.cot(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return -cls(-arg)

            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1+x**2)/x

            if arg.func == acosh:
                x = arg.args[0]
                return x/(sqrt(x-1) * sqrt(x+1))

            if arg.func == atanh:
                return 1/arg.args[0]

            if arg.func == acoth:
                return arg.args[0]


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
            return 2**(2*k) * C.bernoulli(2*k) / C.factorial(2*k) * x**(2*k-1)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -csch(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return acoth


    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] == S.Infinity:
            return S.One + 2*C.exp(-2*x)*C.hyper([1],[],C.exp(-2*x))
        elif args0[0] == S.NegativeInfinity:
            return -S.One - 2*C.exp(2*x)*C.hyper([1],[],C.exp(2*x))
        elif C.re(args0[0]) == 0:
            # No asymptotic series expansion along the imaginary line
            return coth(x)
        else:
            return super(coth, self)._eval_aseries(n, args0, x, logx)


    def _eval_rewrite_as_exp(self, arg):
        neg_exp, pos_exp = C.exp(-arg), C.exp(arg)
        return (pos_exp + neg_exp) / (pos_exp - neg_exp)

    def _eval_rewrite_as_sinh(self, arg):
        return -S.ImaginaryUnit*sinh(arg + S.Pi*S.ImaginaryUnit/2) / sinh(arg)

    def _eval_rewrite_as_csch(self, arg):
        return -S.ImaginaryUnit*csch(arg) / csch(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_cosh(self, arg):
        return S.ImaginaryUnit*cosh(arg) / cosh(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_sech(self, arg):
        return S.ImaginaryUnit*sech(arg + S.Pi*S.ImaginaryUnit/2) / sech(arg)

    def _eval_rewrite_as_tanh(self, arg):
        return 1 / tanh(arg)

    def _eval_rewrite_as_sin(self, arg):
        return S.ImaginaryUnit*C.sin(S.ImaginaryUnit*arg + S.Pi/2) / C.sin(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_csc(self, arg):
        return S.ImaginaryUnit*C.csc(S.ImaginaryUnit*arg) / C.csc(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_cos(self, arg):
        return -S.ImaginaryUnit*C.cos(S.ImaginaryUnit*arg) / C.cos(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_sec(self, arg):
        return -S.ImaginaryUnit*C.sec(S.ImaginaryUnit*arg + S.Pi/2) / C.sec(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_tan(self, arg):
        return S.ImaginaryUnit / C.tan(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cot(self, arg):
        return S.ImaginaryUnit*C.cot(S.ImaginaryUnit*arg)


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def as_real_imag(self, deep=True, **hints):
    # RECHECK
    # See issue 2899
        if self.args[0].is_real:
            if deep:
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + C.sin(im)**2
        return (sinh(re)*cosh(re)/denom, -C.sin(im)*C.cos(im)/denom)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        # Bounded for x in C with Re x != 0
        return False


    def _sage_(self):
        import sage.all as sage
        return sage.coth(self.args[0]._sage_())


class sech(HyperbolicFunction):
    """
    Usage
    =====
      sech(x) -> Returns the hyperbolic secant of x (measured in radians)

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            k = n // 2
            return C.euler(2*k) / C.factorial(2*k) * x**(2*k)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -self*tanh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return asech


    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] == S.Infinity:
            return 2*C.exp(-x)*C.hyper([1],[],-C.exp(-2*x))
        elif args0[0] == S.NegativeInfinity:
            return 2*C.exp(x)*C.hyper([1],[],-C.exp(2*x))
        elif C.re(args0[0]) == 0:
            # No asymptotic series expansion along the imaginary line
            return sech(x)
        else:
            return super(sech, self)._eval_aseries(n, args0, x, logx)


    def _eval_rewrite_as_exp(self, arg):
        return 2 / (C.exp(arg) + C.exp(-arg))

    def _eval_rewrite_as_sinh(self, arg):
        return 1 / C.sin(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_csch(self, arg):
        return S.ImaginaryUnit*csch(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_cosh(self, arg):
        return 1 / cosh(arg)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half_sq = tanh(S.Half*arg)**2
        return (1 - tanh_half_sq) / (1 + tanh_half_sq)

    def _eval_rewrite_as_coth(self, arg):
        coth_half_sq = coth(S.Half*arg)**2
        return (coth_haf_sq - 1) / ( coth_half_sq + 1)

    def _eval_rewrite_as_sin(self, arg):
        return 1 / C.sin(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_csc(self, arg):
        return C.csc(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_cos(self, arg):
        return 1 / C.cos(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_sec(self, arg):
        return C.sec(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_tan(self, arg):
        tan_half_sq = C.tan(S.ImaginaryUnit*S.Half*arg)**2
        return (1 + tan_half_sq) / (1 - tanh_half_sq)

    def _eval_rewrite_as_cot(self, arg):
        cot_half_sq = C.cot(S.ImaginaryUnit*S.Half*arg)**2
        return (cot_half_sq + 1) / (cot_half_sq - 1)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


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
        return sage.sech(self.args[0]._sage_())


class csch(HyperbolicFunction):
    """
    Usage
    =====
      csch(x) -> Returns the hyperbolic cosecant of x (measured in radians)

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

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
            return -2 * (2**(2*k-1)-1) * C.bernoulli(2*k) / C.factorial(2*k) * x**(2*k-1)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -self*coth(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    def inverse(self, argindex=1):
        return acsch


    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] == S.Infinity:
            return 2*C.exp(-x)*C.hyper([1],[],C.exp(-2*x))
        elif args0[0] == S.NegativeInfinity:
            return -2*C.exp(x)*C.hyper([1],[],C.exp(2*x))
        elif C.re(args0[0]) == 0:
            # No asymptotic series expansion along the imaginary line
            return csch(x)
        else:
            return super(csch, self)._eval_aseries(n, args0, x, logx)


    def _eval_rewrite_as_exp(self, arg):
        return 2 / (C.exp(arg) - C.exp(-arg))

    def _eval_rewrite_as_sinh(self, arg):
        return 1 / sinh(arg)

    def _eval_rewrite_as_cosh(self, arg):
        return S.ImaginaryUnit / cosh(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_sech(self, arg):
        return S.ImaginaryUnit*sech(arg + S.Pi*S.ImaginaryUnit/2)

    def _eval_rewrite_as_tanh(self, arg):
        tanh_half = tanh(S.Half*arg)
        return (1 - tanh_half**2) / (2*tanh_half)

    def _eval_rewrite_as_coth(self, arg):
        coth_half = coth(S.Half*arg)
        return (coth_half**2 - 1) / (2*coth_half)

    def _eval_rewrite_as_sin(self, arg):
        return S.ImaginaryUnit / C.sin(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_csc(self, arg):
        return S.ImaginaryUnit*C.csc(S.ImaginaryUnit*arg)

    def _eval_rewrite_as_cos(self, arg):
        return -S.ImaginaryUnit / C.cos(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_sec(self, arg):
        return -S.ImaginaryUnit*C.sec(S.ImaginaryUnit*arg + S.Pi/2)

    def _eval_rewrite_as_tan(self, arg):
        tan_half = C.tan(S.Half*S.ImaginaryUnit*arg)
        return (S.ImaginaryUnit*tan_half**2 + S.ImaginaryUnit) / (2*tan_half)

    def _eval_rewrite_as_cot(self, arg):
        cot_half = C.cot(S.Half*S.ImaginaryUnit*arg)
        return (S.ImaginaryUnit*cot_half**2 + S.ImaginaryUnit) / (2*cot_half)


    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())


    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)


    def _eval_is_real(self):
        return self.args[0].is_real


    def _eval_is_bounded(self):
        # Bounded for x in C with Re x != 0
        return False


    def _sage_(self):
        import sage.all as sage
        return sage.csch(self.args[0]._sage_())

###############################################################################
############################# HYPERBOLIC INVERSES #############################
###############################################################################

class InverseHyperbolicFunction(Function):
    """Base class for inverse hyperbolic functions. """

    nargs = 1


class asinh(InverseHyperbolicFunction):
    """
    Usage
    =====
      asinh(x) -> Returns the inverse hyperbolic sine of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{acsch}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg is S.Zero:
                return S.Zero
            elif arg is S.One:
                return C.log(sqrt(2) + 1)
            elif arg is S.NegativeOne:
                return C.log(sqrt(2) - 1)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.ComplexInfinity

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                return S.ImaginaryUnit * C.asin(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return -cls(-arg)


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
                return -p * (n-2)**2/(n*(n-1)) * x**2
            else:
                k = (n - 1) // 2
                R = C.RisingFactorial(S.Half, k)
                F = C.factorial(k)
                return (-1)**k * R / F * x**n / n


    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / sqrt(self.args[0]**2 + 1)
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.arcsinh(self.args[0]._sage_())


class acosh(InverseHyperbolicFunction):
    """
    Usage
    =====
      acosh(x) -> Returns the inverse hyperbolic cosine of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{asech}, L{atanh}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg is S.Zero:
                return S.Pi*S.ImaginaryUnit / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return S.Pi*S.ImaginaryUnit

        if arg.is_number:
            cst_table = {
                S.ImaginaryUnit : C.log(S.ImaginaryUnit*(1+sqrt(2))),
                -S.ImaginaryUnit : C.log(-S.ImaginaryUnit*(1+sqrt(2))),
                S.Half       : S.Pi/3,
                -S.Half      : 2*S.Pi/3,
                sqrt(2)/2    : S.Pi/4,
                -sqrt(2)/2   : 3*S.Pi/4,
                1/sqrt(2)    : S.Pi/4,
                -1/sqrt(2)   : 3*S.Pi/4,
                sqrt(3)/2    : S.Pi/6,
                -sqrt(3)/2   : 5*S.Pi/6,
                (sqrt(3)-1)/sqrt(2**3) : 5*S.Pi/12,
                -(sqrt(3)-1)/sqrt(2**3) : 7*S.Pi/12,
                sqrt(2+sqrt(2))/2 : S.Pi/8,
                -sqrt(2+sqrt(2))/2 : 7*S.Pi/8,
                sqrt(2-sqrt(2))/2 : 3*S.Pi/8,
                -sqrt(2-sqrt(2))/2 : 5*S.Pi/8,
                (1+sqrt(3))/(2*sqrt(2)) : S.Pi/12,
                -(1+sqrt(3))/(2*sqrt(2)) : 11*S.Pi/12,
                (sqrt(5)+1)/4 : S.Pi/5,
                -(sqrt(5)+1)/4 : 4*S.Pi/5
            }

            if arg in cst_table:
                if arg.is_real:
                    return cst_table[arg]*S.ImaginaryUnit
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            return S.Infinity

        i_coeff = arg.as_coefficient(S.ImaginaryUnit)

        if i_coeff is not None:
            if i_coeff.as_coeff_mul()[0].is_negative:
                return S.ImaginaryUnit * C.acos(i_coeff)
            return S.ImaginaryUnit * C.acos(-i_coeff)
        else:
            if arg.as_coeff_mul()[0].is_negative:
                return -cls(-arg)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n == 0:
            return S.Pi*S.ImaginaryUnit / 2
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
                return -R / F * S.ImaginaryUnit * x**n / n


    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / (sqrt(self.args[0] - 1)*sqrt(self.args[0] + 1))
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.arccosh(self.args[0]._sage_())


class atanh(InverseHyperbolicFunction):
    """
    Usage
    =====
      atanh(x) -> Returns the inverse hyperbolic tangent of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{acoth}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Zero:
                return S.Zero
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg is S.Infinity:
                return -S.ImaginaryUnit * C.atan(arg)
            elif arg is S.NegativeInfinity:
                return S.ImaginaryUnit * C.atan(-arg)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                return S.ImaginaryUnit * C.atan(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return -cls(-arg)


    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n


    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / (1-self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.arctanh(self.args[0]._sage_())


class acoth(InverseHyperbolicFunction):
    """
    Usage
    =====
      acoth(x) -> Returns the inverse hyperbolic cotangent of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{asech}, L{atanh}
    """

    @classmethod
    def eval(cls, arg):
    # RECHECK
        arg = sympify(arg)

        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg is S.Zero:
                return S.Pi*S.ImaginaryUnit / 2
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return 0

            i_coeff = arg.as_coefficient(S.ImaginaryUnit)

            if i_coeff is not None:
                return -S.ImaginaryUnit * C.acot(i_coeff)
            else:
                if arg.as_coeff_mul()[0].is_negative:
                    return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
    # RECHECK
        if n == 0:
            return S.Pi*S.ImaginaryUnit / 2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n


    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1 / (1-self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.arccoth(self.args[0]._sage_())


class asech(InverseHyperbolicFunction):
    """
    Usage
    =====
      asech(x) -> Returns the inverse hyperbolic secant of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acsch}, L{acosh}, L{atanh}, L{acoth}
    """

    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return C.log(arg)
        else:
            return self.func(arg)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / (sqrt(1-self.args[0]) * self.args[0]) * sqrt(1/(1+self.args[0]))
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.arcsech(self.args[0]._sage_())


class acsch(InverseHyperbolicFunction):
    """
    Usage
    =====
      acsch(x) -> Returns the inverse hyperbolic cosecant of x

    See also
    ========
       L{sin}, L{csc}, L{cos}, L{sec}, L{tan}, L{cot}
       L{asin}, L{acsc}, L{acos}, L{asec}, L{atan}, L{acot}
       L{sinh}, L{csch}, L{cosh}, L{sech}, L{tanh}, L{coth}
       L{asinh}, L{acosh}, L{asech}, L{atanh}, L{acoth}
    """

    def _eval_as_leading_term(self, x):
        arg = self.args[0].as_leading_term(x)
        if C.Order(1,x).contains(arg):
            return C.log(arg)
        else:
            return self.func(arg)


    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1 / (sqrt(1+1/self.args[0]**2) * self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)


    def _sage_(self):
        import sage.all as sage
        return sage.arccsch(self.args[0]._sage_())
