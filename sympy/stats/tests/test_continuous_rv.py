from sympy.stats import (P, E, Where, Density, Var, Covar, Skewness, Given,
                         pspace, CDF, ContinuousRV, Sample, Arcsin, Benini,
                         Beta, BetaPrime, Cauchy, Chi, Dagum, Exponential,
                         Gamma, Laplace, Logistic, LogNormal, Maxwell, Nakagami,
                         Normal, Pareto, Rayleigh, StudentT, Triangular,
                         Uniform, UniformSum, Weibull, WignerSemicircle)
from sympy import (Symbol, Dummy, Abs, exp, S, N, pi, simplify, Interval, erf,
                   Eq, log, lowergamma, Sum, symbols, sqrt, And, gamma, beta,
                   Piecewise, Integral, sin, Lambda, factorial, binomial, floor)
from sympy.utilities.pytest import raises, XFAIL

oo = S.Infinity

_x = Dummy("x")
_z = Dummy("z")

def test_single_normal():
    mu = Symbol('mu', real=True, bounded=True)
    sigma = Symbol('sigma', real=True, positive=True, bounded=True)
    X = Normal(0,1)
    Y = X*sigma + mu

    assert simplify(E(Y)) == mu
    assert simplify(Var(Y)) == sigma**2
    pdf = Density(Y)
    x = Symbol('x')
    assert pdf(x) == 2**S.Half*exp(-(x - mu)**2/(2*sigma**2))/(2*pi**S.Half*sigma)

    assert P(X**2 < 1) == erf(2**S.Half/2)

    assert E(X, Eq(X, mu)) == mu

@XFAIL
def test_conditional_1d():
    X = Normal(0,1)
    Y = Given(X, X>=0)

    assert Density(Y) == 2 * Density(X)

    assert Y.pspace.domain.set == Interval(0, oo)
    assert E(Y) == sqrt(2) / sqrt(pi)

    assert E(X**2) == E(Y**2)

def test_ContinuousDomain():
    X = Normal(0,1)
    assert Where(X**2<=1).set == Interval(-1,1)
    assert Where(X**2<=1).symbol == X.symbol
    Where(And(X**2<=1, X>=0)).set == Interval(0,1)
    raises(ValueError, "Where(sin(X)>1)")

    Y = Given(X, X>=0)

    assert Y.pspace.domain.set == Interval(0, oo)

def test_multiple_normal():
    X, Y = Normal(0,1), Normal(0,1)

    assert E(X+Y) == 0
    assert Var(X+Y) == 2
    assert Var(X+X) == 4
    assert Covar(X, Y) == 0
    assert Covar(2*X + Y, -X) == -2*Var(X)

    assert E(X, Eq(X+Y, 0)) == 0
    assert Var(X, Eq(X+Y, 0)) == S.Half

def test_symbolic():
    mu1, mu2 = symbols('mu1 mu2', real=True, bounded=True)
    s1, s2 = symbols('sigma1 sigma2', real=True, bounded=True, positive=True)
    rate = Symbol('lambda', real=True, positive=True, bounded=True)
    X = Normal(mu1, s1)
    Y = Normal(mu2, s2)
    Z = Exponential(rate)
    a, b, c = symbols('a b c', real=True, bounded=True)

    assert E(X) == mu1
    assert E(X+Y) == mu1+mu2
    assert E(a*X+b) == a*E(X)+b
    assert Var(X) == s1**2
    assert simplify(Var(X+a*Y+b)) == Var(X) + a**2*Var(Y)

    assert E(Z) == 1/rate
    assert E(a*Z+b) == a*E(Z)+b
    assert E(X+a*Z+b) == mu1 + a/rate + b

def test_CDF():
    X = Normal(0,1)

    d = CDF(X)
    assert P(X<1) == d(1)
    assert d(0) == S.Half

    d = CDF(X, X>0) # given X>0
    assert d(0) == 0

    Y = Exponential(10)
    d = CDF(Y)
    assert d(-5) == 0
    assert P(Y > 3) == 1 - d(3)

    raises(ValueError, "CDF(X+Y)")

    Z = Exponential(1)
    cdf = CDF(Z)
    z = Symbol('z')
    assert cdf(z) == Piecewise((0, z < 0), (1 - exp(-z), True))

def test_sample():
    z = Symbol('z')
    Z = ContinuousRV(z, exp(-z), set=Interval(0,oo))
    assert Sample(Z) in Z.pspace.domain.set
    sym, val = Z.pspace.sample().items()[0]
    assert sym == Z and val in Interval(0, oo)

def test_ContinuousRV():
    x = Symbol('x')
    pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)) # Normal distribution
    # X and Y should be equivalent
    X = ContinuousRV(x, pdf)
    Y = Normal(0, 1)

    assert Var(X) == Var(Y)
    assert P(X>0) == P(Y>0)


def test_arcsin():
    a = Symbol("a", real=True)
    b = Symbol("b", real=True)
    x = Symbol("x")

    X = Arcsin(a, b, symbol=x)
    assert Density(X) == Lambda(_x, 1/(pi*sqrt((-_x + b)*(_x - a))))


def test_benini():
    alpha = Symbol("alpha", positive=True)
    b = Symbol("beta", positive=True)
    sigma = Symbol("sigma", positive=True)
    x = Symbol("x")

    X = Benini(alpha, b, sigma, symbol=x)
    assert Density(X) == (Lambda(_x, (alpha/_x + 2*b*log(_x/sigma)/_x)
                          *exp(-alpha*log(_x/sigma) - b*log(_x/sigma)**2)))


def test_beta():
    a, b = symbols('alpha beta', positive=True)

    B = Beta(a, b)

    assert pspace(B).domain.set == Interval(0, 1)

    dens = Density(B)
    x = Symbol('x')
    assert dens(x) == x**(a-1)*(1-x)**(b-1) / beta(a,b)

    # This is too slow
    # assert E(B) == a / (a + b)
    # assert Var(B) == (a*b) / ((a+b)**2 * (a+b+1))

    # Full symbolic solution is too much, test with numeric version
    a, b = 1, 2
    B = Beta(a, b)
    assert E(B) == a / S(a + b)
    assert Var(B) == (a*b) / S((a+b)**2 * (a+b+1))


def test_betaprime():
    alpha = Symbol("alpha", positive=True)
    beta = Symbol("beta", positive=True)
    x = Symbol("x")

    X = BetaPrime(alpha, beta, symbol=x)
    assert Density(X) == (Lambda(_x, _x**(alpha - 1)*(_x + 1)**(-alpha - beta)
                          *gamma(alpha + beta)/(gamma(alpha)*gamma(beta))))


def test_cauchy():
    x0 = Symbol("x0")
    gamma = Symbol("gamma", positive=True)
    x = Symbol("x")

    X = Cauchy(x0, gamma, symbol=x)
    assert Density(X) == Lambda(_x, 1/(pi*gamma*(1 + (_x - x0)**2/gamma**2)))


def test_chi():
    k = Symbol("k", integer=True)
    x = Symbol("x")

    X = Chi(k, symbol=x)
    assert Density(X) == (Lambda(_x, 2**(-k/2 + 1)*_x**(k - 1)
                          *exp(-_x**2/2)/gamma(k/2)))


def test_dagum():
    p = Symbol("p", positive=True)
    b = Symbol("b", positive=True)
    a = Symbol("a", positive=True)
    x = Symbol("x")

    X = Dagum(p, a, b, symbol=x)
    assert Density(X) == Lambda(_x,
                                a*p*(_x/b)**(a*p)*((_x/b)**a + 1)**(-p - 1)/_x)


def test_exponential():
    rate = Symbol('lambda', positive=True, real=True, bounded=True)
    X = Exponential(rate)

    assert E(X) == 1/rate
    assert Var(X) == 1/rate**2
    assert Skewness(X) == 2
    assert P(X>0) == S(1)
    assert P(X>1) == exp(-rate)
    assert P(X>10) == exp(-10*rate)

    assert Where(X<=1).set == Interval(0,1)


def test_gamma():
    k = Symbol("k", positive=True)
    theta = Symbol("theta", positive=True)
    x = Symbol("x")

    X = Gamma(k, theta, symbol=x)
    assert Density(X) == Lambda(_x,
                                _x**(k - 1)*theta**(-k)*exp(-_x/theta)/gamma(k))
    assert CDF(X, meijerg=True) == Lambda(_z, Piecewise((0, _z < 0),
    (-k*lowergamma(k, 0)/gamma(k + 1) + k*lowergamma(k, _z/theta)/gamma(k + 1), True)))
    assert Var(X) == (-theta**2*gamma(k + 1)**2/gamma(k)**2 +
           theta*theta**(-k)*theta**(k + 1)*gamma(k + 2)/gamma(k))

    k, theta = symbols('k theta', real=True, bounded=True, positive=True)
    X = Gamma(k, theta)

    assert simplify(E(X)) == k*theta
    # can't get things to simplify on this one so we use subs
    assert Var(X).subs(k,5) == (k*theta**2).subs(k, 5)
    # The following is too slow
    # assert simplify(Skewness(X)).subs(k, 5) == (2/sqrt(k)).subs(k, 5)


def test_laplace():
    mu = Symbol("mu")
    b = Symbol("b", positive=True)
    x = Symbol("x")

    X = Laplace(mu, b, symbol=x)
    assert Density(X) == Lambda(_x, exp(-Abs(_x - mu)/b)/(2*b))


def test_logistic():
    mu = Symbol("mu", real=True)
    s = Symbol("s", positive=True)
    x = Symbol("x")

    X = Logistic(mu, s, symbol=x)
    assert Density(X) == Lambda(_x,
                                exp((-_x + mu)/s)/(s*(exp((-_x + mu)/s) + 1)**2))


def test_lognormal():
    mean = Symbol('mu', real=True, bounded=True)
    std = Symbol('sigma', positive=True, real=True, bounded=True)
    X = LogNormal(mean, std)
    # The sympy integrator can't do this too well
    #assert E(X) == exp(mean+std**2/2)
    #assert Var(X) == (exp(std**2)-1) * exp(2*mean + std**2)

    # Right now, only density function and sampling works
    # Test sampling: Only e^mean in sample std of 0
    for i in range(3):
        X = LogNormal(i, 0)
        assert S(Sample(X)) == N(exp(i))
    # The sympy integrator can't do this too well
    #assert E(X) ==

    mu = Symbol("mu", real=True)
    sigma = Symbol("sigma", positive=True)
    x = Symbol("x")

    X = LogNormal(mu, sigma, symbol=x)
    assert Density(X) == (Lambda(_x, sqrt(2)*exp(-(-mu + log(_x))**2
                                    /(2*sigma**2))/(2*_x*sqrt(pi)*sigma)))

    X = LogNormal(0, 1, symbol=Symbol('x')) # Mean 0, standard deviation 1
    assert Density(X) == Lambda(_x, sqrt(2)*exp(-log(_x)**2/2)/(2*_x*sqrt(pi)))


def test_maxwell():
    a = Symbol("a", positive=True)
    x = Symbol("x")

    X = Maxwell(a, symbol=x)

    assert Density(X) == Lambda(_x, sqrt(2)*_x**2*exp(-_x**2/(2*a**2))/(sqrt(pi)*a**3))
    assert E(X) == 2*sqrt(2)*a/sqrt(pi)
    assert simplify(Var(X)) == a**2*(-8 + 3*pi)/pi


def test_nakagami():
    mu = Symbol("mu", positive=True)
    omega = Symbol("omega", positive=True)
    x = Symbol("x")

    X = Nakagami(mu, omega, symbol=x)
    assert Density(X) == (Lambda(_x, 2*_x**(2*mu - 1)*mu**mu*omega**(-mu)
                                *exp(-_x**2*mu/omega)/gamma(mu)))
    assert simplify(E(X, meijerg=True)) == (sqrt(mu)*sqrt(omega)
           *gamma(mu + S.Half)/gamma(mu + 1))
    assert simplify(Var(X, meijerg=True)) == (omega*(gamma(mu)*gamma(mu + 1)
                          - gamma(mu + S.Half)**2)/(gamma(mu)*gamma(mu + 1)))


def test_pareto():
    xm, beta = symbols('xm beta', positive=True, bounded=True)
    alpha = beta + 5
    X = Pareto(xm, alpha)

    density = Density(X)
    x = Symbol('x')
    assert density(x) == x**(-(alpha+1))*xm**(alpha)*(alpha)

    # These fail because SymPy can not deduce that 1/xm != 0
    # assert simplify(E(X)) == alpha*xm/(alpha-1)
    # assert simplify(Var(X)) == xm**2*alpha / ((alpha-1)**2*(alpha-2))


def test_pareto_numeric():
    xm, beta = 3, 2
    alpha = beta + 5
    X = Pareto(xm, alpha)

    assert E(X) == alpha*xm/S(alpha-1)
    assert Var(X) == xm**2*alpha / S(((alpha-1)**2*(alpha-2)))


def test_rayleigh():
    sigma = Symbol("sigma", positive=True)
    x = Symbol("x")

    X = Rayleigh(sigma, symbol=x)
    assert Density(X) == Lambda(_x, _x*exp(-_x**2/(2*sigma**2))/sigma**2)
    assert E(X) == sqrt(2)*sqrt(pi)*sigma/2
    assert Var(X) == -pi*sigma**2/2 + 2*sigma**2


def test_studentt():
    nu = Symbol("nu", positive=True)
    x = Symbol("x")

    X = StudentT(nu, symbol=x)
    assert Density(X) == (Lambda(_x, (_x**2/nu + 1)**(-nu/2 - S.Half)
                          *gamma(nu/2 + S.Half)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))))


@XFAIL
def test_triangular():
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")
    x = Symbol("x")

    X = Triangular(a,b,c, symbol=x)
    assert Density(X) == Lambda(_x,
             Piecewise(((2*_x - 2*a)/((-a + b)*(-a + c)), And(a <= _x, _x < c)),
                       (2/(-a + b), _x == c),
                       ((-2*_x + 2*b)/((-a + b)*(b - c)), And(_x <= b, c < _x)),
                       (0, True)))


def test_uniform():
    l = Symbol('l', real=True, bounded=True)
    w = Symbol('w', positive=True, bounded=True)
    X = Uniform(l, l+w)

    assert simplify(E(X)) == l + w/2
    assert simplify(Var(X)) == w**2/12

    assert P(X<l) == 0 and P(X>l+w) == 0

    # With numbers all is well
    X = Uniform(3, 5)
    assert P(X<3) == 0 and P(X>5) == 0
    assert P(X<4) == P(X>4) == S.Half


@XFAIL
def test_uniformsum():
    n = Symbol("n", integer=True)
    x = Symbol("x")
    _k = Symbol("k")

    X = UniformSum(n, symbol=x)
    assert Density(X) == (Lambda(_x, Sum((-1)**_k*(-_k + _x)**(n - 1)
                         *binomial(n, _k), (_k, 0, floor(_x)))/factorial(n - 1)))



def test_weibull():
    a, b = symbols('a b', positive=True)
    X = Weibull(a, b)

    assert simplify(E(X)) == simplify(a * gamma(1 + 1/b))
    assert simplify(Var(X)) == simplify(a**2 * gamma(1 + 2/b) - E(X)**2)
    # Skewness tests too slow. Try shortcutting function?


def test_weibull_numeric():
    # Test for integers and rationals
    a = 1
    bvals = [S.Half, 1, S(3)/2, 5]
    for b in bvals:
        X = Weibull(a, b)
        assert simplify(E(X)) == simplify(a * gamma(1 + 1/S(b)))
        assert simplify(Var(X)) == simplify(a**2 * gamma(1 + 2/S(b)) - E(X)**2)
        # Not testing Skew... it's slow with int/frac values > 3/2


def test_wignersemicircle():
    R = Symbol("R", positive=True)
    x = Symbol("x")

    X = WignerSemicircle(R, symbol=x)
    assert Density(X) == Lambda(_x, 2*sqrt(-_x**2 + R**2)/(pi*R**2))
    assert E(X) == 0


def test_prefab_sampling():
    N = Normal(0, 1)
    L = LogNormal(0, 1)
    E = Exponential(1)
    P = Pareto(1, 3)
    W = Weibull(1, 1)
    U = Uniform(0, 1)
    B = Beta(2,5)
    G = Gamma(1,3)

    variables = [N,L,E,P,W,U,B,G]
    niter = 10
    for var in variables:
        for i in xrange(niter):
            assert Sample(var) in var.pspace.domain.set

def test_input_value_assertions():
    a, b = symbols('a b')
    p, q = symbols('p q', positive=True)

    raises(ValueError, "Normal(3, 0)")
    raises(ValueError, "Normal(a, b)")
    Normal(a, p) # No error raised
    raises(ValueError, "Exponential(a)")
    Exponential(p) # No error raised
    for fn_name in ['Pareto', 'Weibull', 'Beta', 'Gamma']:
        raises(ValueError, "%s(a, p)" % fn_name)
        raises(ValueError, "%s(p, a)" % fn_name)
        eval("%s(p, q)" % fn_name) # No error raised

@XFAIL
def test_unevaluated():
    x = Symbol('x')
    X = Normal(0,1, symbol=x)
    assert E(X, evaluate=False) == (
            Integral(sqrt(2)*x*exp(-x**2/2)/(2*sqrt(pi)), (x, -oo, oo)))

    assert E(X+1, evaluate=False) == (
            Integral(sqrt(2)*x*exp(-x**2/2)/(2*sqrt(pi)), (x, -oo, oo)) + 1)

    assert P(X>0, evaluate=False) == (
            Integral(sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)), (x, 0, oo)))

    assert P(X>0, X**2<1, evaluate=False) == (
            Integral(sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)*
            Integral(sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)),
                (x, -1, 1))), (x, 0, 1)))
