from sympy import *

x = Symbol("x")

points = [oo, -oo, I*oo, -I*oo]
funcs = [sin,cos,tan,cot,sec,csc, sinh, cosh, tanh, sech, csch]

for f in funcs:
    s = "Asymptotic expansion of " + f.__name__ + "(x)"
    print("="*len(s))
    print(s)

    for p in points:
        t = "  at point " + str(p)
        print("-"*len(t))
        print(t)

        try:
            print(pretty(f(x)._eval_aseries(10, [p], x, None)))
        except PoleError:
            print("Not implemented")
