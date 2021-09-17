import unittest
import numpy as np
from simsopt._core.graph_optimizable import Optimizable
from simsopt._core.derivative import Derivative


class Opt(Optimizable):
    """
    This class defines a minimal object that can be optimized.
    """

    def __init__(self, n=3, x0=None):
        self.n = n
        x = x0 if x0 is not None else np.ones((n, ))
        super().__init__(x)

    def foo(self):
        return np.sin(self.local_full_x) * np.sum(self.local_full_x)

    def dfoo_vjp(self, v):
        n = self.n
        full_jac = np.zeros((n, n))
        for i in range(n):
            full_jac[i, i] = np.cos(self.local_full_x[i]) * np.sum(self.local_full_x)
            full_jac[:, i] += np.sin(self.local_full_x)
        return Derivative({self: v.T @ full_jac})


class InterSum(Optimizable):

    def __init__(self, opt_a, opt_b):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[opt_a, opt_b])
        self.opt_a = opt_a
        self.opt_b = opt_b

    def bar(self):
        return np.sum(self.opt_a.foo()**2) + np.sum(self.opt_b.foo()**2)

    def dbar_vjp(self, v):
        return v[0] * self.opt_a.dfoo_vjp(2*self.opt_a.foo()) \
            + v[0] * self.opt_b.dfoo_vjp(2*self.opt_b.foo())


class InterProd(Optimizable):

    def __init__(self, opt_a, opt_b):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[opt_a, opt_b])
        self.opt_a = opt_a
        self.opt_b = opt_b

    def bar(self):
        return np.prod(self.opt_a.foo()) * np.prod(self.opt_b.foo())

    def dbar_vjp(self, v):
        n_a = self.opt_a.n
        v_a = np.zeros((n_a, ))
        for i in range(n_a):
            v_a[i] = np.prod(self.opt_a.foo())/self.opt_a.foo()[i]
        v_a *= np.prod(self.opt_b.foo())

        n_b = self.opt_b.n
        v_b = np.zeros((n_b, ))
        for i in range(n_b):
            v_b[i] = np.prod(self.opt_b.foo())/self.opt_b.foo()[i]
        v_b *= np.prod(self.opt_a.foo())

        return v[0] * self.opt_a.dfoo_vjp(v_a) \
            + v[0] * self.opt_b.dfoo_vjp(v_b) \



class Obj(Optimizable):

    def __init__(self, inter_a, inter_b):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[inter_a, inter_b])
        self.inter_a = inter_a
        self.inter_b = inter_b

    def J(self):
        return self.inter_a.bar() * self.inter_b.bar()

    def dJ(self):
        return self.inter_a.dbar_vjp([self.inter_b.bar()]) + self.inter_b.dbar_vjp([self.inter_a.bar()])


class DerivativeTests(unittest.TestCase):

    def test_taylor_graph(self):
        np.random.seed(1)
        # built a reasonably complex graph of two inputs, that both feed into
        # two intermediary results and then are combined into a final result.
        # i.e. f(g(x, y), h(x, y))

        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        intersum = InterSum(opt1, opt2)
        interprod = InterProd(opt1, opt2)
        obj = Obj(intersum, interprod)
        x = obj.x + 0.05 * np.random.standard_normal(size=obj.x.shape)
        obj.x = x
        x = obj.x
        h = np.random.standard_normal(size=x.shape)
        f = obj.J()
        df = obj.dJ()
        dfh = np.sum(df(obj)*h)
        err_old = 1e9
        for i in range(5, 11):
            eps = 0.5**i
            obj.x = x + 3 * eps * h
            fppp = obj.J()
            obj.x = x + 2 * eps * h
            fpp = obj.J()
            obj.x = x + eps * h
            fp = obj.J()
            obj.x = x - eps * h
            fm = obj.J()
            obj.x = x - 2 * eps * h
            fmm = obj.J()
            obj.x = x - 3 * eps * h
            fmmm = obj.J()
            # print(np.abs((fp-fm)/(2*eps) - dfh))
            dfhest = ((1/12) * fmm - (2/3) * fm + (2/3) * fp - (1/12) * fpp)/eps
            err = np.abs(dfhest - dfh)
            assert err < (0.6)**4 * err_old
            print(err_old/err)
            err_old = err

            # dfhest = ((-1/60)*fmmm + (3/20)*fmm -(3/4)*fm+(3/4)*fp-(3/20)*fpp + (1/60)*fppp)/eps
            # err = np.abs(dfhest - dfh)
            # print(err_old/err)
            # err_old = err

    def test_add_mul(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))

        dj1p2 = dj1 + dj2 + 2*dj1
        assert np.allclose(dj1p2(opt1), 3*dj1(opt1))
        assert np.allclose(dj1p2(opt2), dj2(opt2))

    def test_iadd_imul(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj1_ = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))

        dj1 += dj2
        assert np.allclose(dj1(opt2), dj2(opt2))
        dj1 += dj1_
        assert np.allclose(dj1(opt1), 2*dj1_(opt1))
        dj1 *= 1.5
        assert np.allclose(dj1(opt2), 1.5*dj2(opt2))
        assert np.allclose(dj1(opt1), 3*dj1_(opt1))

    def test_zero_when_not_found(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        assert np.allclose(dj1(opt2), np.zeros((2, )))
