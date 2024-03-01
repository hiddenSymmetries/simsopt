import unittest
import numpy as np
from simsopt._core.optimizable import Optimizable, ScaledOptimizable, OptimizableSum
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives.utilities import Weight


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

    @derivative_dec
    def dJ(self):
        return self.inter_a.dbar_vjp([self.inter_b.bar()]) + self.inter_b.dbar_vjp([self.inter_a.bar()])


def taylor_test(obj):
    np.random.seed(1)
    x = obj.x + 0.05 * np.random.standard_normal(size=obj.x.shape)
    obj.x = x
    x = obj.x
    h = np.random.standard_normal(size=x.shape)
    f = obj.J()
    df = obj.dJ()
    dfh = np.sum(df * h)
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


class DerivativeTests(unittest.TestCase):

    def test_taylor_graph(self):
        # built a reasonably complex graph of two inputs, that both feed into
        # two intermediary results and then are combined into a final result.
        # i.e. f(g(x, y), h(x, y))

        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        intersum = InterSum(opt1, opt2)
        interprod = InterProd(opt1, opt2)
        obj = Obj(intersum, interprod)
        taylor_test(obj)

    def test_scaled_optimizable(self):
        """
        Confirm that values and derivatives behave correctly for the
        ScaledOptimizable class
        """
        opt1a = Opt(n=3)
        opt1b = Opt(n=2)
        intersum1 = InterSum(opt1a, opt1b)
        interprod1 = InterProd(opt1a, opt1b)
        obj1 = Obj(intersum1, interprod1)

        factor = 1.3
        obj2 = ScaledOptimizable(factor, obj1)
        taylor_test(obj2)
        np.testing.assert_allclose(obj2.J(), factor * obj1.J())
        np.testing.assert_allclose(obj2.dJ(), factor * obj1.dJ())

        obj3 = ScaledOptimizable(Weight(factor), obj1)
        taylor_test(obj3)
        np.testing.assert_allclose(obj3.J(), factor * obj1.J())
        np.testing.assert_allclose(obj3.dJ(), factor * obj1.dJ())

    def test_scaled_optimizable_operator(self):
        """
        Confirm that values and derivatives behave correctly when an
        Optimizable object is scaled by a constant, overloading the *
        operator.
        """
        opt1a = Opt(n=2)
        opt1b = Opt(n=5)
        intersum1 = InterSum(opt1a, opt1b)
        interprod1 = InterProd(opt1a, opt1b)
        obj1 = Obj(intersum1, interprod1)

        factor = -10
        obj2 = factor * obj1
        taylor_test(obj2)
        np.testing.assert_allclose(obj2.J(), factor * obj1.J())
        np.testing.assert_allclose(obj2.dJ(), factor * obj1.dJ())

        factor = -10
        w = Weight(factor)
        obj3 = w * obj1
        taylor_test(obj3)
        np.testing.assert_allclose(obj3.J(), factor * obj1.J())
        np.testing.assert_allclose(obj3.dJ(), factor * obj1.dJ())
        w *= 3
        taylor_test(obj3)
        np.testing.assert_allclose(obj3.J(), 3*factor * obj1.J())
        np.testing.assert_allclose(obj3.dJ(), 3*factor * obj1.dJ())

    def test_optimizable_sum(self):
        """
        Confirm that values and derivatives behave correctly for the
        OptimizableSum class
        """
        opt1a = Opt(n=3)
        opt1b = Opt(n=2)
        intersum1 = InterSum(opt1a, opt1b)
        interprod1 = InterProd(opt1a, opt1b)
        obj1 = Obj(intersum1, interprod1)

        opt2a = Opt(n=4)
        opt2b = Opt(n=5)
        intersum2 = InterSum(opt2a, opt2b)
        interprod2 = InterProd(opt2a, opt2b)
        obj2 = Obj(intersum2, interprod2)

        obj3 = OptimizableSum((obj1, obj2))
        taylor_test(obj3)
        np.testing.assert_allclose(obj3.J(), obj1.J() + obj2.J())

        for n in range(1, 4):
            obj4 = OptimizableSum([obj1] * n)
            np.testing.assert_allclose(obj4.J(), n * obj1.J())
            np.testing.assert_allclose(obj4.dJ(), n * obj1.dJ())

    def test_optimizable_sum_operator(self):
        """
        Confirm that values and derivatives behave correctly when
        Optimizable objects are summed using the overloaded +
        operator.
        """
        opt1a = Opt(n=3)
        opt1b = Opt(n=2)
        intersum1 = InterSum(opt1a, opt1b)
        interprod1 = InterProd(opt1a, opt1b)
        obj1 = Obj(intersum1, interprod1)

        opt2a = Opt(n=4)
        opt2b = Opt(n=5)
        intersum2 = InterSum(opt2a, opt2b)
        interprod2 = InterProd(opt2a, opt2b)
        obj2 = Obj(intersum2, interprod2)

        obj3 = obj1 + obj2
        taylor_test(obj3)
        np.testing.assert_allclose(obj3.J(), obj1.J() + obj2.J())

        obj4 = obj1 + obj2 + obj1
        taylor_test(obj4)
        np.testing.assert_allclose(obj4.J(), 2 * obj1.J() + obj2.J())

        # Try combining objects with sum():
        for n in range(1, 4):
            obj5 = sum([obj1] * n)
            np.testing.assert_allclose(obj5.J(), n * obj1.J())
            np.testing.assert_allclose(obj5.dJ(), n * obj1.dJ())

    def test_taylor_scaled_summed(self):
        """
        Perform a Taylor test for a case in which Optimizable objects are
        scaled and added together.
        """
        opt1a = Opt(n=3)
        opt1b = Opt(n=2)
        intersum1 = InterSum(opt1a, opt1b)
        interprod1 = InterProd(opt1a, opt1b)
        obj1 = Obj(intersum1, interprod1)

        opt2a = Opt(n=4)
        opt2b = Opt(n=5)
        intersum2 = InterSum(opt2a, opt2b)
        interprod2 = InterProd(opt2a, opt2b)
        obj2 = Obj(intersum2, interprod2)

        # Scale and sum the objects to get the total objective:
        factor = 1.3
        obj = obj1 + factor * obj2
        taylor_test(obj)

    def test_add_mul(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))

        dj1p2 = dj1 + dj2 + 2*dj1
        assert np.allclose(dj1p2(opt1), 3*dj1(opt1))
        assert np.allclose(dj1p2(opt2), dj2(opt2))

    def test_sub_mul(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))

        dj1m2 = dj1 - dj2 - 2*dj1
        assert np.allclose(dj1m2(opt1), -1*dj1(opt1))
        assert np.allclose(dj1m2(opt2), -dj2(opt2))

    def test_iadd(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj1_ = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))
        dj2_ = opt2.dfoo_vjp(np.ones(2))

        dj1 += dj1_
        dj1 += dj2
        assert np.allclose(dj1(opt1), 2*dj1_(opt1))
        assert np.allclose(dj1(opt2), dj2_(opt2))

    def test_isub(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj1_ = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))
        dj2_ = opt2.dfoo_vjp(np.ones(2))

        dj1 -= 2*dj1_
        dj1 -= dj2
        assert np.allclose(dj1(opt1), (-1)*dj1_(opt1))
        assert np.allclose(dj1(opt2), -dj2_(opt2))

    def test_imul(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        dj2 = opt2.dfoo_vjp(np.ones(2))

        dj1_ = opt1.dfoo_vjp(np.ones(3))
        dj2_ = opt2.dfoo_vjp(np.ones(2))

        dj1 *= 2.
        assert np.allclose(dj1(opt1), 2*dj1_(opt1))
        dj = dj1 + 4*dj2
        assert np.allclose(dj(opt1), 2*dj1_(opt1))
        assert np.allclose(dj(opt2), 4*dj2_(opt2))

    def test_zero_when_not_found(self):
        opt1 = Opt(n=3)
        opt2 = Opt(n=2)

        dj1 = opt1.dfoo_vjp(np.ones(3))
        assert np.allclose(dj1(opt2), np.zeros((2, )))
