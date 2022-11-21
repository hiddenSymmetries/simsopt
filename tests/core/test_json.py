"""
Credits: github.com/materialsvirtuallab/monty

The MIT License (MIT)

Copyright (c) 2014 Materials Virtual Lab
Copyright (c) 2022 Hidden Symmetries Team

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import unittest
from enum import Enum

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from bson.objectid import ObjectId
except ImportError:
    ObjectId = None

from simsopt._core.json import GSONDecoder, GSONEncoder, GSONable, _load_redirect, jsanitize, SIMSON

test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_files")


class GoodGSONClass(GSONable):
    def __init__(self, a, b, c, d=1, *values, **kwargs):
        self.a = a
        self.b = b
        self._c = c
        self._d = d
        self.values = values
        self.kwargs = kwargs

    def __eq__(self, other):
        return (
            self.a == other.a
            and self.b == other.b
            and self._c == other._c
            and self._d == other._d
            and self.kwargs == other.kwargs
            and self.values == other.values
        )


class GoodNestedGSONClass(GSONable):
    def __init__(self, a_list, b_dict, c_list_dict_list, **kwargs):
        assert isinstance(a_list, list)
        assert isinstance(b_dict, dict)
        assert isinstance(c_list_dict_list, list)
        assert isinstance(c_list_dict_list[0], dict)
        first_key = list(c_list_dict_list[0].keys())[0]
        assert isinstance(c_list_dict_list[0][first_key], list)
        self.a_list = a_list
        self.b_dict = b_dict
        self._c_list_dict_list = c_list_dict_list
        self.kwargs = kwargs


class MethodSerializationClass(GSONable):
    def __init__(self, a):
        self.a = a

    def method(self):
        pass

    @staticmethod
    def staticmethod(self):
        pass

    @classmethod
    def classmethod(cls):
        pass

    def __call__(self, b):
        # override call for instances
        return self.__class__(b)

    class NestedClass:
        def inner_method(self):
            pass


class MethodNonSerializationClass:
    def __init__(self, a):
        self.a = a

    def method(self):
        pass


def my_callable(a, b):
    return a + b


class EnumTest(GSONable, Enum):
    a = 1
    b = 2


class ClassContainingDataFrame(GSONable):
    def __init__(self, df):
        self.df = df


class ClassContainingSeries(GSONable):
    def __init__(self, s):
        self.s = s


class ClassContainingNumpyArray(GSONable):
    def __init__(self, np_a):
        self.np_a = np_a


@dataclasses.dataclass
class Point:
    x: float = 1
    y: float = 2


class Coordinates(GSONable):
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)


@dataclasses.dataclass
class NestedDataClass:
    points: list[Point]


class GSONableTest(unittest.TestCase):
    def setUp(self):
        self.good_cls = GoodGSONClass

        class BadGSONClass(GSONable):
            def __init__(self, a, b):
                self.a = a
                self.b = b

            def as_dict(self, serial_objs_dict):
                d = {"init": {"a": self.a, "b": self.b}}
                return d

        self.bad_cls = BadGSONClass

        class BadGSONClass2(GSONable):
            def __init__(self, a, b):
                self.a = a
                self.c = b

        self.bad_cls2 = BadGSONClass2

        class AutoGSON(GSONable):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        self.auto_gson = AutoGSON

    def test_to_from_dict(self):
        obj = self.good_cls("Hello", "World", "Python")
        d = obj.as_dict(serial_objs_dict={})
        self.assertIsNotNone(d)
        self.good_cls.from_dict(d, serial_objs_dict={}, recon_objs=[])
        jsonstr = obj.to_json()
        d = json.loads(jsonstr)
        # self.assertTrue(d["@class"], "GoodGSONClass")
        obj = self.bad_cls("Hello", "World")
        d = obj.as_dict(serial_objs_dict={})
        self.assertIsNotNone(d)
        self.assertRaises(TypeError, self.bad_cls.from_dict, d)
        obj = self.bad_cls2("Hello", "World")
        serial_objs_dict = {}
        self.assertRaises(NotImplementedError, obj.as_dict, serial_objs_dict)
        obj = self.auto_gson(2, 3)
        d = obj.as_dict(serial_objs_dict={})
        self.auto_gson.from_dict(d, serial_objs_dict={}, recon_objs={})

    def test_nested_to_from_dict(self):
        GGC = GoodGSONClass
        a_list = [GGC(1, 1.0, "one"), GGC(2, 2.0, "two")]
        b_dict = {"first": GGC(3, 3.0, "three"), "second": GGC(4, 4.0, "four")}
        c_list_dict_list = [
            {
                "list1": [
                    GGC(5, 5.0, "five"),
                    GGC(6, 6.0, "six"),
                    GGC(7, 7.0, "seven"),
                ],
                "list2": [GGC(8, 8.0, "eight")],
            },
            {
                "list3": [
                    GGC(9, 9.0, "nine"),
                    GGC(10, 10.0, "ten"),
                    GGC(11, 11.0, "eleven"),
                    GGC(12, 12.0, "twelve"),
                ],
                "list4": [GGC(13, 13.0, "thirteen"), GGC(14, 14.0, "fourteen")],
                "list5": [GGC(15, 15.0, "fifteen")],
            },
        ]
        obj = GoodNestedGSONClass(a_list=a_list, b_dict=b_dict, c_list_dict_list=c_list_dict_list)

        serial_objs_dict = {}
        obj_dict = obj.as_dict(serial_objs_dict=serial_objs_dict)
        obj2 = GoodNestedGSONClass.from_dict(obj_dict, serial_objs_dict=serial_objs_dict,
                                             recon_objs={})
        self.assertTrue([obj2.a_list[ii] == aa for ii, aa in enumerate(obj.a_list)])
        self.assertTrue([obj2.b_dict[kk] == val for kk, val in obj.b_dict.items()])
        self.assertEqual(len(obj.a_list), len(obj2.a_list))
        self.assertEqual(len(obj.b_dict), len(obj2.b_dict))

    def test_enum_serialization(self):
        e = EnumTest.a
        serial_objs_dict = {}
        d = e.as_dict(serial_objs_dict=serial_objs_dict)
        e_new = EnumTest.from_dict(d, serial_objs_dict=serial_objs_dict, recon_objs={})
        self.assertEqual(e_new.name, e.name)
        self.assertEqual(e_new.value, e.value)

        d = {"123": EnumTest.a}
        f = jsanitize(d)
        self.assertEqual(f["123"], "EnumTest.a")

        f = jsanitize(d, strict=True)
        self.assertEqual(f["123"]["@module"], "core.test_json")
        self.assertEqual(f["123"]["@class"], "EnumTest")
        self.assertEqual(f["123"]["value"], 1)

        f = jsanitize(d, strict=True, enum_values=True)
        self.assertEqual(f["123"], 1)

        f = jsanitize(d, enum_values=True)
        self.assertEqual(f["123"], 1)


class A(GSONable):
    def __init__(self, b, c):
        self.b = b
        self.c = c
        self.name = str(id(self))

    def __repr__(self):
        return f"class: A\nname: {self.name}"


class B(GSONable):
    def __init__(self, d):
        self.d = d
        self.name = str(id(self))

    def __repr__(self):
        return f"class: B\nname: {self.name}"


class C(GSONable):
    def __init__(self, d):
        self.d = d
        self.name = str(id(self))

    def __repr__(self):
        return f"class: C\nname: {self.name}"


class D(GSONable):
    def __init__(self, e=1.0):
        self.e = e
        self.name = str(id(self))

    def __repr__(self):
        return f"class: D\nname: {self.name}"


class SIMSONTest(unittest.TestCase):
    def test_nested_to_from_dict(self):
        GGC = GoodGSONClass
        a_list = [GGC(1, 1.0, "one"), GGC(2, 2.0, "two")]
        b_dict = {"first": GGC(3, 3.0, "three"), "second": GGC(4, 4.0, "four")}
        c_list_dict_list = [
            {
                "list1": [
                    GGC(5, 5.0, "five"),
                    GGC(6, 6.0, "six"),
                    GGC(7, 7.0, "seven"),
                ],
                "list2": [GGC(8, 8.0, "eight")],
            },
            {
                "list3": [
                    GGC(9, 9.0, "nine"),
                    GGC(10, 10.0, "ten"),
                    GGC(11, 11.0, "eleven"),
                    GGC(12, 12.0, "twelve"),
                ],
                "list4": [GGC(13, 13.0, "thirteen"), GGC(14, 14.0, "fourteen")],
                "list5": [GGC(15, 15.0, "fifteen")],
            },
        ]
        obj = GoodNestedGSONClass(a_list=a_list, b_dict=b_dict, c_list_dict_list=c_list_dict_list)

        s = json.dumps(SIMSON(obj), cls=GSONEncoder)
        obj4 = json.loads(s, cls=GSONDecoder)
        self.assertTrue([obj4.a_list[ii] == aa for ii, aa in enumerate(obj.a_list)])
        self.assertTrue([obj4.b_dict[kk] == val for kk, val in obj.b_dict.items()])
        self.assertEqual(len(obj.a_list), len(obj4.a_list))
        self.assertEqual(len(obj.b_dict), len(obj4.b_dict))

    def test_diamond_graph_json(self):

        d = D()
        b = B(d)
        c = C(d)
        a1 = A(b, c)
        a2 = A(b, c)
        sims_json = SIMSON([a1, a2])

        json_str = json.dumps(sims_json, cls=GSONEncoder)
        recon_sims = json.loads(json_str, cls=GSONDecoder)
        new_b1 = recon_sims[0].b
        new_b2 = recon_sims[1].b
        self.assertEqual(new_b1.name, new_b2.name)
        self.assertEqual(id(new_b1.d), id(recon_sims[0].c.d))


class JsonTest(unittest.TestCase):
    def test_as_from_dict(self):
        obj = GoodGSONClass(1, 2, 3, hello="world")
        s = json.dumps(SIMSON(obj), cls=GSONEncoder)
        obj2 = json.loads(s, cls=GSONDecoder)
        self.assertEqual(obj2.a, 1)
        self.assertEqual(obj2.b, 2)
        self.assertEqual(obj2._c, 3)
        self.assertEqual(obj2._d, 1)
        self.assertEqual(obj2.kwargs, {"hello": "world", "values": []})
        obj = GoodGSONClass(obj, 2, 3)
        s = json.dumps(SIMSON(obj), cls=GSONEncoder)
        obj2 = json.loads(s, cls=GSONDecoder)
        self.assertEqual(obj2.a.a, 1)
        self.assertEqual(obj2.b, 2)
        self.assertEqual(obj2._c, 3)
        self.assertEqual(obj2._d, 1)
        listobj = [obj, obj2]
        s = json.dumps(SIMSON(listobj), cls=GSONEncoder)
        listobj2 = json.loads(s, cls=GSONDecoder)
        self.assertEqual(listobj2[0].a.a, 1)

    def test_datetime(self):
        dt = datetime.datetime.now()
        jsonstr = json.dumps(dt, cls=GSONEncoder)
        d = json.loads(jsonstr, cls=GSONDecoder)
        self.assertEqual(type(d), datetime.datetime)
        self.assertEqual(dt, d)
        # Test a nested datetime.
        a = {"dt": dt, "a": 1}
        jsonstr = json.dumps(a, cls=GSONEncoder)
        d = json.loads(jsonstr, cls=GSONDecoder)
        self.assertEqual(type(d["dt"]), datetime.datetime)

    def test_uuid(self):
        from uuid import UUID, uuid4

        uuid = uuid4()
        jsonstr = json.dumps(uuid, cls=GSONEncoder)
        d = json.loads(jsonstr, cls=GSONDecoder)
        self.assertEqual(type(d), UUID)
        self.assertEqual(uuid, d)
        # Test a nested UUID.
        a = {"uuid": uuid, "a": 1}
        jsonstr = json.dumps(a, cls=GSONEncoder)
        d = json.loads(jsonstr, cls=GSONDecoder)
        self.assertEqual(type(d["uuid"]), UUID)

    def test_nan(self):
        x = [float("NaN")]
        djson = json.dumps(x, cls=GSONEncoder)
        d = json.loads(djson)
        self.assertEqual(type(d[0]), float)

    def test_numpy(self):
        x = np.array([1, 2, 3], dtype="int64")
        self.assertRaises(TypeError, json.dumps, x)
        djson = json.dumps(x, cls=GSONEncoder)
        d = json.loads(djson)
        self.assertEqual(d["@class"], "array")
        self.assertEqual(d["@module"], "numpy")
        self.assertEqual(d["data"], [1, 2, 3])
        self.assertEqual(d["dtype"], "int64")
        x = json.loads(djson, cls=GSONDecoder)
        self.assertEqual(type(x), np.ndarray)
        x = np.min([1, 2, 3]) > 2
        self.assertRaises(TypeError, json.dumps, x)

        x = np.array([1 + 1j, 2 + 1j, 3 + 1j], dtype="complex64")
        self.assertRaises(TypeError, json.dumps, x)
        djson = json.dumps(x, cls=GSONEncoder)
        d = json.loads(djson)
        self.assertEqual(d["@class"], "array")
        self.assertEqual(d["@module"], "numpy")
        self.assertEqual(d["data"], [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        self.assertEqual(d["dtype"], "complex64")
        x = json.loads(djson, cls=GSONDecoder)
        self.assertEqual(type(x), np.ndarray)
        self.assertEqual(x.dtype, "complex64")

        x = np.array([[1 + 1j, 2 + 1j], [3 + 1j, 4 + 1j]], dtype="complex64")
        self.assertRaises(TypeError, json.dumps, x)
        djson = json.dumps(x, cls=GSONEncoder)
        d = json.loads(djson)
        self.assertEqual(d["@class"], "array")
        self.assertEqual(d["@module"], "numpy")
        self.assertEqual(d["data"], [[[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]]])
        self.assertEqual(d["dtype"], "complex64")
        x = json.loads(djson, cls=GSONDecoder)
        self.assertEqual(type(x), np.ndarray)
        self.assertEqual(x.dtype, "complex64")

        x = {"energies": [np.float64(1234.5)]}
        d = jsanitize(x, strict=True)
        assert type(d["energies"][0]) == float

        # Test data nested in a class
        x = np.array([[1 + 1j, 2 + 1j], [3 + 1j, 4 + 1j]], dtype="complex64")
        cls = ClassContainingNumpyArray(np_a={"a": [{"b": x}]})

        json_str = json.dumps(SIMSON(cls), cls=GSONEncoder)

        obj = json.loads(json_str, cls=GSONDecoder)
        self.assertIsInstance(obj, ClassContainingNumpyArray)
        self.assertIsInstance(obj.np_a["a"][0]["b"], np.ndarray)
        self.assertEqual(obj.np_a["a"][0]["b"][0][1], 2 + 1j)

    @unittest.skipIf(pd is None, "Pandas not found")
    def test_pandas(self):

        cls = ClassContainingDataFrame(df=pd.DataFrame([{"a": 1, "b": 1}, {"a": 1, "b": 2}]))

        json_str = GSONEncoder().encode(SIMSON(cls))
        obj = json.loads(json_str, cls=GSONDecoder)
        self.assertIsInstance(obj, ClassContainingDataFrame)
        self.assertIsInstance(obj.df, pd.DataFrame)
        self.assertEqual(list(obj.df.a), [1, 1])

        cls = ClassContainingSeries(s=pd.Series({"a": [1, 2, 3], "b": [4, 5, 6]}))

        json_str = GSONEncoder().encode(SIMSON(cls))
        obj = json.loads(json_str, cls=GSONDecoder)
        self.assertIsInstance(obj, ClassContainingSeries)
        self.assertIsInstance(obj.s, pd.Series)
        self.assertEqual(list(obj.s.a), [1, 2, 3])

        cls = ClassContainingSeries(s={"df": [pd.Series({"a": [1, 2, 3], "b": [4, 5, 6]})]})

        json_str = GSONEncoder().encode(SIMSON(cls))
        obj = json.loads(json_str, cls=GSONDecoder)
        self.assertIsInstance(obj, ClassContainingSeries)
        self.assertIsInstance(obj.s["df"][0], pd.Series)
        self.assertEqual(list(obj.s["df"][0].a), [1, 2, 3])

    def test_callable(self):
        instance = MethodSerializationClass(a=1)
        for function in [
            # builtins
            str,
            list,
            sum,
            open,
            # functions
            os.path.join,
            my_callable,
            # unbound methods
            MethodSerializationClass.NestedClass.inner_method,
            MethodSerializationClass.staticmethod,
            instance.staticmethod,
            # methods bound to classes
            MethodSerializationClass.classmethod,
            instance.classmethod,
            # classes
            MethodSerializationClass,
            Enum,
        ]:
            self.assertRaises(TypeError, json.dumps, function)
            djson = json.dumps(SIMSON(function), cls=GSONEncoder)
            x = json.loads(djson, cls=GSONDecoder)
            self.assertEqual(x, function)

        # test method bound to instance
        for function in [instance.method]:
            self.assertRaises(TypeError, json.dumps, function)
            djson = json.dumps(SIMSON(function), cls=GSONEncoder)
            x = json.loads(djson, cls=GSONDecoder)

            # can't just check functions are equal as the instance the function is bound
            # to will be different. Instead, we check that the serialized instance
            # is the same, and that the function qualname is the same
            self.assertEqual(x.__qualname__, function.__qualname__)

        # test method bound to object that is not serializable
        for function in [MethodNonSerializationClass(1).method]:
            self.assertRaises(TypeError, json.dumps, function, cls=GSONEncoder)

        # test that callable GSONable objects still get serialized as the objects
        # rather than as a callable
        djson = json.dumps(SIMSON(instance), cls=GSONEncoder)
        self.assertTrue("@class" in djson)

    @unittest.skipIf(ObjectId is None, "bson not found")
    def test_objectid(self):
        oid = ObjectId("562e8301218dcbbc3d7d91ce")
        self.assertRaises(TypeError, json.dumps, oid)
        djson = json.dumps(oid, cls=GSONEncoder)
        x = json.loads(djson, cls=GSONDecoder)
        self.assertEqual(type(x), ObjectId)

    @unittest.skipIf(pd is None or ObjectId is None, "pandas/bson not found")
    def test_jsanitize(self):
        # clean_json should have no effect on None types.
        d = {"hello": 1, "world": None}
        clean = jsanitize(d)
        self.assertIsNone(clean["world"])
        self.assertEqual(json.loads(json.dumps(d)), json.loads(json.dumps(clean)))

        d = {"hello": GoodGSONClass(1, 2, 3)}
        self.assertRaises(TypeError, json.dumps, d)
        clean = jsanitize(d)
        self.assertIsInstance(clean["hello"], str)
        clean_strict = jsanitize(d, strict=True)
        self.assertEqual(clean_strict["hello"]["a"], 1)
        self.assertEqual(clean_strict["hello"]["b"], 2)
        clean_recursive_gsonable = jsanitize(d, recursive_gsonable=True)
        self.assertEqual(clean_recursive_gsonable["hello"]["a"], 1)
        self.assertEqual(clean_recursive_gsonable["hello"]["b"], 2)

        d = {"dt": datetime.datetime.now()}
        clean = jsanitize(d)
        self.assertIsInstance(clean["dt"], str)
        clean = jsanitize(d, allow_bson=True)
        self.assertIsInstance(clean["dt"], datetime.datetime)

        d = {
            "a": ["b", np.array([1, 2, 3])],
            "b": ObjectId.from_datetime(datetime.datetime.now()),
        }
        clean = jsanitize(d)
        self.assertEqual(clean["a"], ["b", [1, 2, 3]])
        self.assertIsInstance(clean["b"], str)

        rnd_bin = bytes(np.random.rand(10))
        d = {"a": bytes(rnd_bin)}
        clean = jsanitize(d, allow_bson=True)
        self.assertEqual(clean["a"], bytes(rnd_bin))
        self.assertIsInstance(clean["a"], bytes)

        p = pathlib.Path("/home/user/")
        clean = jsanitize(p, strict=True)
        self.assertIn(clean, ["/home/user", "\\home\\user"])

        # test jsanitizing callables (including classes)
        instance = MethodSerializationClass(a=1)
        for function in [
            # builtins
            str,
            list,
            sum,
            open,
            # functions
            os.path.join,
            my_callable,
            # unbound methods
            MethodSerializationClass.NestedClass.inner_method,
            MethodSerializationClass.staticmethod,
            instance.staticmethod,
            # methods bound to classes
            MethodSerializationClass.classmethod,
            instance.classmethod,
            # classes
            MethodSerializationClass,
            Enum,
        ]:
            d = {"f": function}
            clean = jsanitize(d)
            self.assertTrue("@module" in clean["f"])
            self.assertTrue("@callable" in clean["f"])

        # test method bound to instance
        for function in [instance.method]:
            d = {"f": function}
            clean = jsanitize(d)
            self.assertTrue("@module" in clean["f"])
            self.assertTrue("@callable" in clean["f"])
            self.assertTrue(clean["f"].get("@bound", None) is not None)
            self.assertTrue("@class" in clean["f"]["@bound"])

        # test method bound to object that is not serializable
        for function in [MethodNonSerializationClass(1).method]:
            d = {"f": function}
            clean = jsanitize(d)
            self.assertTrue(isinstance(clean["f"], str))

            # test that strict checking gives an error
            self.assertRaises(AttributeError, jsanitize, d, strict=True)

        # test that callable GSONable objects still get serialized as the objects
        # rather than as a callable
        d = {"c": instance}
        clean = jsanitize(d, strict=True)
        self.assertTrue("@class" in clean["c"])

        # test on pandas
        df = pd.DataFrame([{"a": 1, "b": 1}, {"a": 1, "b": 2}])
        clean = jsanitize(df)
        self.assertEqual(clean, df.to_dict())

        s = pd.Series({"a": [1, 2, 3], "b": [4, 5, 6]})
        clean = jsanitize(s)
        self.assertEqual(clean, s.to_dict())

    def test_redirect(self):
        GSONable.REDIRECT["core.test_json"] = {"test_class": {"@class": "GoodGSONClass", "@module": "core.test_json"}}

        d = {
            "@class": "test_class",
            "@module": "core.test_json",
            "a": 1,
            "b": 1,
            "c": 1,
        }

        obj = json.loads(json.dumps(d), cls=GSONDecoder)
        self.assertEqual(type(obj), GoodGSONClass)

        d["@class"] = "not_there"
        obj = json.loads(json.dumps(d), cls=GSONDecoder)
        self.assertEqual(type(obj), dict)

    def test_redirect_settings_file(self):
        data = _load_redirect(os.path.join(test_dir, "test_settings.yaml"))
        self.assertEqual(
            data,
            {"old_module": {"old_class": {"@class": "new_class", "@module": "new_module"}}},
        )


if __name__ == "__main__":
    unittest.main()
