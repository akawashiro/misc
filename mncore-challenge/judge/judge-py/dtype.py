from enum import Enum

import numpy as np


class Dtype(Enum):
    ULong = 0
    UInt = 1
    UShort = 2

    Long = 3
    Int = 4
    Short = 5

    Double = 6
    Float = 7
    Half = 8

    String = 9

    def __str__(self):
        return DTYPE_TO_STR[self]

    def is_unsigned_integral(self):
        return self in [Dtype.UShort, Dtype.UInt, Dtype.ULong]

    def is_integral(self):
        return self in [Dtype.Short, Dtype.Int, Dtype.Long]

    def is_floating(self):
        return self in [Dtype.Half, Dtype.Float, Dtype.Double]

    def num_bits(self):
        if self == Dtype.ULong or self == Dtype.Long or self == Dtype.Double:
            return 64

        if self == Dtype.UInt or self == Dtype.Int or self == Dtype.Float:
            return 32

        if self == Dtype.UShort or self == Dtype.Short or self == Dtype.Half:
            return 16

        if self == Dtype.String:
            return 8

        raise AssertionError(f"Unknown dtype: {self}")

    def exponent_bits(self):
        TABLE = {
            Dtype.Double: 11,
            Dtype.Float: 8,
            Dtype.Half: 6,
        }
        assert self in TABLE, f"{self} is not a floating point type"

        return TABLE[self]

    def mantissa_bits(self):
        TABLE = {
            Dtype.Double: 52,
            Dtype.Float: 23,
            Dtype.Half: 9,
        }
        assert self in TABLE, f"{self} is not a floating point type"

        return TABLE[self]

    def num_elements_in_long_word(self):
        return 64 // self.num_bits()

    def to_unsigned(self):
        TABLE = {
            Dtype.Long: Dtype.ULong,
            Dtype.Int: Dtype.UInt,
            Dtype.Short: Dtype.UShort,
            Dtype.Double: Dtype.ULong,
            Dtype.Float: Dtype.UInt,
            Dtype.Half: Dtype.UShort,
        }
        assert self in TABLE, f"{self} is not a signed integer nor a floating point type"

        return TABLE[self]

    def to_numpy_dtype(self):
        TABLE = {
            Dtype.ULong: np.uint64,
            Dtype.UInt: np.uint32,
            Dtype.UShort: np.uint16,
            Dtype.Long: np.int64,
            Dtype.Int: np.int32,
            Dtype.Short: np.int16,
            Dtype.Double: np.float64,
            Dtype.Float: np.float32,
            Dtype.Half: np.float16,
            Dtype.String: str,
        }
        return TABLE[self]

    @staticmethod
    def deserialize(dtype_str):
        for dtype, s in DTYPE_TO_STR.items():
            if s == dtype_str:
                return dtype

        raise AssertionError(f"Unknown dtype: {dtype_str}")


class NumpyHalf:
    @staticmethod
    def exponent_bits():
        return 5

    @staticmethod
    def mantissa_bits():
        return 10


DTYPE_TO_STR = {
    Dtype.ULong: "ULong",
    Dtype.UInt: "UInt",
    Dtype.UShort: "UShort",
    Dtype.Long: "Long",
    Dtype.Int: "Int",
    Dtype.Short: "Short",
    Dtype.Double: "Double",
    Dtype.Float: "Float",
    Dtype.Half: "Half",
    Dtype.String: "String",
}
