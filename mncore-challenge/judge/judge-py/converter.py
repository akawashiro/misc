import numpy as np
from dtype import Dtype, NumpyHalf


def decompose_float(f: int, dtype: Dtype):
    mantissa = f & ((1 << dtype.mantissa_bits()) - 1)
    f >>= dtype.mantissa_bits()

    exponent = f & ((1 << dtype.exponent_bits()) - 1)
    f >>= dtype.exponent_bits()

    sign = f

    return sign, exponent, mantissa


def compose_float(sign: int, exponent: int, mantissa: int, dtype: Dtype):
    f = sign
    f = (f << dtype.exponent_bits()) + exponent
    f = (f << dtype.mantissa_bits()) + mantissa
    return f


def normalize_float(value: int, dtype: Dtype):
    sign, exponent, mantissa = decompose_float(value, dtype)
    if exponent == 0 or exponent == ((1 << dtype.exponent_bits()) - 1):
        mantissa = 0
    return compose_float(sign, exponent, mantissa, dtype)


def cast(value: int, from_dtype: Dtype, to_dtype: Dtype):
    sign, exponent, mantissa = decompose_float(value, from_dtype)

    exponent = exponent - (2 ** (from_dtype.exponent_bits() - 1)) + (2 ** (to_dtype.exponent_bits() - 1)) if exponent != 0 else 0
    mantissa = mantissa * (2 ** to_dtype.mantissa_bits()) // (2 ** from_dtype.mantissa_bits())

    assert 0 <= exponent and exponent < 2 ** to_dtype.exponent_bits()
    # exponent = max(0, min((2 ** to_dtype.exponent_bits()) - 1, exponent))

    return compose_float(sign, exponent, mantissa, to_dtype)


def from_payload(payload: str, dtype: Dtype):
    if dtype == Dtype.ULong:
        return [int(payload, 16)]

    if dtype == Dtype.UInt:
        return [int(payload[0:8], 16), int(payload[8:16], 16)]

    if dtype == Dtype.UShort:
        return [int(payload[0:4], 16), int(payload[4:8], 16), int(payload[8:12], 16), int(payload[12:16], 16)]

    if dtype == Dtype.Long or dtype == Dtype.Int or dtype == Dtype.Short:
        values = from_payload(payload, dtype.to_unsigned())
        values_signed = []
        for v in values:
            if v & (1 << (dtype.num_bits() - 1)):
                v -= 1 << dtype.num_bits()
            values_signed += [v]
        return values_signed

    if dtype == Dtype.Double or dtype == Dtype.Float or dtype == Dtype.Half:
        values = [normalize_float(v, dtype) for v in from_payload(payload, dtype.to_unsigned())]
        if dtype == Dtype.Half:
            dtype = Dtype.Float
            values = [cast(v, Dtype.Half, dtype) for v in values]
        np_values = np.array(values, dtype=dtype.to_unsigned().to_numpy_dtype()).view(dtype.to_numpy_dtype())
        return [float(v) for v in np_values]

    if dtype == Dtype.String:
        res = ""
        for i in range(8):
            res += chr(int(payload[i * 2 : i * 2 + 2], 16))
        return res

    raise AssertionError(f"other dtypes are not supported: dtype={dtype}")


def bcast_lw(lw_values, dtype):
    return lw_values * dtype.num_elements_in_long_word()


def to_payload(lw_values, dtype: Dtype):
    if len(lw_values) == 1 and dtype.num_elements_in_long_word() > 1:
        lw_values = bcast_lw(lw_values, dtype)

    assert len(lw_values) == dtype.num_elements_in_long_word(), f"padding is not supported: lw_values={lw_values} dtype={dtype}"

    if dtype == Dtype.ULong:
        return f"{lw_values[0]:016X}"

    if dtype == Dtype.UInt:
        return f"{lw_values[0]:08X}{lw_values[1]:08X}"

    if dtype == Dtype.UShort:
        return f"{lw_values[0]:04X}{lw_values[1]:04X}{lw_values[2]:04X}{lw_values[3]:04X}"

    if dtype == Dtype.Long or dtype == Dtype.Int or dtype == Dtype.Short:
        lw_values = [(v + (1 << dtype.num_bits()) if v < 0 else v) for v in lw_values]
        return to_payload(lw_values, dtype.to_unsigned())

    if dtype == Dtype.Double:
        bits = np.float64(lw_values[0]).view(np.uint64)
        return f"{bits:016X}"

    if dtype == Dtype.Float:
        bits = np.float32(lw_values).view(np.uint32)
        return f"{bits[0]:08X}{bits[1]:08X}"

    if dtype == Dtype.Half:
        bits = [cast(np.float16(v).view(np.uint16), NumpyHalf, Dtype.Half) for v in lw_values]
        return f"{bits[0]:04X}{bits[1]:04X}{bits[2]:04X}{bits[3]:04X}"

    raise AssertionError(f"other dtypes are not supported: dtype={dtype}")
