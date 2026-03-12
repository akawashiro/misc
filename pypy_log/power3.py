import __pypy__
import time

def power10_fastest(x):
    return x * x * x * x * x * x  * x * x * x * x

def power_fast(n):
    if n == 0:
        return lambda x: 1
    else:
        esc = power_fast(n - 1)
        return lambda x: __pypy__.intop.int_mul(x, __pypy__._promote(esc)(x))

def power_naive(n, x):
    if n == 0:
        return 1
    else:
        return __pypy__.intop.int_mul(x, power_naive(n - 1, x))

power_fast_10 = power_fast(10)

assert(power10_fastest(2), power_fast_10(2))
assert(power10_fastest(2), power_naive(10, 2))
N_WARMUP = 10000
for _ in range(N_WARMUP):
    power10_fastest(2)
    power_fast_10(2)
    power_naive(10, 2)

N = 10000000
# fastest
t0_fastest = time.perf_counter()
for _ in range(N):
    power10_fastest(2)
t1_fastest = time.perf_counter()

# fast version
t0_fast = time.perf_counter()
for _ in range(N):
    power_fast_10(2)
t1_fast = time.perf_counter()

# naive version
t0_naive = time.perf_counter()
for _ in range(N):
    power_naive(10, 2)
t1_naive = time.perf_counter()

print("fastest", t1_fastest - t0_fastest)
print("fast:", t1_fast - t0_fast)
print("naive:", t1_naive - t0_naive)
