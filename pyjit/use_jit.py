import dis
import hello

def print_hello():
    print('hello')

print("========== print_hello ==========")
dis.dis(print_hello)

def add(a, b):
    return a + b

print("========== add ==========")
dis.dis(add)

hello.jit_enable()
add(1, 2)
