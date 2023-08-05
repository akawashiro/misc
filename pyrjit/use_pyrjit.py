import pyrjit
print(pyrjit.version())
pyrjit.enable()

def test():
    print("Hello World!")

def add(a, b):
    return a + b

test()
add(1, 2)
