import hello

def hoge():
    print('hoge')

hello.jit_enable()
hello.greet('World')
hoge()
