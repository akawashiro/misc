#include <cstdio>

void f() { throw 42; }

int main() {
    try {
        f();
    } catch (int i) {
        printf("i = %d\n", i);
    }
}
