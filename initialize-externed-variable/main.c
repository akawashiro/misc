#include <stdio.h>

// extern unsigned long hoge_var = 0xaaaaaaaa;
// main.c:3:22: warning: ‘hoge_var’ initialized and declared ‘extern’
//     3 | extern unsigned long hoge_var = 0xaaaaaaaa;
//       |
extern unsigned long hoge_var;

int main() {
    printf("hoge_var = %lx\n", hoge_var);
    return 0;
}
