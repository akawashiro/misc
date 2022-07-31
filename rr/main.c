#include <stdio.h>
#include <stdint.h>

int main(){
    uint64_t i = 0;
    uint64_t x;
    for(i = 0; i < 0xf; i++){
        x += (i * i);
    }
    return 0;
}
