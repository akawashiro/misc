#! /bin/bash -eux

gcc -shared -fpic -fPIC foo.c -o libfoo.so
gcc -o main main.c libfoo.so
LD_BIND_NOW=1 ./main
