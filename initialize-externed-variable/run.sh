#! /bin/bash -eux

gcc -fPIC -shared -Wl,-soname,libhoge.so -o libhoge.so hoge.c
gcc -o main main.c libhoge.so
./main
