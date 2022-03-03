#! /bin/bash -eux

gcc -o libaudit.so -g -shared -fpic -fPIC -Wl,-soname,libaudit.so audit.c
gcc -o hello hello.c -g
LD_AUDIT=libaudit.so ./hello
LD_BIND_NOW=1 LD_AUDIT=libaudit.so ./hello
# export LD_BIND_NOW=1
ltrace ./hello
