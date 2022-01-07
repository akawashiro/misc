#! /bin/bash -eux

gcc -o libaudit.so -shared -fpic -fPIC -Wl,-soname,libaudit.so audit.c
gcc -o hello hello.c
LD_BIND_NOW=1 LD_AUDIT=libaudit.so ./hello
