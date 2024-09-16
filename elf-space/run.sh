#!/bin/bash -eux

nasm -f elf64 ./hello.asm
ld -o hello ./hello.o
gcc-11 -o lifegame ./lifegame.c
gcc-11 -o convert ./convert.c
./convert
# ./hello_with_parasite

gcc-11 -o whitespace ./whitespace.c
./whitespace ./helloworld.ws
