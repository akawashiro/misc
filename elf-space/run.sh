#!/bin/bash -eux

nasm -f elf64 ./hello_syscall.asm
ld -o hello_syscall ./hello_syscall.o
gcc-11 -o lifegame ./lifegame.c
gcc-11 -o convert ./convert.c
./convert
./hello_syscall_with_parasite

