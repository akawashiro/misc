#!/bin/bash -eux

nasm -f elf64 ./hello_syscall.asm
ld -o hello_syscall ./hello_syscall.o
gcc -o lifegame ./lifegame.c
gcc -o convert ./convert.c
./convert
./hello_syscall_with_parasite
