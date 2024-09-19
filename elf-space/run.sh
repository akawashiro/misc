#!/bin/bash -eux

nasm -f elf64 ./hello.asm
ld -o hello ./hello.o
gcc-11 -o lifegame ./lifegame.c
gcc-11 -o convert ./convert.c
./convert
# ./hello_with_parasite

gcc-11 -o whitespace ./whitespace.c
./whitespace

python3 ./emit_hello_whitespace.py > ./whitespace_program_to_embed.ws
xxd -include ./whitespace_program_to_embed.ws > ./whitespace_program_to_embed.h
gcc-11 -o ./whitespace_to_embed ./whitespace_to_embed.c
