#!/bin/bash -eux

nasm -f elf64 ./hello.asm
ld -o hello ./hello.o

gcc-11 -o lifegame ./lifegame.c
gcc-11 -o ./embed_lifegame_in_hello ./embed_lifegame_in_hello.c
./embed_lifegame_in_hello

./hello_with_lifegame

python3 ./emit_hello_whitespace.py > ./whitespace_program_to_embed.ws
xxd -include ./whitespace_program_to_embed.ws > ./whitespace_program_to_embed.h
gcc-11 -fno-stack-protector -o ./whitespace_to_embed ./whitespace_to_embed.c

gcc-11 -o ./embed_whitespace_in_hello ./embed_whitespace_in_hello.c
./embed_whitespace_in_hello

./hello_with_whitespace
