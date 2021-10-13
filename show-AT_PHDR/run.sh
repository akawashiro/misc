#! /bin/bash -eux

gcc main.c -o main.static.out -static
~/elftools/build/reorder_phdrs -i main.static.out -o main.static.out.reorderd -r "1,0,2,3,4,5,6,7,8,9"
chmod u+x main.static.out.reorderd
~/elftools/build/reorder_phdrs -i main.static.out -o main.static.out.broken.reorderd -r "3,1,2,0,4,5,6,7,8,9"
chmod u+x main.static.out.broken.reorderd

# gcc -Wl,--verbose main.c -o main.out 2>&1 | tee default_gcc.lds
# cp default_gcc.lds change_addr.lds
# nvim change_addr.lds
# gcc main.c -T change_addr.lds -o main.lds.out
# ./main.lds.out
