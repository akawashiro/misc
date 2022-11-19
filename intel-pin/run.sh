#! /bin/bash -eux

# Get tar.gz from https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html
# This script uses Pin 3.25 Linux IA32 and intel64 (x86 32 bit and 64 bit).

rm -rf pin-gcc-linux
mkdir pin-gcc-linux
tar zxvf pin-gcc-linux.tar.gz --directory pin-gcc-linux
cd pin-gcc-linux/pin-3.25-98650-g8f6168173-gcc-linux/source/tools/ManualExamples/
make all TARGET=intel64 -j 40

# Show the number of instructions
../../../pin -t obj-intel64/inscount0.so -o inscount0.log -- /bin/ls
cat inscount0.log

# Show the number of instructions
../../../pin -t obj-intel64/inscount1.so -o inscount1.log -- /bin/ls
cat inscount1.log

# Can we show instruction opcode?
# Show the trace of IP register
../../../pin -t obj-intel64/itrace.so -- /bin/ls
cat itrace.out

# Show statistics of function calls
../../../pin -t obj-intel64/proccount.so -- /bin/ls
cat proccount.out

# Show all memory accesses
../../../pin -t obj-intel64/pinatrace.so -- /bin/ls
cat pinatrace.out

# Show all strace
# See https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl for syscall table
# What is https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl#L32?
../../../pin -t obj-intel64/strace.so -- /bin/ls
cat strace.out
