#! /bin/bash -eux

for s in $(find . -name '*.c'); do
    filename=$(basename "${s}")
    base=$(echo "${filename%.*}")
    for compiler in clang-14 clang-15 gcc-11 gcc-12; do
        output=${base}-${compiler}
        ${compiler} -o ${output} ${s}
        objdump --disassemble=main -M intel ${output}
    done
done
