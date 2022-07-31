#! /bin/bash -eux

rm -rf ninja-build-Oz || true
rm -rf ninja-build-O2 || true
rm -rf ninja-build-O3 || true
rm -rf ninja-build-Oz-1 || true
rm -rf ninja-build-Oz-2 || true
rm -rf ninja-build-Oz-3 || true

if [[ ! -d ninja ]]
then
    git clone https://github.com/ninja-build/ninja.git
fi

build_dirs=()

build_dir="ninja-build-Oz"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-Oz -ffunction-sections" -DCMAKE_C_FLAGS="-Oz -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

build_dir="ninja-build-O2"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-O2 -ffunction-sections" -DCMAKE_C_FLAGS="-O2 -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

build_dir="ninja-build-O3"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-O3 -ffunction-sections" -DCMAKE_C_FLAGS="-O3 -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

export XX_PARAM_MIN_CROSSJUMP_INSNS=100
build_dir="ninja-build-Oz-${XX_PARAM_MIN_CROSSJUMP_INSNS}"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-Oz -ffunction-sections" -DCMAKE_C_FLAGS="-Oz -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

export XX_PARAM_MIN_CROSSJUMP_INSNS=1
build_dir="ninja-build-Oz-${XX_PARAM_MIN_CROSSJUMP_INSNS}"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-Oz -ffunction-sections" -DCMAKE_C_FLAGS="-Oz -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

export XX_PARAM_MIN_CROSSJUMP_INSNS=2
build_dir="ninja-build-Oz-${XX_PARAM_MIN_CROSSJUMP_INSNS}"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-Oz -ffunction-sections" -DCMAKE_C_FLAGS="-Oz -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

export XX_PARAM_MIN_CROSSJUMP_INSNS=3
build_dir="ninja-build-Oz-${XX_PARAM_MIN_CROSSJUMP_INSNS}"
cmake -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=${HOME}/gcc-install/bin/gcc -DCMAKE_CXX_COMPILER=${HOME}/gcc-install/bin/g++ -DCMAKE_CXX_FLAGS="-Oz -ffunction-sections" -DCMAKE_C_FLAGS="-Oz -ffunction-sections" -S ninja -B ${build_dir}
cmake --build ${build_dir}
build_dirs+=("${build_dir}")

rm file_size_compare

for b in "${build_dirs[@]}"
do
    ls -al ${b}/ninja >> file_size_compare
    objdump -d ${b}/ninja --demangle > objdump-${b}
    readelf -S ${b}/ninja > sections-${b}
    readelf -l ${b}/ninja > segments-${b}
done
