#! /bin/bash

set -eux -o pipefail

GHQ_ROOT=$(ghq root)
TMP_DIR=${HOME}/tmp
SCRIPT_DIR=$(
	cd $(dirname $0)
	pwd
)

TRITON_SRC_DIR=${GHQ_ROOT}/github.com/openai/triton

if [[ ! -d $TRITON_SRC_DIR ]]; then
	echo "Cloning triton..."
	ghq get openai/triton
fi

LLVM_GIT_HASH=$(cat ${TRITON_SRC_DIR}/cmake/llvm-hash.txt)
LLVM_SRC_DIR=${TMP_DIR}/triton-llvm
LLVM_BUILD_DIR=${TMP_DIR}/triton-llvm-build

if [ ! -d $LLVM_SRC_DIR ]; then
	git clone https://github.com/llvm/llvm-project.git $LLVM_SRC_DIR
fi
cd $LLVM_SRC_DIR
git checkout $LLVM_GIT_HASH

cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -S ${LLVM_SRC_DIR}/llvm -B ${LLVM_BUILD_DIR} -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER=$(which clang-15) -DCMAKE_CXX_COMPILER=$(which clang++-15)
cmake --build $LLVM_BUILD_DIR

export LLVM_BUILD_DIR=${LLVM_BUILD_DIR}

VENV_DIR=${SCRIPT_DIR}/build_and_install_triton_venv
python3 -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

cd ${TRITON_SRC_DIR}
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR CMAKE_C_COMPILER=$(which clang-15) CMAKE_CXX_COMPILER=$(which clang++-15) pip install -e python
COMPILE_COMMAND_JSON=$(realpath $(find ${TRITON_SRC_DIR}/python/build -name 'compile_commands.json'))
ln -sf ${COMPILE_COMMAND_JSON} ${TRITON_SRC_DIR}/compile_commands.json
