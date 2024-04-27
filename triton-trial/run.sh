#! /bin/bash

set -eux -o pipefail

GHQ_ROOT=$(ghq root)
TMP_DIR=${HOME}/tmp

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

cd $TRITON_SRC_DIR
python3 -m venv venv
source venv/bin/activate
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR CMAKE_C_COMPILER=$(which clang-15) CMAKE_CXX_COMPILER=$(which clang++-15) pip install -e python
