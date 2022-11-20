# How to use ninjatracing

```
git clone git@github.com:pytorch/pytorch.git
cd pytorch
git checkout v1.13.0
git submodule update --recursive --init
mkdir -p build
rm -rf *
CC=clang-14 CXX=clang++-14 cmake -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -G Ninja -S ~/pytorch -B ~/pytorch/build -DUSE_ITT=OFF -DUSE_FBGEMM=OFF -DUSE_XNNPACK=OFF -DUSE_MAGMA=OFF -DUSE_METAL=OFF -DUSE_NUMA=OFF -DBUILD_PYTHON=OFF
ninja
git clone git@github.com:nico/ninjatracing.git
./ninjatracing/ninjatracing .ninja_log > pytorch_ninja_build_trace.json
```

