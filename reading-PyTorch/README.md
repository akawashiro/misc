# Reading PyTorch source code

```
$ clone-pytorch pytorch
$ git log --oneline | head -n 1
4648baa911 Revert "Use dynamo fake tensor mode in aot_autograd, move aot_autograd compilation to lowering time [Merger of 89672 and 89773] (#90039)"
$ python -m venv myenv
$ myenv/bin/activate
$ cd pytorch
$ pip install setuptools==59.5.0 pyyaml typing_extensions six requests dataclasses future
$ git submodule sync
$ git submodule update --init --recursive --jobs 0
$ export CMAKE_EXPORT_COMPILE_COMMANDS=ON
$ USE_CUDA=0 USE_CUDNN=0 USE_FBGEMM=0 USE_KINETO=0 USE_MKLDNN=0 USE_NNPACK=0 USE_QNNPACK=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0 USE_MPI=0 USE_SYSTEM_NCCL=0 BUILD_CAFFE2_OPS=0 BUILD_CAFFE2=0 USE_OPENMP=0 python3 setup.py develop
$ ln -sf build/compile_commands.json
```
