#include <Python.h>
#include <frameobject.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <opcode.h>

#define LOG_KEY_VALUE(key, value) " " << key << "=" << value
#define LOG_SHOW(key) LOG_KEY_VALUE(#key, key)
#define LOG_64BITS(key) LOG_KEY_VALUE(#key, HexString(key, 16))
#define LOG_32BITS(key) LOG_KEY_VALUE(#key, HexString(key, 8))
#define LOG_16BITS(key) LOG_KEY_VALUE(#key, HexString(key, 4))
#define LOG_8BITS(key) LOG_KEY_VALUE(#key, HexString(key, 2))
#define LOG_BITS(key) LOG_KEY_VALUE(#key, HexString(key))
#define LOG_DWEHPE(type) LOG_KEY_VALUE(#type, ShowDW_EH_PE(type))

template <class T> std::string HexString(T num, int length = -1) {
  if (length == -1) {
    length = sizeof(T) / 2;
  }
  std::stringstream ss;
  ss << "0x" << std::uppercase << std::setfill('0') << std::setw(length)
     << std::hex << (+num & ((1 << (length * 4)) - 1));
  return ss.str();
}

PyObject *PyJit_EvalFrame(PyThreadState *ts, PyFrameObject *f, int throwflag) {
  LOG(INFO) << "PyJit_EvalFrame";
  const auto co_code = f->f_code->co_code;
  LOG(INFO) << LOG_SHOW(co_code->ob_type->tp_name)
            << LOG_SHOW(co_code->ob_refcnt) << LOG_SHOW(PyCode_Check(co_code))
            << LOG_SHOW(PyBytes_Check(co_code))
            << LOG_SHOW(PyBytes_Size(co_code));
  const auto co_code_size = PyBytes_Size(co_code);
  const auto buf = PyBytes_AsString(co_code);
  for (int i = 0; i < co_code_size; ++i) {
    LOG(INFO) << LOG_8BITS(buf[i]) << LOG_8BITS(LOAD_FAST)
              << LOG_8BITS(BINARY_ADD) << LOG_8BITS(RETURN_VALUE)
              << LOG_SHOW(sizeof(PyObject));
  }

  const auto na = f->f_code->co_argcount;
  LOG(INFO) << LOG_SHOW(na);

  for (int i = 0; i < na; i++) {
    LOG(INFO) << f->f_localsplus[i] << " "
              << f->f_localsplus[i]->ob_type->tp_name << " "
              << PyLong_AS_LONG(f->f_localsplus[i]);
  }

  return _PyEval_EvalFrameDefault(ts, f, throwflag);
}

static PyInterpreterState *inter() { return PyInterpreterState_Main(); }

extern "C" {
static PyObject *greet(PyObject *self, PyObject *args) {
  const char *name;

  /* Parse the input, from Python string to C string */
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;
  /* If the above function returns -1, an appropriate Python exception will
   * have been set, and the function simply returns NULL
   */

  printf("Hello %s\n", name);

  /* Returns a None Python object */
  Py_RETURN_NONE;
}

PyObject *jit_enable(PyObject *self, PyObject *args) {
  LOG(INFO) << "jit_enable";
  auto prev = _PyInterpreterState_GetEvalFrameFunc(inter());
  _PyInterpreterState_SetEvalFrameFunc(inter(), PyJit_EvalFrame);
  if (prev == PyJit_EvalFrame) {
    Py_RETURN_FALSE;
  }
  Py_RETURN_TRUE;
}

/* Define functions in module */
static PyMethodDef HelloMethods[] = {
    {"greet", greet, METH_VARARGS, "Greet somebody (in C)."},
    {"jit_enable", jit_enable, METH_NOARGS, "Enable JIT."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Create PyModuleDef stucture */
static struct PyModuleDef helloStruct = {PyModuleDef_HEAD_INIT,
                                         "hello",
                                         "",
                                         -1,
                                         HelloMethods,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL};

/* Module initialization */
PyObject *PyInit_hello(void) { return PyModule_Create(&helloStruct); }
}
