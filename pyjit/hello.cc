#include <Python.h>
#include <iostream>

PyObject *PyJit_EvalFrame(PyThreadState *ts, PyFrameObject *f, int throwflag) {
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
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
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
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
