use log::info;
use pyo3::ffi::{
    PyFrameObject, PyInterpreterState_Get, PyThreadState, 
    _PyInterpreterState_GetEvalFrameFunc, _PyInterpreterState_SetEvalFrameFunc, PyObject,
};
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

static mut ORIGINAL_FRAME: Option<extern "C" fn(state: *mut PyThreadState, frame: *mut PyFrameObject, c: i32) -> *mut PyObject> = None;

extern "C" fn eval(state: *mut PyThreadState, frame: *mut PyFrameObject, c: i32) -> *mut PyObject {
    info!(target: "pyrjit", "eval()");
    unsafe {
        if let Some(original) = ORIGINAL_FRAME {
            original(state, frame, c)
        } else {
            panic!("original frame not found");
        }
    }
}

#[pyfunction]
fn enable() -> PyResult<()> {
    info!(target: "pyrjit", "enable()");
    let state = unsafe { PyInterpreterState_Get() };
    unsafe { ORIGINAL_FRAME = Some(_PyInterpreterState_GetEvalFrameFunc(state)) };
    unsafe { _PyInterpreterState_SetEvalFrameFunc(state, eval) };
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyrjit(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(enable, m)?)?;
    Ok(())
}
