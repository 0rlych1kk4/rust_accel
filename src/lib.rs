// src/lib.rs  (safe: no unwrap across Python boundary, NumPy-native, parallelized)
use ndarray::Axis;
use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Returns a flattened, owned copy of a 2D gradient (row-major).
/// Python input: numpy.ndarray[np.float32] with ndim=2
#[pyfunction]
fn clone_grad(grad: PyReadonlyArray2<f32>) -> PyResult<Vec<f32>> {
    // as_array() is a view; to_owned() makes an owned ndarray; into_raw_vec flattens row-major.
    Ok(grad.as_array().to_owned().into_raw_vec())
}

/// Computes a loss: sum_k sum_{i,j} input[i,j] * grads[k,i,j]
///
/// Python inputs:
/// - input: numpy.ndarray[np.float32] shape (rows, cols)
/// - grads: numpy.ndarray[np.float32] shape (n, rows, cols)
#[pyfunction]
fn optimize_reconstruction(input: PyReadonlyArray2<f32>, grads: PyReadonlyArray3<f32>) -> PyResult<f32> {
    let x = input.as_array();
    let g = grads.as_array();

    // Validate shapes (return Python exception; never panic)
    if x.ndim() != 2 {
        return Err(PyValueError::new_err("input must be a 2D numpy array"));
    }
    if g.ndim() != 3 {
        return Err(PyValueError::new_err("grads must be a 3D numpy array with shape (n, rows, cols)"));
    }

    let x_shape = x.shape();
    let g_shape = g.shape();

    // grads: (n, rows, cols)
    if g_shape[1] != x_shape[0] || g_shape[2] != x_shape[1] {
        return Err(PyValueError::new_err(format!(
            "Shape mismatch: input is ({}, {}), grads must be (n, {}, {}) but got ({}, {}, {})",
            x_shape[0], x_shape[1],
            x_shape[0], x_shape[1],
            g_shape[0], g_shape[1], g_shape[2],
        )));
    }

    let n = g_shape[0];

    // Parallelize across k (each slice independent)
    let loss: f32 = (0..n)
        .into_par_iter()
        .map(|k| {
            let grad_k = g.index_axis(Axis(0), k);
            (&x * &grad_k).sum()
        })
        .sum();

    Ok(loss)
}

#[pymodule]
fn rust_accel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clone_grad, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_reconstruction, m)?)?;
    Ok(())
}
