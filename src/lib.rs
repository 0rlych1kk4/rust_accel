use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2};

#[pyfunction]
fn clone_grad(grad: Vec<f32>, rows: usize, cols: usize) -> Vec<f32> {
    let array = ArrayView2::from_shape((rows, cols), &grad).unwrap();
    array.to_owned().into_raw_vec()
}

#[pyfunction]
fn optimize_reconstruction(input: Vec<f32>, grads: Vec<Vec<f32>>, rows: usize, cols: usize) -> f32 {
    let input_array = ArrayView2::from_shape((rows, cols), &input).unwrap();
    let mut loss = 0.0;
    
    for grad in grads {
        let grad_array = ArrayView2::from_shape((rows, cols), &grad).unwrap();
        loss += (&input_array * &grad_array).sum();
    }
    
    loss
}

#[pymodule]
fn rust_accel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clone_grad, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_reconstruction, m)?)?;
    Ok(())
}
