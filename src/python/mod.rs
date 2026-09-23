#![allow(
    clippy::multiple_inherent_impl,
    reason = "Defining methods that are only needed for this module and shouldn't be compiled otherwise."
)]
#![allow(
    clippy::needless_pass_by_value,
    reason = "PyArrays must be passed by value"
)]
mod macros;
use macros::generate_filter_bindings;

use pyo3::prelude::*;

use pyo3::exceptions::PyValueError;

use numpy::{PyArray1, PyReadonlyArray1};

use crate::Error;
use crate::algorithms::{Lms, Nlms};
use crate::filters::{AdaptiveFilter, BlockFilterBase, FilterBase};
use crate::types::signals::{InputSignal, NoiseReference};

impl Error {
    fn to_pyerr(&self) -> PyErr {
        match *self {
            Self::EmptyInputArr
            | Self::WindowSizeZero
            | Self::BlockSizeZero
            | Self::NoiseRefTooShort { .. }
            | Self::NonPositiveStepSize
            | Self::NonPositiveEpsilon => PyValueError::new_err(self.to_string()),
        }
    }
}

impl InputSignal {
    fn from_pyarray(input_signal: &PyReadonlyArray1<f64>) -> PyResult<InputSignal> {
        // TODO: if we split off a no_std core module, decide whether we want to copy
        // the input or use references (if we copy the input can be non-contiguous,
        // but it's more costly)

        // NOTE: references must be contiguous (leaving old comments/code in here
        // in case we later want to go back to references instead of copies):
        //
        // In Python, we use NumPy arrays as inputs, so we have to convert them to the Rust input types.
        // Because we're using slices in Rust, the input NumPy arrays need to be contiguous.
        // Strided slices like x[::2] or x[:, 0] are not allowed, and need to be made contiguous first.
        // let input_signal = input_signal.as_slice().map_err(|_e| {
        //     PyValueError::new_err(
        //         "input_signal must be a contiguous NumPy array; use numpy.ascontiguousarray().",
        //     )
        // })?;

        let input_signal = input_signal
            .as_array()
            .iter()
            .copied()
            .collect::<Vec<f64>>();

        let input_signal = InputSignal::new(input_signal).map_err(|e| e.to_pyerr())?;
        Ok(input_signal)
    }
}
impl NoiseReference {
    fn from_pyarray(noise_ref: &PyReadonlyArray1<f64>) -> PyResult<NoiseReference> {
        // let noise_ref = noise_ref.as_slice().map_err(|_e| {
        //     PyValueError::new_err(
        //         "noise_ref must be a contiguous NumPy array; use numpy.ascontiguousarray().",
        //     )
        // })?;
        let noise_ref = noise_ref.as_array().iter().copied().collect::<Vec<f64>>();

        let noise_ref = NoiseReference::new(noise_ref).map_err(|e| e.to_pyerr())?;
        Ok(noise_ref)
    }
}

#[derive(Debug, Clone, Copy)]
enum FilterOperation {
    Adapt,
    Filter,
}

// Because the wrappers for adapt() and filter() would only differ in one line,
// we use this underlying implementation.
fn adapt_filter_impl<'py, F>(
    filter: &mut F,
    py: Python<'py>,
    input_signal: PyReadonlyArray1<f64>,
    noise_ref: PyReadonlyArray1<f64>,
    op: FilterOperation,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    F: AdaptiveFilter,
{
    let input_signal = InputSignal::from_pyarray(&input_signal)?;
    let noise_ref = NoiseReference::from_pyarray(&noise_ref)?;

    let output_signal = match op {
        FilterOperation::Adapt => filter.adapt(&input_signal, &noise_ref),
        FilterOperation::Filter => filter.filter(&input_signal, &noise_ref),
    }
    .map_err(|e| e.to_pyerr())?;

    Ok(PyArray1::from_vec(py, output_signal))
}

#[pymodule]
mod adaptif {
    #[pymodule_export]
    use super::{BlockLMSFilter, LMSFilter, NLMSFilter};
}

#[pyclass]
pub struct LMSFilter(FilterBase<Lms>);
#[pymethods]
impl LMSFilter {
    #[new]
    fn new(mu: f64, window_size: usize) -> PyResult<Self> {
        let lms = Lms::new(mu).map_err(|e| e.to_pyerr())?;
        let filter = FilterBase::<Lms>::new(lms, window_size).map_err(|e| e.to_pyerr())?;

        Ok(Self(filter))
    }

    #[staticmethod]
    fn from_weights(mu: f64, weights: PyReadonlyArray1<f64>) -> PyResult<Self> {
        // TODO: like with signals, decide whether weights should be copied or referenced
        let lms = Lms::new(mu).map_err(|e| e.to_pyerr())?;

        let weights = weights.as_array().iter().copied().collect::<Vec<f64>>();

        let filter = FilterBase::<Lms>::from_weights(lms, weights).map_err(|e| e.to_pyerr())?;

        Ok(Self(filter))
    }
}

generate_filter_bindings!(LMSFilter);

#[pyclass]
pub struct NLMSFilter(FilterBase<Nlms>);
#[pymethods]
impl NLMSFilter {
    #[new]
    fn new(mu: f64, eps: f64, window_size: usize) -> PyResult<Self> {
        let nlms = Nlms::new(mu, eps).map_err(|e| e.to_pyerr())?;
        let filter = FilterBase::<Nlms>::new(nlms, window_size).map_err(|e| e.to_pyerr())?;

        Ok(Self(filter))
    }

    #[staticmethod]
    fn from_weights(mu: f64, eps: f64, weights: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let nlms = Nlms::new(mu, eps).map_err(|e| e.to_pyerr())?;

        let weights = weights.as_array().iter().copied().collect::<Vec<f64>>();

        let filter = FilterBase::<Nlms>::from_weights(nlms, weights).map_err(|e| e.to_pyerr())?;

        Ok(Self(filter))
    }
}

generate_filter_bindings!(NLMSFilter);

#[pyclass]
pub struct BlockLMSFilter(BlockFilterBase<Lms>);
#[pymethods]
impl BlockLMSFilter {
    #[new]
    fn new(mu: f64, window_size: usize, block_size: usize) -> PyResult<Self> {
        let lms = Lms::new(mu).map_err(|e| e.to_pyerr())?;
        let filter =
            BlockFilterBase::<Lms>::new(lms, window_size, block_size).map_err(|e| e.to_pyerr())?;

        Ok(Self(filter))
    }

    #[staticmethod]
    fn from_weights(mu: f64, weights: PyReadonlyArray1<f64>, block_size: usize) -> PyResult<Self> {
        let lms = Lms::new(mu).map_err(|e| e.to_pyerr())?;

        let weights = weights.as_array().iter().copied().collect::<Vec<f64>>();

        let filter = BlockFilterBase::<Lms>::from_weights(lms, weights, block_size)
            .map_err(|e| e.to_pyerr())?;

        Ok(Self(filter))
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.0.block_size()
    }
}

generate_filter_bindings!(BlockLMSFilter);
