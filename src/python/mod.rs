#![allow(
    clippy::multiple_inherent_impl,
    reason = "Defining methods that are only needed for this module and shouldn't be compiled otherwise."
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

// In Python, we use NumPy arrays as inputs, so we have to convert them to the Rust input types.
// Because we're using slices in Rust, the input NumPy arrays need to be contiguous.
// Strided slices like x[::2] or x[:, 0] are not allowed, and need to be made contiguous first.
impl<'a> InputSignal<'a> {
    fn from_pyarray(input_signal: &'a PyReadonlyArray1<f64>) -> PyResult<InputSignal<'a>> {
        let input_signal = input_signal.as_slice().map_err(|_e| {
            PyValueError::new_err(
                "input_signal must be a contiguous NumPy array; use numpy.ascontiguousarray().",
            )
        })?;

        let input_signal = InputSignal::new(input_signal).map_err(|e| e.to_pyerr())?;
        Ok(input_signal)
    }
}
impl<'a> NoiseReference<'a> {
    fn from_pyarray(noise_ref: &'a PyReadonlyArray1<f64>) -> PyResult<NoiseReference<'a>> {
        let noise_ref = noise_ref.as_slice().map_err(|_e| {
            PyValueError::new_err(
                "noise_ref must be a contiguous NumPy array; use numpy.ascontiguousarray().",
            )
        })?;
        let noise_ref = NoiseReference::new(noise_ref).map_err(|e| e.to_pyerr())?;
        Ok(noise_ref)
    }
}

#[derive(Debug, Clone, Copy)]
enum FilterOperation {
    Adapt,
    Filter,
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "PyArrays must be passed by value"
)]
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

    #[getter]
    fn block_size(&self) -> usize {
        self.0.block_size()
    }
}

generate_filter_bindings!(BlockLMSFilter);
