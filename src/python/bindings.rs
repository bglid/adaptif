#![allow(
    clippy::needless_pass_by_value,
    reason = "PyArrays must be passed by value"
)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use numpy::{PyArray1, PyReadonlyArray1};

use super::macros::generate_filter_bindings;

use crate::Error;
use crate::algorithms::Algorithm;
use crate::filters::{FilterBase, ProcessingMode};
use crate::filters::{LmsFilter, LmsOptions, NlmsFilter, NlmsOptions, RlsFilter, RlsOptions};
use crate::types::signals::{InputSignal, NoiseReference};

impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::EmptyInputArr
            | Error::WindowSizeZero
            | Error::BlockSizeZero
            | Error::NoiseRefTooShort { .. }
            | Error::NonPositiveStepSize
            | Error::InvalidForgettingFactorRange
            | Error::NonPositivePInitScale
            | Error::NonPositiveEpsilon
            | Error::WeightsDontMatchWindowSize { .. } => PyValueError::new_err(value.to_string()),
        }
    }
}

// Python floats are double precision, so we use f64 instead of a generic
impl TryFrom<&PyReadonlyArray1<'_, f64>> for InputSignal<f64> {
    type Error = PyErr;
    fn try_from(value: &PyReadonlyArray1<f64>) -> Result<Self, Self::Error> {
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

        let input_signal = value.as_array().iter().copied().collect::<Vec<f64>>();

        let input_signal = InputSignal::new(input_signal).map_err(PyErr::from)?;
        Ok(input_signal)
    }
}

impl TryFrom<&PyReadonlyArray1<'_, f64>> for NoiseReference<f64> {
    type Error = PyErr;
    fn try_from(value: &PyReadonlyArray1<f64>) -> Result<Self, Self::Error> {
        // let noise_ref = noise_ref.as_slice().map_err(|_e| {
        //     PyValueError::new_err(
        //         "noise_ref must be a contiguous NumPy array; use numpy.ascontiguousarray().",
        //     )
        // })?;
        let noise_ref = value.as_array().iter().copied().collect::<Vec<f64>>();

        let noise_ref = NoiseReference::new(noise_ref).map_err(PyErr::from)?;
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
fn adapt_filter_impl<'py, A>(
    filter: &mut FilterBase<f64, A>,
    py: Python<'py>,
    input_signal: PyReadonlyArray1<f64>,
    noise_ref: PyReadonlyArray1<f64>,
    op: FilterOperation,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    A: Algorithm<f64>,
{
    let input_signal = InputSignal::try_from(&input_signal)?;
    let noise_ref = NoiseReference::try_from(&noise_ref)?;

    let output_signal = match op {
        FilterOperation::Adapt => filter.adapt(&input_signal, &noise_ref),
        FilterOperation::Filter => filter.filter(&input_signal, &noise_ref),
    }
    .map_err(PyErr::from)?;

    Ok(PyArray1::from_vec(py, output_signal))
}

#[pymodule]
mod adaptif {
    #[pymodule_export]
    use super::{LMSFilter, NLMSFilter, RLSFilter};
}

#[pyclass]
pub struct LMSFilter {
    inner: LmsFilter<f64>,
}
#[pymethods]
impl LMSFilter {
    #[new]
    #[pyo3(signature = (mu, window_size, block_size=1))]
    fn new(mu: f64, window_size: usize, block_size: usize) -> PyResult<Self> {
        let filter = LmsFilter::new(
            LmsOptions { mu, window_size },
            ProcessingMode::Block { block_size }, // Note that block_size: 1 is functionally
                                                  // equivalent to ProcessingMode::Sample
        )
        .map_err(PyErr::from)?;

        Ok(Self { inner: filter })
    }
}

generate_filter_bindings!(LMSFilter);

#[pyclass]
pub struct NLMSFilter {
    inner: NlmsFilter<f64>,
}
#[pymethods]
impl NLMSFilter {
    #[new]
    #[pyo3(signature = (mu, eps, window_size, block_size=1))]
    fn new(mu: f64, eps: f64, window_size: usize, block_size: usize) -> PyResult<Self> {
        let filter = NlmsFilter::new(
            NlmsOptions {
                mu,
                eps,
                window_size,
            },
            ProcessingMode::Block { block_size },
        )
        .map_err(PyErr::from)?;

        Ok(Self { inner: filter })
    }
}

generate_filter_bindings!(NLMSFilter);

#[pyclass]
pub struct RLSFilter {
    inner: RlsFilter<f64>,
}
#[pymethods]
impl RLSFilter {
    #[new]
    #[pyo3(signature = (forgetting_factor, p_init_scale, window_size, block_size=1))]
    fn new(
        forgetting_factor: f64,
        p_init_scale: f64,
        window_size: usize,
        block_size: usize,
    ) -> PyResult<Self> {
        let filter = RlsFilter::new(
            RlsOptions {
                forgetting_factor,
                p_init_scale,
                window_size,
            },
            ProcessingMode::Block { block_size },
        )
        .map_err(PyErr::from)?;

        Ok(Self { inner: filter })
    }
}

generate_filter_bindings!(RLSFilter);
