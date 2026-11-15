/// Declarative macro for generating filter binding setup.
macro_rules! generate_filter_bindings {
    ($name: ident) => {
        #[pymethods]
        impl $name {
            /// Return the `window_size` of a filter. This is also the number of filter weights.
            #[getter]
            fn window_size(&self) -> usize {
                self.inner.window_size()
            }

            /// Return the `block_size` of a filter.
            #[getter]
            fn block_size(&self) -> usize {
                self.inner.block_size()
            }

            /// Return the `weights` of a filter.
            #[getter]
            fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, Vec::from(self.inner.weights())))
            }

            /// Replace the filter's weights with previously learned `weights`.
            ///
            /// Args:
            ///     `weights`: One-dimensional float64 `NumPy` array
            ///         containing the previously learned weights. Length must match `window_size`.
            ///
            /// Raises:
            ///     `ValueError`: If the number of weights doesn't match the `window_size`.
            fn load_weights(&mut self, weights: PyReadonlyArray1<f64>) -> PyResult<()> {
                // TODO: like with signals, decide whether weights should be copied or referenced
                let weights = weights.as_array().iter().copied().collect::<Vec<f64>>();

                self.inner.load_weights(weights).map_err(PyErr::from)?;

                Ok(())
            }

            /// Adapt the filter weights using an input and reference signal.
            ///
            /// Updates the filter weights while processing the signals.
            ///
            /// Args:
            ///     `input_signal`: One-dimensional float64 NumPy array
            ///         containing the signal to process.
            ///     `noise_ref`: One-dimensional float64 NumPy array
            ///         containing the reference noise signal.
            ///
            /// Returns:
            ///     A one-dimensional `NumPy` array containing the error signal.
            ///
            /// Raises:
            ///     `ValueError`: If either signal is empty or `noise_ref` is shorter than `input_signal`
            fn adapt<'py>(
                &mut self,
                py: Python<'py>,
                input_signal: PyReadonlyArray1<f64>,
                noise_ref: PyReadonlyArray1<f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                adapt_filter_impl(
                    &mut self.inner,
                    py,
                    input_signal,
                    noise_ref,
                    FilterOperation::Adapt,
                )
            }

            /// Filter the input signal without updating filter weights.
            ///
            /// Uses the current filter weights to process the input
            /// and reference signals. Unlike `adapt()`, this method
            /// does not perform weight adaptation.
            ///
            /// Args:
            ///     `input_signal`: One-dimensional float64 NumPy array
            ///         containing the signal to process.
            ///     `noise_ref`: One-dimensional float64 NumPy array
            ///         containing the reference noise signal.
            ///
            /// Returns:
            ///     A `NumPy` array containing the filtered output.
            ///
            /// Raises:
            ///     `ValueError`: If the inputs violate filter constraints.
            fn filter<'py>(
                // self has to be mutable so that we can call adapt_filter_impl().
                // In Python there is no immutability, and we later pass an immutable reference
                // to the Rust filter() fn with the actual implementation, so this is fine.
                &mut self,
                py: Python<'py>,
                input_signal: PyReadonlyArray1<f64>,
                noise_ref: PyReadonlyArray1<f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                adapt_filter_impl(
                    &mut self.inner,
                    py,
                    input_signal,
                    noise_ref,
                    FilterOperation::Filter,
                )
            }
        }
    };
}

pub(crate) use generate_filter_bindings;
