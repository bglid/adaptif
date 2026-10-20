/// Declarative macro for generating filter binding setup.
macro_rules! generate_filter_bindings {
    ($name: ident) => {
        #[pymethods]
        impl $name {
            #[getter]
            fn window_size(&self) -> usize {
                self.inner.window_size()
            }

            #[getter]
            fn block_size(&self) -> usize {
                self.inner.block_size()
            }

            #[getter]
            fn weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, Vec::from(self.inner.weights())))
            }

            fn load_weights(&mut self, weights: PyReadonlyArray1<f64>) -> PyResult<()> {
                // TODO: like with signals, decide whether weights should be copied or referenced
                let weights = weights.as_array().iter().copied().collect::<Vec<f64>>();

                self.inner.load_weights(weights).map_err(PyErr::from)?;

                Ok(())
            }

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
