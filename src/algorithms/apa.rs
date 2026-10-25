use std::num::NonZero;
use std::ops::{Deref, DerefMut};

use crate::types::buffers::{NoiseWindow, SampleBuffer};
use crate::types::signals::OutputSample;
use crate::types::{FilterWeights, Float, WindowSize};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone, PartialEq)]
/// Affine Projection Algorithm.
pub struct Apa<F: Float> {
    /// Step size for weight updates.
    mu: F,
    /// Regularization term for matrix inversion.
    eps: F,
    /// Projection order (L) defining number of recent input vectors to use in update.
    projection_order: usize, // NOTE: See question on data type for projection order
    /// Stores up to L recent input vectors.
    input_history: InputHistory<F>,
    /// Stores the L most recent desired outputs.
    desired_history: DesiredHistory<F>,
    /// Preallocated the capacity for storing the recalculated error vector per weight update.
    error_vector: ErrorVector<F>,
}

impl<F: Float> Algorithm<F> for Apa<F> {
    /// Updates the filter weights using the following algorithm:
    ///
    /// $w_{n+1} = ``w_n`` + \mu * X(n)(X(n)^{T}X(n) + \epsilon I)^{-1} e(n)$
    /// where $``X(n)``$ is the matrix of the last $``L``$ noise reference signal vectors,
    /// where $``L``$ is the projection order,
    /// where $``I``$ is the identity matrix of the filter's projection order,
    /// and $``e(n)``$ is the error vector for the current time step,
    /// whose vector size is decided by $``L``$.
    fn update_step(
        &mut self,
        weights: &mut FilterWeights<F>,
        error: OutputSample<F>,
        noise_window: NoiseWindow<F>,
    ) {
        // Updates Input history once window size is known
        if self.input_history.window_size != weights.window_size() {
            self.input_history = InputHistory::new(self.projection_order, weights.window_size());
        }

        let desired_output = weights
            .iter()
            .copied()
            .zip(noise_window.iter())
            .fold(error.0, |acc, (w, x)| acc + w * x); // TODO: Question, should error.0 (OutputSample<F>) also be error.inner?
        self.update_desired_history(desired_output);

        self.update_input_history(&noise_window);

        // recalculating the entire error vector with updated weights
        self.recalculate_error_vector(weights);

        // let mut inner_matrix = vec![F::zero(); self.projection_order * self.projection_order];

        // $w_{n+1} = ``w_n`` + \mu * X(n)(X(n)^{T}X(n) + \epsilon I)^{-1} e(n)$
        // for (w, x) in weights.iter_mut().zip() {
        //     *w += self.mu * self.input_history.iter().zip(flat_input_dot.zip());
        // }
    }
}

impl<F: Float> Apa<F> {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    /// Returns an error if eps <= 0.0.
    /// Returns an error if projection order <= 0.
    pub fn new(mu: F, eps: F, projection_order: usize) -> Result<Self> {
        if mu <= F::zero() {
            return Err(Error::NonPositiveStepSize);
        }

        if eps <= F::zero() {
            return Err(Error::NonPositiveEpsilon);
        }

        // NOTE: Question, should projection order be usize? It's an int, and can't be negative
        // What's a better design decision? To make it usize to avoid ever having negative values
        // OR, have it signed and return an error if ever negative (i.e <= 0 here)
        if projection_order == 0 {
            return Err(Error::InvalidProjectionOrder);
        }

        Ok(Apa {
            mu,
            eps,
            projection_order,
            input_history: InputHistory::new(projection_order, WindowSize::new(1)?),
            desired_history: DesiredHistory::new(projection_order),
            error_vector: ErrorVector::new(projection_order),
        })
    }

    /// Updates the input history vector used in the APA weight update.
    fn update_input_history(&mut self, noise_ref: &NoiseWindow<F>) {
        // Discards the oldest window of input (up to 1 window size) and inserts newest at end
        // for i in 0..noise_ref.len() {
        //     if let Some(&sample) = noise_ref.get(i) {
        //         self.input_history.inner.push(sample);
        //     }
        // }
        for sample in noise_ref.iter() {
            self.input_history.inner.push(sample);
        }

        // if let Some(last_window) = self.input_history.inner.get_mut(
        //     (self.input_history.inner.len() - *self.input_history.window_size)
        //         ..self.input_history.inner.len(),
        // ) {
        //     last_window
        //         .iter_mut()
        //         .zip(noise_ref.iter())
        //         .for_each(|(input_h, noise_ref)| *input_h = noise_ref);
        // }
    }

    // TODO: Look over again, is this really necessary?
    /// Updates the desired history vector used in the APA weight update.
    fn update_desired_history(&mut self, desired_output: F) {
        self.desired_history.inner.push(desired_output);
    }

    /// Rebuilds the entire ``ErrorVector`` from scratch using new weight updates.
    pub fn recalculate_error_vector(&mut self, weights: &mut FilterWeights<F>) {
        // have to get oldest first because push inserts at front
        #[allow(clippy::missing_panics_doc, reason = "failure independent of fn input")]
        for i_index in (0..self.projection_order).rev() {
            let desired_sample = *self.desired_history.get(i_index).expect(
                "
                desired sample get should only fail if index >= desired_history.len().
                desired_history.len() == projection_order, so index (i) is always <= desired_history.len(),",
            );

            // prediction using flattened input history matrix by weights
            let predicted_sample =
                weights
                    .iter()
                    .enumerate()
                    .fold(F::zero(), |acc, (j_index, &w)| {
                        let x = *self
                            .input_history
                            .get(
                                i_index * (*weights.window_size()) + (*weights.window_size())
                                    - 1
                                    - j_index,
                            )
                            .expect(""); // TODO: expect message here check
                        acc + w * x
                    });
            self.error_vector.push(desired_sample - predicted_sample);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L x Window size dimensional matrix of previous input vectors, where L = projection order.
struct InputHistory<F: Float> {
    /// Flattened matrix to store the L x Window size matrix.
    inner: SampleBuffer<F>, // NOTE: doing flattened matrix since window size is known to avoid Vec<Vec<...>>
    /// Window size of APA filter.
    window_size: WindowSize,
}

impl<F: Float> InputHistory<F> {
    pub fn new(projection_order: usize, window_size: WindowSize) -> Self {
        let capacity = NonZero::new(projection_order * (*window_size))
            .expect("Projection order and WindowSize should ensure that capacity is > 0.");
        InputHistory {
            inner: (SampleBuffer::new(capacity)),
            window_size,
        }
    }
}
impl<F: Float> Deref for InputHistory<F> {
    type Target = SampleBuffer<F>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for InputHistory<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L-dimensional vector of the desired output history, where L = projection order.
struct DesiredHistory<F: Float> {
    inner: SampleBuffer<F>,
}
impl<F: Float> DesiredHistory<F> {
    pub fn new(projection_order: usize) -> Self {
        // NOTE: projection order is capacity
        let capacity = NonZero::new(projection_order).expect(
            "Desired history is initialized by APA projection order, which must be greater than 0.",
        );
        DesiredHistory {
            inner: SampleBuffer::new(capacity),
        }
    }
}
impl<F: Float> Deref for DesiredHistory<F> {
    type Target = SampleBuffer<F>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for DesiredHistory<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L-dimensional vector of the newly calculated error vectory, where L = projection order.
struct ErrorVector<F: Float> {
    inner: SampleBuffer<F>,
}
impl<F: Float> ErrorVector<F> {
    pub fn new(projection_order: usize) -> Self {
        let capacity = NonZero::new(projection_order).expect(
            "Desired history is initialized by APA projection order, which must be greater than 0.",
        );
        ErrorVector {
            inner: SampleBuffer::new(capacity),
        }
    }
}
impl<F: Float> Deref for ErrorVector<F> {
    type Target = SampleBuffer<F>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<F: Float> DerefMut for ErrorVector<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{all_approx_equal, block_noise_buffer_from},
        types::{BlockSize, FilterWeights, WindowSize, buffers::BlockError},
    };

    #[test]
    fn update_apa_1() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(1).unwrap();

        let mut apa = Apa::new(0.5, 1.0, 1).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(2.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[1.0, -1.0], window_size, block_size);
        let expected = [1.0 / 3.0, -1.0 / 3.0];
        let mut weights = FilterWeights::new(window_size);

        apa.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn update_apa_2() {
        let window_size = WindowSize::new(2).unwrap();
        let block_size = BlockSize::new(2).unwrap();

        let mut apa = Apa::new(0.5, 1.0, 2).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(2.0));
            e_n.push(OutputSample(-4.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[1.0, 0.0, 1.0], window_size, block_size);
        let expected = [0.5, -1.0];
        let mut weights = FilterWeights::new(window_size);

        apa.update_block(&mut weights, &e_n, &x_n);

        assert!(all_approx_equal(weights.iter(), expected.iter()));
    }

    #[test]
    fn mu_range() {
        Apa::new(0.5, 1.0, 2).unwrap();

        assert!(matches!(
            Apa::new(0.0, 1.0, 2),
            Err(Error::NonPositiveStepSize)
        ));
        assert!(matches!(
            Apa::new(-1.0, 1.0, 2),
            Err(Error::NonPositiveStepSize)
        ));
    }

    #[test]
    fn eps_range() {
        Apa::new(0.5, 1.0, 2).unwrap();

        assert!(matches!(
            Apa::new(0.5, 0.0, 2),
            Err(Error::NonPositiveEpsilon)
        ));
        assert!(matches!(
            Apa::new(0.5, -1.0, 2),
            Err(Error::NonPositiveEpsilon)
        ));
    }

    #[test]
    fn projection_order_range() {
        Apa::new(0.5, 1.0, 2).unwrap();

        assert!(matches!(
            Apa::new(0.5, 1.0, 0),
            Err(Error::InvalidProjectionOrder)
        ));
    }
}
