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
    projection_order: usize,
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
        let window_size = weights.window_size();
        if self.input_history.window_size != window_size {
            self.input_history = InputHistory::new(self.projection_order, window_size);
        }

        let desired_output = (*error)
            + weights
                .iter()
                .copied()
                .zip(noise_window.iter())
                .map(|(w, x)| w * x)
                .sum();
        self.desired_history.push(desired_output);

        self.update_input_history(&noise_window);

        // recalculating the entire error vector with prior to weight update
        self.recalculate_error_vector(weights);

        let inner_matrix = self.matrix_matrix_product(*weights.window_size());

        let inverted_matrix = gauss_jordan(&inner_matrix, self.projection_order);

        let matrix_error_product = self.matrix_vector_product(&inverted_matrix);

        // TODO: Generalize the matrix vector product to reuse it here too
        // $w_{n+1} = ``w_n`` + \mu * X(n)(X(n)^{T}X(n) + \epsilon I)^{-1} e(n)$
        for (j, w) in weights.iter_mut().enumerate() {
            let mut update = F::zero();

            for i in 0..self.projection_order {
                let x = *self
                    .input_history
                    .get(i * (*window_size) + (*window_size) - 1 - j)
                    .expect("matrix bounds");

                update += x * (*matrix_error_product.get(i).expect("matrix bounds"));
            }
            *w += self.mu * update;
        }
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
        for sample in noise_ref.iter() {
            self.input_history.push(sample);
        }
    }

    /// Rebuilds the entire ``ErrorVector`` from scratch using new weight updates.
    fn recalculate_error_vector(&mut self, weights: &FilterWeights<F>) {
        // have to get oldest first because push inserts at front
        #[allow(clippy::missing_panics_doc, reason = "failure independent of fn input")]
        for i in (0..self.projection_order).rev() {
            let desired_sample = *self.desired_history.get(i).expect(
                "
                desired sample get should only fail if index >= desired_history.len().
                desired_history.len() == projection_order, so index (i) is always <= desired_history.len(),",
            );

            // prediction using flattened input history matrix by weights
            // Since flattened, window size boundaries are essentially new rows
            let predicted_sample = weights.iter().enumerate().fold(F::zero(), |acc, (j, &w)| {
                let x = *self
                    .input_history
                    .get(i * (*weights.window_size()) + (*weights.window_size()) - 1 - j)
                    .expect(""); // TODO: expect message here check
                acc + w * x
            });
            self.error_vector.push(desired_sample - predicted_sample);
        }
    }

    // TODO: Break this and following function into non-methods for reusable linalg utils

    /// Returns the inner matrix prior to inversion.
    ///
    /// Calculates the Dot Product of two matricies and adds an epsilon value along the identity
    /// matrix. ``n`` is the window size.
    fn matrix_matrix_product(&self, window: usize) -> SampleBuffer<F> {
        let mut dot_product = SampleBuffer::new(
            NonZero::new(self.projection_order * self.projection_order)
                .expect("projection order ensures that capacity is > 0"),
        );

        // a little fp to make indexing sane to fn look at
        // Similar to what is done above in ``recalculate_error_vector()``, as well as everywhere else...
        // so may be worth making a function
        let x = |i: usize, j: usize| -> F {
            *self
                .input_history
                .get(i * window + window - 1 - j)
                .expect("internal error with matrix bounds.")
        };

        // push() adds to the front so doing rev()
        for i in (0..self.projection_order).rev() {
            for j in (0..self.projection_order).rev() {
                // getting dot product, where k is the index for this row by col
                let product = (0..window).map(|k| x(i, k) * x(j, k)).sum::<F>();
                let regularization = if i == j { self.eps } else { F::zero() };
                dot_product.push(product + regularization);
            }
        }
        dot_product
    }

    fn matrix_vector_product(&self, matrix: &SampleBuffer<F>) -> SampleBuffer<F> {
        let mut dot_product = SampleBuffer::new(
            NonZero::new(self.projection_order)
                .expect("projection order ensures that capacity is > 0"),
        );

        // push() adds to the front so doing rev()
        for i in (0..self.projection_order).rev() {
            let mut product = F::zero();

            for j in 0..self.projection_order {
                product += (*matrix
                    .get(i * self.projection_order + j)
                    .expect("matrix bounds"))
                    * (*self.error_vector.get(j).expect("vector bounds"));
            }
            dot_product.push(product);
        }
        dot_product
    }
}

// TODO: Yeah this should go in it's own file...
// From what I see online, doing Xn^-1 requires a "GaussJordan" elimination (or a linalg solver)
// this is the GaussJordan approach pulled from this: https://gist.github.com/barbarbar338/b66e2a404abb3a7f7c4d542204a3626a
// The simplest example I could find... lol
// Also, https://en.wikipedia.org/wiki/Gaussian_elimination
fn gauss_jordan<F>(matrix: &SampleBuffer<F>, n: usize) -> SampleBuffer<F>
where
    F: Float,
{
    // NOTE: If we keep this monstrosity then I may do a closure or secondary function for the weird flat indexing here
    // ... or to make the .get() a bit easier

    // Augmented Identity matrix [A I]
    // According to wikipedia, this has the shape of n * 2n [Augmented matrix x identity matrix]
    let width = 2 * n;
    let mut augmented_flat_mat =
        SampleBuffer::new(NonZero::new(n * width).expect("n must be positive"));

    for i in 0..n {
        for j in 0..n {
            if let Some(val) = matrix.get(i * n + j) {
                *augmented_flat_mat.get_mut(i * width + j).expect(
                    "Internal error with matrix bounds, range is set by projection order",
                ) = *val;
            }
        }
        // identity matrix part, again flat so n * i is like jumping to next row
        *augmented_flat_mat
            .get_mut(i * width + n + i)
            .expect("Internal error with matrix bounds, range is set by projection order") =
            F::one();
    }

    // TODO: improve expect() all along here too, there's a lot but I want to see if this works first
    for i in 0..n {
        let scale = *augmented_flat_mat
            .get(i * width + i)
            .expect("Matrix bounds erro");
        for j in 0..(2 * n) {
            *augmented_flat_mat
                .get_mut(i * width + j)
                .expect("Matrix bounds error") /= scale;
        }

        // Eliminate non-diagonal elements in the current column
        for j in 0..n {
            if j != i {
                let elimination_factor = *augmented_flat_mat
                    .get(j * width + i)
                    .expect("Matrix bounds error");
                for k in 0..(2 * n) {
                    let leading_entry = *augmented_flat_mat
                        .get_mut(i * width + k)
                        .expect("Matrix bounds error");
                    *augmented_flat_mat
                        .get_mut(j * width + k)
                        .expect("Matrix bounds error") -= elimination_factor * leading_entry;
                }
            }
        }
    }

    // Now you apparently take the inverse matrix out of the augmented matrix
    let mut inverted_matrix = SampleBuffer::new(NonZero::new(n * n).expect("n must be positive"));
    for i in 0..n {
        for j in 0..n {
            *inverted_matrix
                .get_mut(i * n + j)
                .expect("Matrix bounds error") = *augmented_flat_mat
                .get(i * width + n + j)
                .expect("Matrix bounds error");
        }
    }
    inverted_matrix
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
        let block_size = BlockSize::new(1).unwrap();

        let mut apa = Apa::new(0.5, 1.0, 2).unwrap();
        let e_n = {
            let mut e_n = BlockError::new(block_size);
            e_n.push(OutputSample(2.0));
            e_n
        };
        let x_n = block_noise_buffer_from(&[1.0, 0.0], window_size, block_size);
        let expected = [0.5, 0.0];
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
