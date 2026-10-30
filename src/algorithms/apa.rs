use num_traits::Zero as _; // TODO: Check w Elias why this needs to be imported
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
    /// Preallocated capacity for inner Gram matrix in APA update.
    gram_matrix: Matrix<F>,
    /// Augmented matrix used in Gauss Jordan algorithm for matrix inversion.
    augmented_matrix: Matrix<F>,
    /// Preallocated vector matrix to hold result of linear algebra solver.
    solver_result: Matrix<F>,
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
        if self.input_history.shape().cols != *window_size {
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

        self.input_history.update(&noise_window);

        // recalculating the entire error vector with prior to weight update
        self.error_vector
            .update(&self.desired_history, &self.input_history, weights);

        self.input_history
            .matmul(&self.input_history.transpose(), &mut self.gram_matrix)
            .expect("matrix dims must match");
        self.gram_matrix.add_eps(self.eps);

        self.gram_matrix
            .inverse(&mut self.augmented_matrix)
            .expect("Issue with gauss jordan matrix inversion");

        self.gram_matrix
            .matmul(&self.error_vector, &mut self.solver_result)
            .expect("matrix x vector dims must match");

        // TODO: Generalize the matrix vector product to reuse it here too
        // $w_{n+1} = ``w_n`` + \mu * X(n)(X(n)^{T}X(n) + \epsilon I)^{-1} e(n)$
        #[allow(
            clippy::indexing_slicing,
            reason = "Bounds are defined by projection order"
        )]
        for (j, w) in weights.iter_mut().enumerate() {
            let mut update = F::zero();

            for i in 0..self.input_history.shape().rows {
                let x = *self.input_history.get_2d(i, j).expect("matrix bounds");

                update += x * self.solver_result[i];
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
            gram_matrix: Matrix::new(Shape {
                rows: projection_order,
                cols: projection_order,
            }),
            augmented_matrix: Matrix::new(Shape {
                rows: projection_order,
                cols: 2 * projection_order,
            }),
            solver_result: Matrix::new(Shape {
                rows: projection_order,
                cols: 1,
            }),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L x window size dimensional matrix of previous input vectors, where L = projection order.
struct InputHistory<F: Float> {
    /// Flattened matrix to store the L x Window size matrix.
    inner: SampleBuffer<F>, // NOTE: doing flattened matrix since window size is known to avoid Vec<Vec<...>>
    /// Shape of the input history. Has dimensions L x window size.
    shape: Shape,
}

impl<F: Float> InputHistory<F> {
    pub fn new(projection_order: usize, window_size: WindowSize) -> Self {
        let capacity = NonZero::new(projection_order * (*window_size))
            .expect("Projection order and WindowSize should ensure that capacity is > 0.");
        InputHistory {
            inner: SampleBuffer::new(capacity),
            shape: Shape {
                rows: projection_order,
                cols: *window_size,
            },
        }
    }

    /// Updates the input history vector used in the APA weight update.
    pub fn update(&mut self, noise_ref: &NoiseWindow<F>) {
        for sample in noise_ref.iter() {
            self.push(sample);
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

impl<F: Float> MatrixOps<'_> for InputHistory<F> {
    type Item = F;
    fn shape(&self) -> Shape {
        self.shape
    }
    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return None;
        }

        self.inner
            .get(row * (self.shape.cols) + (self.shape.cols) - 1 - col)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L-dimensional vector of the desired output history, where L = projection order.
struct DesiredHistory<F: Float> {
    inner: SampleBuffer<F>,
}
impl<F: Float> DesiredHistory<F> {
    pub fn new(projection_order: usize) -> Self {
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

// NOTE: This is will switch once we have matrix trait
pub trait MatrixOps<'a> {
    type Item: Float;

    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item>;
    fn shape(&self) -> Shape;

    fn transpose(&'a self) -> Transpose<'a, Self>
    where
        Self: Sized,
    {
        Transpose { matrix: self }
    }

    fn matmul<'rhs, M>(&self, rhs: &M, out: &mut [Self::Item]) -> Option<()>
    where
        M: MatrixOps<'rhs, Item = Self::Item>,
    {
        if self.shape().cols != rhs.shape().rows {
            return None;
        }

        if out.len() != self.shape().rows * rhs.shape().cols {
            return None;
        }
        // need to fill the output matrix with zeros since this'll be used across APA updates
        out.fill(Self::Item::zero());

        let (n_rows, inner, n_cols) = (self.shape().rows, self.shape().cols, rhs.shape().cols);

        #[allow(clippy::indexing_slicing, reason = "Bounds are checked before...")]
        for i in 0..n_rows {
            for k in 0..inner {
                let lhs = *self.get_2d(i, k)?;

                for j in 0..n_cols {
                    out[i * n_cols + j] += lhs * (*rhs.get_2d(k, j)?);
                }
            }
        }
        Some(())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L-dimensional vector of the newly calculated error vectory, where L = projection order.
struct Matrix<F: Float> {
    inner: Box<[F]>,
    shape: Shape,
}
impl<F: Float> Matrix<F> {
    pub fn new(shape: Shape) -> Self {
        Matrix {
            inner: vec![F::zero(); shape.rows * shape.cols].into_boxed_slice(),
            shape,
        }
    }

    // TODO: DOUBLE CHECK THIS ALLOW
    #[allow(
        clippy::indexing_slicing,
        reason = "Bounds are defined by projection order"
    )]
    /// Gauss-Jordan elimination algorithm for getting the inverse of a matrix.
    /// Uses row reduction to compute the inverse matrix.
    /// The matrix isn't invertible if it can't be reduced to the identity matrix.
    pub fn inverse(&mut self, augmented_matrix: &mut [F]) -> Option<()> {
        // Augmented Identity matrix [A I] has the shape of n * 2n [Augmented matrix x identity matrix]
        let n = self.shape.rows;
        let width = 2 * n;

        // Algorithm requres these shape bounds of Matrix A = (n x n) & Matrix AI = (n * 2n)
        if self.shape.cols != n || augmented_matrix.len() != n * width {
            return None;
        }

        // NOTE: This down to line 336, we might be able to simplify, since the left side of AI is the matrix we are inverting
        // We need the AI matrix to be fresh every time we call update step since
        // we don't want the augmented matrix data from previous calls.
        augmented_matrix.fill(F::zero());

        // The first step is to copy the values of the matrix we are inverting to left (n x n)
        for i in 0..n {
            for j in 0..n {
                augmented_matrix[i * width + j] = *self.get_2d(i, j)?;
            }
            // In the right half of the AI matrix, we 'augment' by creating a (n x n) identity matrix
            augmented_matrix[i * width + n + i] = F::one();
        }

        // transforming into the `reduced row echelon form`
        // This step divdes row by the leading entry so that the leading entry equals 1
        for i in 0..n {
            let scale = augmented_matrix[i * width + i];
            for j in 0..width {
                augmented_matrix[i * width + j] /= scale;
            }

            // We want 0s below the first pivot (leading entry)
            // Eliminate non-diagonal elements in the current column
            for j in 0..n {
                if j != i {
                    // We get the row reduction amounts here to bring the values under the leading entry to 0
                    let elimination_factor = augmented_matrix[j * width + i];
                    for k in 0..width {
                        let leading_entry = augmented_matrix[i * width + k];
                        augmented_matrix[j * width + k] -= elimination_factor * leading_entry;
                    }
                }
            }
        }

        // Now for the purposes of matrix inversion, we write the inverted matrix,
        // which exists in the right-hand former identity side, to our matrix which
        // we wish to invert.
        for i in 0..n {
            for j in 0..n {
                self[i * n + j] = augmented_matrix[i * width + n + j];
            }
        }
        Some(())
    }
    #[allow(
        clippy::indexing_slicing,
        reason = "Bounds are defined by projection order"
    )]
    fn add_eps(&mut self, eps: F) {
        let (n_rows, n_cols) = (self.shape.rows, self.shape.cols);
        for i in 0..n_rows {
            self[i * n_cols + i] += eps;
        }
    }
}

impl<F: Float> Deref for Matrix<F> {
    type Target = Box<[F]>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<F: Float> DerefMut for Matrix<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<F: Float> MatrixOps<'_> for Matrix<F> {
    type Item = F;
    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item> {
        if row >= self.shape.rows || col >= self.shape.cols {
            return None;
        }
        self.inner.get(row * self.shape().cols + col)
    }
    fn shape(&self) -> Shape {
        self.shape
    }
}

#[derive(Debug, Clone)]
pub struct Transpose<'a, M>
where
    M: MatrixOps<'a>,
{
    matrix: &'a M,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

impl<'a, M: MatrixOps<'a>> MatrixOps<'a> for Transpose<'a, M> {
    type Item = M::Item;
    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item> {
        self.matrix.get_2d(col, row)
    }

    fn shape(&self) -> Shape {
        Shape {
            rows: self.matrix.shape().cols,
            cols: self.matrix.shape().rows,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// L-dimensional vector of the newly calculated error vectory, where L = projection order.
struct ErrorVector<F: Float> {
    inner: Matrix<F>,
}
impl<F: Float> ErrorVector<F> {
    pub fn new(projection_order: usize) -> Self {
        ErrorVector {
            inner: Matrix::new(Shape {
                rows: projection_order,
                cols: 1,
            }),
        }
    }
    /// Rebuilds the entire ``ErrorVector`` from scratch using new weight updates.
    pub fn update(
        &mut self,
        desired_history: &DesiredHistory<F>,
        input_history: &InputHistory<F>,
        weights: &FilterWeights<F>,
    ) {
        #[allow(
            clippy::indexing_slicing,
            reason = "Bounds are defined by projection order"
        )]
        for i in 0..self.inner.shape().rows {
            // desired_history.len() == projection_order, so index (i) is always <= desired_history.len()"
            let desired_sample = desired_history[i];

            // prediction using flattened input history matrix by weights
            // Since flattened, window size boundaries are essentially new rows
            let predicted_sample = weights.iter().enumerate().fold(F::zero(), |acc, (j, &w)| {
                let x = *input_history.get_2d(i, j).expect("Input history bounds");
                acc + w * x
            });
            self.inner[i] = desired_sample - predicted_sample;
        }
    }
}

impl<F: Float> MatrixOps<'_> for ErrorVector<F> {
    type Item = F;
    fn get_2d(&self, row: usize, col: usize) -> Option<&Self::Item> {
        self.inner.get_2d(row, col)
    }
    fn shape(&self) -> Shape {
        self.inner.shape()
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
