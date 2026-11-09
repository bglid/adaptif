#![allow(clippy::unwrap_used, reason = "Only used in tests")]

use crate::types::Float;

// TODO: see if we can replace these three float comparison fns with a dev dependency
pub fn approx_equal<F>(a: F, b: F, eps: F) -> bool
where
    F: Float,
{
    (a - b).abs() < eps
}

pub fn approx_equal_iter<'a, 'b, I, J, F>(a: I, b: J, eps: F) -> bool
where
    I: Iterator<Item = &'a F>,
    J: Iterator<Item = &'b F>,
    F: Float + 'a + 'b,
{
    a.zip(b).all(|(x, y)| approx_equal(*x, *y, eps))
}

pub fn all_approx_equal<'a, 'b, I, J, F>(a: I, b: J) -> bool
where
    I: ExactSizeIterator<Item = &'a F>,
    J: ExactSizeIterator<Item = &'b F>,
    F: Float + 'a + 'b,
{
    if a.len() == b.len() {
        approx_equal_iter(a, b, F::from(1e-6).unwrap())
    } else {
        false
    }
}
