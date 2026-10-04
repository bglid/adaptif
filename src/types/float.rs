use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

// The Float trait from num_traits doesn't require the traits below, which means generic
// Float types don't permit certain operations (e.g. `a += b`, `x.iter().sum()`).
// To keep declaring generics terse, we export this wrapper trait instead.
pub trait Float: num_traits::Float + Sum + AddAssign + SubAssign + MulAssign + DivAssign {}
impl Float for f32 {}
impl Float for f64 {}
