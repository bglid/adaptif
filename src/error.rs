use std::error;
use std::fmt::Display;
use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    EmptyInputArr,
    NoiseRefTooShort { input_len: usize, noise_len: usize },
    NonPositiveStepSize,
    NonPositiveEpsilon,
}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyInputArr => write!(f, "empty input array."),
            Self::NoiseRefTooShort {
                input_len,
                noise_len,
            } => write!(
                f,
                "length of the noise reference ({noise_len}) must be equal to or greater than that of the input signal ({input_len})."
            ),
            Self::NonPositiveStepSize => write!(f, "step size (mu) must be greater than 0.0"),
            Self::NonPositiveEpsilon => write!(f, "epsilon (eps) must be greater than 0.0"),
        }
    }
}
impl error::Error for Error {}
