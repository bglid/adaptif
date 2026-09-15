use std::error;
use std::fmt::Display;
use std::result;

pub type Result<T> = result::Result<T, Error>;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    EmptyInputArr,
    WindowSizeZero,
    BlockSizeZero,
    NoiseRefTooShort { input_len: usize, noise_len: usize },
    NonPositiveStepSize,
    NonPositiveEpsilon,
    IncorrectLambdaRange,
    NonPositiveDelta,
}
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyInputArr => write!(f, "empty input array."),
            Self::WindowSizeZero => write!(f, "window size must be greater than zero."),
            Self::BlockSizeZero => write!(f, "block size must be greater than zero."),
            Self::NoiseRefTooShort {
                input_len,
                noise_len,
            } => write!(
                f,
                "length of the noise reference ({noise_len}) must be equal to or greater than that of the input signal ({input_len})."
            ),
            Self::NonPositiveStepSize => write!(f, "step size (mu) must be greater than 0.0"),
            Self::NonPositiveEpsilon => write!(f, "epsilon (eps) must be greater than 0.0"),
            Self::IncorrectLambdaRange => {
                write!(f, "lambda, the forgetting factor, must be > 0.0 and <= 1.0")
            }
            Self::NonPositiveDelta => write!(f, "delta must be greater than 0.0"),
        }
    }
}
impl error::Error for Error {}
