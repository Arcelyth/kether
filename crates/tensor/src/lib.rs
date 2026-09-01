#![warn(missing_docs)]

//! A tensor library with tape-based autograd. 
//!
//! Tensor values are immutable and shared with the tape. Graph metadata is
//! append-only, gradient buffers are lazy and reusable, and operators
//! accumulate directly into their inputs without allocating backward results.
//!
//! # Example
//!
//! ```
//! use kether_tensor::{Tape, Tensor};
//! let mut tape = Tape::<f32>::new();
//! let x = Tensor::new(&[2.0, 3.0], [2], true, &mut tape);
//! let c = Tensor::new(&[4.0, 5.0], [2], false, &mut tape);
//! let y = x.mul(&c, &mut tape);
//! y.backward(&mut tape);
//! assert_eq!(x.grad(&tape), Some(&[4.0, 5.0][..]));
//! ```

pub mod operators;
pub mod shape;
pub mod storage;
pub mod tape;
pub mod tensor;
pub mod macros;

pub use operators::{Operator, AddOp, MulOp};
pub use shape::Shape;
pub use storage::Storage;
pub use tape::Tape;
pub use tensor::{Tensor, apply};
