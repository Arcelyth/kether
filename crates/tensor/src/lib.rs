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
