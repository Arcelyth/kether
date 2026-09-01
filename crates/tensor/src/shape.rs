/// The dimensions of a statically ranked tensor.
///
/// DIM is the number of axes. A shape may contain a zero-length axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape<const DIM: usize> {
    dims: [usize; DIM],
}

impl<const DIM: usize> Shape<DIM> {
    /// Creates a shape from its axis lengths.
    pub const fn new(dims: [usize; DIM]) -> Self {
        Self { dims }
    }

    /// Returns all axis lengths.
    pub const fn shape(&self) -> [usize; DIM] {
        self.dims
    }


    /// Returns the total element count, or None if multiplication overflows.
    pub fn checked_numel(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
    }

    /// Returns the total number of elements.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions' product does not fit in usize.
    pub fn numel(&self) -> usize {
        self.checked_numel().expect("shape element count overflow")
    }
}

impl<const DIM: usize> From<[usize; DIM]> for Shape<DIM> {
    fn from(value: [usize; DIM]) -> Self {
        Self::new(value)
    }
}
