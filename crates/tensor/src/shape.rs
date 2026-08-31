#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape<const DIM: usize> {
    dims: [usize; DIM],
}

impl<const DIM: usize> Shape<DIM> {
    pub const fn new(dims: [usize; DIM]) -> Self {
        Self { dims }
    }

    pub const fn shape(&self) -> [usize; DIM] {
        self.dims
    }

    pub fn checked_numel(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
    }

    pub fn numel(&self) -> usize {
        self.checked_numel().expect("shape element count overflow")
    }
}

impl<const DIM: usize> From<[usize; DIM]> for Shape<DIM> {
    fn from(value: [usize; DIM]) -> Self {
        Self::new(value)
    }
}
