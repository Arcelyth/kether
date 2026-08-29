#[derive(Debug)]
pub struct Shape<const DIM: usize> {
    shape: [usize; DIM],
}

impl<const DIM: usize> Shape<DIM> {
    pub fn new(shape: [usize; DIM]) -> Self {
        Self { shape }
    }

    pub fn shape(&self) -> [usize; DIM] {
        self.shape
    }
}
