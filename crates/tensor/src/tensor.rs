use crate::shape::Shape;
use crate::storage::Storage;

use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Arc, Mutex};

/// Tensor struct has `T` and `DIM` type parameters
/// for data type and dimensions.
#[derive(Debug)]
pub struct Tensor<T, const DIM: usize> {
    pub data: Storage<T>,
    pub shape: Shape<DIM>,
    pub grad: Option<Arc<Mutex<Tensor<T, DIM>>>>,
    pub do_grad: bool,
}

impl<T, const DIM: usize> Tensor<T, DIM> {
    pub fn new(data: Vec<T>, shape: [usize; DIM], do_grad: bool) -> Self {
        Self {
            data: Storage::new(data),
            shape: Shape::<DIM>::new(shape),
            grad: None,
            do_grad,
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data.data()
    }

    pub fn shape(&self) -> [usize; DIM] {
        self.shape.shape()
    }
}

impl<T, const DIM: usize> Add for Tensor<T, DIM>
where
    T: Add<Output = T>,
{
    type Output = Tensor<T, DIM>;

    fn add(self, other: Self) -> Self::Output {
        Self {
            data: self.data + other.data,
            shape: self.shape,
            grad: None,
            do_grad: false,
        }
    }
}

impl<T, const DIM: usize> Sub for Tensor<T, DIM>
where
    T: Sub<Output = T>,
{
    type Output = Tensor<T, DIM>;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            data: self.data - other.data,
            shape: self.shape,
            grad: None,
            do_grad: false,
        }
    }
}

impl<T, const DIM: usize> Mul for Tensor<T, DIM>
where
    T: Mul<Output = T>,
{
    type Output = Tensor<T, DIM>;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            data: self.data * other.data,
            shape: self.shape,
            grad: None,
            do_grad: false,
        }
    }
}

impl<T, const DIM: usize> Div for Tensor<T, DIM>
where
    T: Div<Output = T>,
{
    type Output = Tensor<T, DIM>;

    fn div(self, other: Self) -> Self::Output {
        Self {
            data: self.data / other.data,
            shape: self.shape,
            grad: None,
            do_grad: false,
        }
    }
}

impl<T, const DIM: usize> Neg for Tensor<T, DIM>
where
    T: Neg<Output = T>,
{
    type Output = Tensor<T, DIM>;

    fn neg(self) -> Self::Output {
        Self {
            data: -self.data,
            shape: self.shape,
            grad: None,
            do_grad: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add() {
        let a = Tensor::<i32, 1>::new(vec![1, 2, 3], [3], false);
        let b = Tensor::<i32, 1>::new(vec![4, 5, 6], [3], false);
        let c = Tensor::<i32, 1>::new(vec![6, -6, 6], [3], false);
        let d = Tensor::<i32, 1>::new(vec![3, 3, -3], [3], false);
        let c = -((a + b) * c / d);

        assert_eq!(c.data(), vec![-10, 14, 18]);
        assert_eq!(c.shape(), [3]);
        assert_eq!(c.do_grad, false);
    }
}
