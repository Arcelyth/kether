use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

/// Cheaply cloneable, immutable tensor storage.
#[derive(Debug, Clone)]
pub struct Storage<T> {
    data: Arc<[T]>,
}

impl<T> Storage<T> {
    /// Moves a vector into shared immutable storage.
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Self { data: data.into() }
    }

    /// Wraps an existing shared slice without copying its elements.
    #[inline]
    pub fn from_arc(data: Arc<[T]>) -> Self {
        Self { data }
    }

    /// Borrows all stored elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Borrows all stored elements. This is an alias for as_slice.
    #[inline]
    pub fn data(&self) -> &[T] {
        self.as_slice()
    }

    /// Returns the number of stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true when no elements are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    fn zip_map(self, other: Self, op: impl Fn(T, T) -> T) -> Self
    where
        T: Copy,
    {
        assert_eq!(self.len(), other.len(), "storage size mismatch");
        Self::new(
            self.data
                .iter()
                .copied()
                .zip(other.data.iter().copied())
                .map(|(x, y)| op(x, y))
                .collect(),
        )
    }
}

impl<T: Copy + Add<Output = T>> Add for Storage<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.zip_map(other, T::add)
    }
}

impl<T: Copy + Sub<Output = T>> Sub for Storage<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.zip_map(other, T::sub)
    }
}

impl<T: Copy + Mul<Output = T>> Mul for Storage<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.zip_map(other, T::mul)
    }
}

impl<T: Copy + Div<Output = T>> Div for Storage<T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.zip_map(other, T::div)
    }
}

impl<T: Copy + Neg<Output = T>> Neg for Storage<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(self.data.iter().copied().map(T::neg).collect())
    }
}
