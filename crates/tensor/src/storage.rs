use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug)]
pub struct Storage<T> {
    data: Vec<T>,
}

impl<T> Storage<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    fn binop(self, other: Self, op: fn(T, T) -> T) -> Vec<T> {
        self.data
            .into_iter()
            .zip(other.data)
            .map(|(x, y)| op(x, y))
            .collect()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }
}

impl<T> Add for Storage<T>
where
    T: Add<Output = T>,
{
    type Output = Storage<T>;

    fn add(self, other: Self) -> Self::Output {
        Self {
            data: self.binop(other, T::add),
        }
    }
}

impl<T> Sub for Storage<T>
where
    T: Sub<Output = T>,
{
    type Output = Storage<T>;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            data: self.binop(other, T::sub),
        }
    }
}

impl<T> Mul for Storage<T>
where
    T: Mul<Output = T>,
{
    type Output = Storage<T>;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            data: self.binop(other, T::mul),
        }
    }
}

impl<T> Div for Storage<T>
where
    T: Div<Output = T>,
{
    type Output = Storage<T>;

    fn div(self, other: Self) -> Self::Output {
        Self {
            data: self.binop(other, T::div),
        }
    }
}

impl<T> Neg for Storage<T>
where
    T: Neg<Output = T>,
{
    type Output = Storage<T>;

    fn neg(self) -> Self::Output {
        Self {
            data: self.data.into_iter().map(|x| -x).collect(),
        }
    }
}
