use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::register_op;
use crate::storage::Storage;

/// An operation recorded by the reverse-mode autograd tape.
pub trait Operator<T>: std::fmt::Debug + Send + Sync {
    /// Returns a stable, human-readable operation name.
    fn name(&self) -> &'static str;

    /// Computes the operation output from immutable input buffers.
    ///
    /// The returned vector must have the element count expected by apply.
    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T>;

    /// backward_input is called once per differentiable input and must add its
    /// contribution to grad_input. This avoids temporary nested gradient vectors.
    fn backward_input(
        &self,
        input: usize,
        grad_output: &[T],
        inputs: &[Storage<T>],
        grad_input: &mut [T],
    );
}

/// Addition operator.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddOp;

impl<T: Copy + Add<Output = T> + Send + Sync> Operator<T> for AddOp {
    fn name(&self) -> &'static str {
        "Add"
    }

    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T> {
        assert_eq!(inputs.len(), 2, "Add expects two inputs");
        assert_eq!(inputs[0].len(), inputs[1].len(), "Add size mismatch");
        inputs[0]
            .as_slice()
            .iter()
            .zip(inputs[1].as_slice())
            .map(|(&a, &b)| a + b)
            .collect()
    }

    fn backward_input(&self, input: usize, grad_output: &[T], _: &[Storage<T>], grad: &mut [T]) {
        assert!(input < 2, "Add has only two inputs");
        for (target, &source) in grad.iter_mut().zip(grad_output) {
            *target = *target + source;
        }
    }
}

/// Subtraction operator.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubOp;

impl<T> Operator<T> for SubOp
where
    T: Copy + Add<Output = T> + Neg<Output = T> + Sub<Output = T> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "Sub"
    }

    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T> {
        assert_eq!(inputs.len(), 2, "Sub expects two inputs");
        assert_eq!(inputs[0].len(), inputs[1].len(), "Sub size mismatch");
        inputs[0]
            .as_slice()
            .iter()
            .zip(inputs[1].as_slice())
            .map(|(&a, &b)| a - b)
            .collect()
    }

    fn backward_input(&self, input: usize, grad_output: &[T], _: &[Storage<T>], grad: &mut [T]) {
        match input {
            0 => {
                for (target, &upstream) in grad.iter_mut().zip(grad_output) {
                    *target = *target + upstream;
                }
            }
            1 => {
                for (target, &upstream) in grad.iter_mut().zip(grad_output) {
                    *target = *target + -upstream;
                }
            }
            _ => panic!("Sub has only two inputs"),
        }
    }
}

/// Multiplication operator.
#[derive(Debug, Clone, Copy, Default)]
pub struct MulOp;

impl<T> Operator<T> for MulOp
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "Mul"
    }

    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T> {
        assert_eq!(inputs.len(), 2, "Mul expects two inputs");
        assert_eq!(inputs[0].len(), inputs[1].len(), "Mul size mismatch");
        inputs[0]
            .as_slice()
            .iter()
            .zip(inputs[1].as_slice())
            .map(|(&a, &b)| a * b)
            .collect()
    }

    fn backward_input(
        &self,
        input: usize,
        grad_output: &[T],
        inputs: &[Storage<T>],
        grad: &mut [T],
    ) {
        let factor = match input {
            0 => inputs[1].as_slice(),
            1 => inputs[0].as_slice(),
            _ => panic!("Mul has only two inputs"),
        };
        for ((target, &upstream), &value) in grad.iter_mut().zip(grad_output).zip(factor) {
            *target = *target + upstream * value;
        }
    }
}

/// Division operator.
#[derive(Debug, Clone, Copy, Default)]
pub struct DivOp;

impl<T> Operator<T> for DivOp
where
    T: Copy + Add<Output = T> + Div<Output = T> + Mul<Output = T> + Neg<Output = T> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "Div"
    }

    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T> {
        assert_eq!(inputs.len(), 2, "Div expects two inputs");
        assert_eq!(inputs[0].len(), inputs[1].len(), "Div size mismatch");
        inputs[0]
            .as_slice()
            .iter()
            .zip(inputs[1].as_slice())
            .map(|(&a, &b)| a / b)
            .collect()
    }

    fn backward_input(
        &self,
        input: usize,
        grad_output: &[T],
        inputs: &[Storage<T>],
        grad: &mut [T],
    ) {
        let numerator = inputs[0].as_slice();
        let denominator = inputs[1].as_slice();
        match input {
            0 => {
                for ((target, &upstream), &denom) in
                    grad.iter_mut().zip(grad_output).zip(denominator)
                {
                    *target = *target + upstream / denom;
                }
            }
            1 => {
                for (((target, &upstream), &numer), &denom) in grad
                    .iter_mut()
                    .zip(grad_output)
                    .zip(numerator)
                    .zip(denominator)
                {
                    *target = *target + -(upstream * numer / (denom * denom));
                }
            }
            _ => panic!("Div has only two inputs"),
        }
    }
}

/// Negation operator.
#[derive(Debug, Clone, Copy, Default)]
pub struct NegOp;

impl<T> Operator<T> for NegOp
where
    T: Copy + Add<Output = T> + Neg<Output = T> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "Neg"
    }

    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T> {
        assert_eq!(inputs.len(), 1, "Neg expects one input");
        inputs[0].as_slice().iter().map(|&x| -x).collect()
    }

    fn backward_input(&self, input: usize, grad_output: &[T], _: &[Storage<T>], grad: &mut [T]) {
        assert_eq!(input, 0, "Neg has only one input");
        for (target, &upstream) in grad.iter_mut().zip(grad_output) {
            *target = *target + -upstream;
        }
    }
}

register_op!(
    ///
    mul(rhs) => MulOp,
    ///
    add(rhs) => AddOp,
    /// 
    sub(rhs) => SubOp,
    ///
    div(rhs) => DivOp,
    ///
    neg() => NegOp,
);
