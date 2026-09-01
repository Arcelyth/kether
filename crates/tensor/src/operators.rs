use std::ops::{Add, Mul};

use crate::storage::Storage;
use crate::register_op;

/// An operation recorded by the reverse-mode autograd tape.
///
/// backward_input is called once per differentiable input and must add its
/// contribution to grad_input. This avoids temporary nested gradient vectors.
pub trait Operator<T>: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn forward(&self, inputs: &[Storage<T>]) -> Vec<T>;
    fn backward_input(
        &self,
        input: usize,
        grad_output: &[T],
        inputs: &[Storage<T>],
        grad_input: &mut [T],
    );
}

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

register_op!(
    mul(rhs) => MulOp,
    add(rhs) => AddOp,
);
