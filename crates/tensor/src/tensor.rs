use crate::shape::Shape;
use crate::storage::Storage;
use crate::tape::Tape;
use crate::operators::Operator;

/// An immutable, fixed-rank tensor optionally attached to an autograd tape.
///
/// Cloning shares element storage. Gradients live in the associated Tape.
#[derive(Debug, Clone)]
pub struct Tensor<T, const DIM: usize> {
    data: Storage<T>,
    shape: Shape<DIM>,
    node_id: Option<usize>,
    graph_id: Option<u64>,
}

impl<T, const DIM: usize> Tensor<T, DIM>
where
    T: Copy + Default + From<u8> + 'static,
{

    /// Creates a tensor and optionally registers it as a differentiable leaf.
    ///
    /// # Panics
    ///
    /// Panics if the data length differs from the shape's element count.
    pub fn new(data: &[T], shape: [usize; DIM], requires_grad: bool, tape: &mut Tape<T>) -> Self {
        let shape = Shape::new(shape);
        assert_eq!(data.len(), shape.numel(), "data size does not match shape");
        let data = Storage::new(data.to_vec());
        let node_id = requires_grad.then(|| tape.register_leaf(data.clone()));
        let graph_id = requires_grad.then(|| tape.graph_id());
        Self {
            data,
            shape,
            node_id,
            graph_id,
        }
    }

    /// Borrows elements in contiguous row-major order.
    pub fn data(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns the tensor's shape.
    pub const fn shape(&self) -> Shape<DIM> {
        self.shape
    }

    /// Returns whether the tensor participates in automatic differentiation.
    pub const fn requires_grad(&self) -> bool {
        self.node_id.is_some()
    }

    /// Returns the tape-local node identifier, if differentiable.
    pub const fn node_id(&self) -> Option<usize> {
        self.node_id
    }

    /// Borrows this tensor's gradient, if it has been computed.
    pub fn grad<'a>(&self, tape: &'a Tape<T>) -> Option<&'a [T]> {
        self.node_id.and_then(|id| tape.grad(id))
    }

    /// Backpropagates an all-ones seed from this tensor.
    ///
    /// # Panics
    ///
    /// Panics if this is a constant or belongs to another or cleared tape.
    pub fn backward(&self, tape: &mut Tape<T>) {
        assert_eq!(
            self.graph_id,
            Some(tape.graph_id()),
            "tensor belongs to a different or cleared tape"
        );
        tape.backward(self.node_id.expect("tensor does not require gradients"));
    }

    /// Backpropagates an explicit vector-Jacobian seed.
    ///
    /// # Panics
    ///
    /// Panics for an invalid tape association or a seed of the wrong length.
    pub fn backward_with_grad(&self, tape: &mut Tape<T>, seed: &[T]) {
        assert_eq!(
            self.graph_id,
            Some(tape.graph_id()),
            "tensor belongs to a different or cleared tape"
        );
        tape.backward_with_grad(
            self.node_id.expect("tensor does not require gradients"),
            seed,
        );
    }
}

/// Applies a same-shape operation. Constant inputs are supported and are not
/// assigned graph nodes; their values are retained only when needed by backward.
///
/// The output is recorded only if at least one input requires gradients.
///
/// # Panics
///
/// Panics for empty inputs, shape mismatches, mixed tape generations, or an
/// operator output with the wrong number of elements.
pub fn apply<T, const DIM: usize, O>(
    op: O,
    inputs: &[&Tensor<T, DIM>],
    tape: &mut Tape<T>,
) -> Tensor<T, DIM>
where
    T: Copy + Default + From<u8> + 'static,
    O: Operator<T> + 'static,
{
    let first = inputs
        .first()
        .expect("an operation needs at least one input");
    assert!(
        inputs.iter().all(|input| input.shape == first.shape),
        "input shape mismatch"
    );
    assert!(
        inputs
            .iter()
            .filter(|input| input.requires_grad())
            .all(|input| input.graph_id == Some(tape.graph_id())),
        "input tensor belongs to a different or cleared tape"
    );

    let input_data: Vec<Storage<T>> = inputs.iter().map(|input| input.data.clone()).collect();
    let output = Storage::new(op.forward(&input_data));
    assert_eq!(
        output.len(),
        first.shape.numel(),
        "operator returned wrong output size"
    );

    let node_id = inputs.iter().any(|input| input.requires_grad()).then(|| {
        let edges = inputs
            .iter()
            .zip(input_data)
            .map(|(input, data)| (input.node_id, data))
            .collect();
        tape.apply(Box::new(op), edges, output.clone())
    });
    let graph_id = node_id.map(|_| tape.graph_id());
    Tensor {
        data: output,
        shape: first.shape,
        node_id,
        graph_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_and_shared_subgraph_gradients_are_correct() {
        let mut tape = Tape::<f32>::new();
        let x = Tensor::new(&[2.0, 3.0], [2], true, &mut tape);
        let y = Tensor::new(&[4.0, 5.0], [2], true, &mut tape);
        let xy = x.mul(&y, &mut tape);
        let z = x.add(&xy, &mut tape);

        assert_eq!(z.data(), &[10.0, 18.0]);
        z.backward(&mut tape);
        assert_eq!(x.grad(&tape), Some(&[5.0, 6.0][..]));
        assert_eq!(y.grad(&tape), Some(&[2.0, 3.0][..]));
    }

    #[test]
    fn constants_work_and_repeated_backward_does_not_accumulate_stale_grads() {
        let mut tape = Tape::<f32>::new();
        let x = Tensor::new(&[2.0, 3.0], [2], true, &mut tape);
        let c = Tensor::new(&[4.0, 5.0], [2], false, &mut tape);
        let y = x.mul(&c, &mut tape);
        y.backward_with_grad(&mut tape, &[2.0, 3.0]);
        assert_eq!(x.grad(&tape), Some(&[8.0, 15.0][..]));
        y.backward(&mut tape);
        assert_eq!(x.grad(&tape), Some(&[4.0, 5.0][..]));
        assert_eq!(tape.len(), 2);
    }

    #[test]
    fn backward_only_visits_root_ancestors() {
        let mut tape = Tape::<f32>::new();
        let x = Tensor::new(&[2.0], [1], true, &mut tape);
        let y = x.mul(&x, &mut tape);
        let unrelated = Tensor::new(&[7.0], [1], true, &mut tape);
        y.backward(&mut tape);
        assert_eq!(x.grad(&tape), Some(&[4.0][..]));
        assert_eq!(unrelated.grad(&tape), None);
    }

    #[test]
    #[should_panic(expected = "different or cleared tape")]
    fn rejects_tensors_from_another_tape() {
        let mut first = Tape::<f32>::new();
        let mut second = Tape::<f32>::new();
        let x = Tensor::new(&[2.0], [1], true, &mut first);
        let y = Tensor::new(&[3.0], [1], true, &mut second);
        let _ = x.add(&y, &mut first);
    }

    #[test]
    #[should_panic(expected = "different or cleared tape")]
    fn clearing_a_tape_invalidates_existing_tensors() {
        let mut tape = Tape::<f32>::new();
        let x = Tensor::new(&[2.0], [1], true, &mut tape);
        tape.clear();
        x.backward(&mut tape);
    }

    #[test]
    fn relu_forward_and_backward_are_correct() {
        let mut tape = Tape::<f32>::new();
        let x = Tensor::new(&[-2.0, 0.0, 3.0], [3], true, &mut tape);

        let y = x.relu(&mut tape);

        assert_eq!(y.data(), &[0.0, 0.0, 3.0]);
        y.backward(&mut tape);
        assert_eq!(x.grad(&tape), Some(&[0.0, 0.0, 1.0][..]));
    }
}
