use crate::storage::Storage;
use crate::operators::Operator;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

fn next_graph_id() -> u64 {
    NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug)]
struct TapeNode<T> {
    inputs: Vec<Option<usize>>,
    input_values: Vec<Storage<T>>,
    value: Storage<T>,
    grad: Option<Vec<T>>,
    op: Option<Box<dyn Operator<T>>>,
}

/// An append-only reverse-mode graph.
///
/// Values use reference-counted immutable storage, while gradient buffers are
/// allocated lazily on the first backward pass and reused by later passes.
#[derive(Debug)]
pub struct Tape<T> {
    nodes: Vec<TapeNode<T>>,
    graph_id: u64,
    reachable: Vec<bool>,
    stack: Vec<usize>,
}

impl<T> Default for Tape<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            graph_id: next_graph_id(),
            reachable: Vec::new(),
            stack: Vec::new(),
        }
    }
}

impl<T> Tape<T>
where
    T: Copy + Default + From<u8> + 'static,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn graph_id(&self) -> u64 {
        self.graph_id
    }

    /// Returns the number of recorded leaf and operation nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the tape contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Removes all graph nodes, values, and gradient buffers.
    ///
    /// This invalidates tensors attached to the old graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.graph_id = next_graph_id();
    }

    /// Registers a differentiable leaf and returns its node identifier.
    ///
    /// Most callers should use Tensor::new instead.
    pub fn register_leaf(&mut self, data: Storage<T>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TapeNode {
            inputs: Vec::new(),
            input_values: Vec::new(),
            value: data,
            grad: None,
            op: None,
        });
        id
    }

    pub(crate) fn apply(
        &mut self,
        op: Box<dyn Operator<T>>,
        inputs: Vec<(Option<usize>, Storage<T>)>,
        value: Storage<T>,
    ) -> usize {
        let id = self.nodes.len();
        let (inputs, input_values) = inputs.into_iter().unzip();
        self.nodes.push(TapeNode {
            inputs,
            input_values,
            value,
            grad: None,
            op: Some(op),
        });
        id
    }

        /// Backpropagates a gradient of one through every element of the root.
    ///
    /// For a non-scalar root this differentiates the sum of its elements.
    /// Existing gradient buffers are reset and reused.
    ///
    /// # Panics
    ///
    /// Panics if root is not a node in this tape.
    pub fn backward(&mut self, root: usize) {
        let len = self
            .nodes
            .get(root)
            .expect("invalid root node id")
            .value
            .len();
        self.reset_gradients();
        self.ensure_grad(root).fill(T::from(1));
        debug_assert_eq!(self.nodes[root].value.len(), len);
        self.propagate(root);
    }

    /// Backpropagates an explicit vector-Jacobian seed.
    ///
    /// # Panics
    ///
    /// Panics if root is invalid or seed has the wrong length.
    pub fn backward_with_grad(&mut self, root: usize, seed: &[T]) {
        let root_len = self
            .nodes
            .get(root)
            .expect("invalid root node id")
            .value
            .len();
        assert_eq!(seed.len(), root_len, "root gradient size mismatch");

        self.reset_gradients();
        self.ensure_grad(root).copy_from_slice(seed);
        self.propagate(root);
    }

    fn reset_gradients(&mut self) {
        for node in &mut self.nodes {
            if let Some(grad) = &mut node.grad {
                grad.fill(T::default());
            }
        }
    }

    fn propagate(&mut self, root: usize) {
        self.reachable.resize(self.nodes.len(), false);
        self.reachable.fill(false);
        self.stack.clear();
        self.stack.push(root);
        while let Some(id) = self.stack.pop() {
            if self.reachable[id] {
                continue;
            }
            self.reachable[id] = true;
            for &input in &self.nodes[id].inputs {
                if let Some(parent) = input {
                    self.stack.push(parent);
                }
            }
        }

        for id in (0..=root).rev() {
            if !self.reachable[id] {
                continue;
            }
            let (parents, current_and_later) = self.nodes.split_at_mut(id);
            let current = &current_and_later[0];
            let Some(op) = current.op.as_ref() else {
                continue;
            };
            let Some(grad_output) = current.grad.as_deref() else {
                continue;
            };

            for (input_index, input) in current.inputs.iter().enumerate() {
                let Some(parent_id) = input else { continue };
                let parent = &mut parents[*parent_id];
                let grad = parent
                    .grad
                    .get_or_insert_with(|| vec![T::default(); parent.value.len()]);
                op.backward_input(input_index, grad_output, &current.input_values, grad);
            }
        }
    }

    fn ensure_grad(&mut self, id: usize) -> &mut [T] {
        let node = &mut self.nodes[id];
        node.grad
            .get_or_insert_with(|| vec![T::default(); node.value.len()])
    }

    /// Returns a node's gradient if one has been computed.
    ///
    /// Invalid identifiers and nodes not involved in backward return None.
    pub fn grad(&self, node: usize) -> Option<&[T]> {
        self.nodes.get(node)?.grad.as_deref()
    }

    /// Returns a node's computed gradient.
    ///
    /// # Panics
    ///
    /// Panics if the node is invalid or has no computed gradient.
    pub fn get_grad(&self, node: usize) -> &[T] {
        self.grad(node)
            .expect("gradient unavailable; call backward first")
    }
}
