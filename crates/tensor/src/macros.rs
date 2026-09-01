/// Register an operation method for Tensor.
///
/// The operator must implement Default; the generated method constructs one
/// value and delegates graph construction and validation to tensor::apply.
#[macro_export]
macro_rules! register_op {
    (
        $(
            $(#[$meta:meta])*
            $name:ident ( $($arg:ident),* $(,)? ) => $op:path
        ),* $(,)?
    ) => {
        impl<T, const DIM: usize> $crate::tensor::Tensor<T, DIM>
        where
            T: Copy + Default + From<u8> + 'static,
        {
            $(
                $(#[$meta])*
                pub fn $name(
                    &self,
                    $($arg: &Self,)*
                    tape: &mut $crate::tape::Tape<T>,
                ) -> Self
                where
                    $op: $crate::operators::Operator<T> + Default + 'static,
                {
                    $crate::tensor::apply(<$op>::default(), &[self, $($arg),*], tape)
                }
            )*
        }
    };
}

