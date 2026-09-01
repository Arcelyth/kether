# Kether

Currently includes: 
- A tensor library with tape-based autograd. 

## Example

```
use kether_tensor::{Tape, Tensor};
let mut tape = Tape::<f32>::new();
let x = Tensor::new(&[2.0, 3.0], [2], true, &mut tape);
let c = Tensor::new(&[4.0, 5.0], [2], false, &mut tape);
let y = x.mul(&c, &mut tape);
y.backward(&mut tape);
assert_eq!(x.grad(&tape), Some(&[4.0, 5.0][..]));
```
