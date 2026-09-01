use kether_tensor::{Tape, Tensor};

fn main() {
    let mut tape = Tape::<f32>::new();

    let x = Tensor::new(&[2.0, 3.0], [2], true, &mut tape);
    let y = Tensor::new(&[4.0, 5.0], [2], true, &mut tape);
    let bias = Tensor::new(&[1.0, 1.0], [2], false, &mut tape);

    // z = x * y + bias
    let z = x.mul(&y, &mut tape).add(&bias, &mut tape);
    println!("z         = {:?}", z.data());

    z.backward(&mut tape);
    println!("dz/dx     = {:?}", x.grad(&tape).expect("x gradient"));
    println!("dz/dy     = {:?}", y.grad(&tape).expect("y gradient"));
    println!("dz/dbias  = {:?}", bias.grad(&tape));

    z.backward_with_grad(&mut tape, &[2.0, 3.0]);
    println!("vjp x     = {:?}", x.grad(&tape).expect("x VJP"));
    println!("vjp y     = {:?}", y.grad(&tape).expect("y VJP"));
}
