extern crate pairing;

fn main() {
    let mut b = 0u64;
    let a = pairing::sbb(0, 0, &mut b);
    println!("{}", a);
}
