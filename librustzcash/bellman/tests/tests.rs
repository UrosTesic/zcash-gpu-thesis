/*use pairing::{bls12_381::Bls12,
    CurveAffine,
    CurveProjective,
    Engine,
    PrimeField,
    Field,
    PrimeFieldRepr};

use multiexp::FullDensity;
use multicore::Worker;

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use std::mem::transmute;
use std::sync::Arc;
use std::time::Instant;

type G1Affine = <Bls12 as Engine>::G1Affine;
type G1Projective = <Bls12 as Engine>::G1;
type FrRepr = <<Bls12 as Engine>::Fr as PrimeField>::Repr;
type Fr = <Bls12 as Engine>::Fr;

fn load_data(num_elements: usize, points: &mut Vec<G1Affine>, exponents: &mut Vec<FrRepr>) {
    let path = "point_exp_pairs.txt";
    let file = File::open(path).unwrap();
    let reader = BufReader::new(&file);

    let mut projective_u64 = [0u64; 18];
    let mut exponent_u64 = [0u64; 4];

    let format_string = "G1 {{ x: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), y: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), z: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])) }} FrRepr([{d}, {d}, {d}, {d}])";
    for (num, line) in reader.lines().enumerate() {
        if num_elements == num {
            break;
        } else {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            projective_u64[0] = temp.0.unwrap();
            projective_u64[1] = temp.1.unwrap();
            projective_u64[2] = temp.2.unwrap();
            projective_u64[3] = temp.3.unwrap();
            projective_u64[4] = temp.4.unwrap();
            projective_u64[5] = temp.5.unwrap();
            projective_u64[6] = temp.6.unwrap();
            projective_u64[7] = temp.7.unwrap();
            projective_u64[8] = temp.8.unwrap();
            projective_u64[9] = temp.9.unwrap();
            projective_u64[10] = temp.10.unwrap();
            projective_u64[11] = temp.11.unwrap();
            projective_u64[12] = temp.12.unwrap();
            projective_u64[13] = temp.13.unwrap();
            projective_u64[14] = temp.14.unwrap();
            projective_u64[15] = temp.15.unwrap();
            projective_u64[16] = temp.16.unwrap();
            projective_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(projective_u64);
                let exp: FrRepr = transmute(exponent_u64);

                points.push(proj.into_affine());
                exponents.push(exp);
            }
        }
    }
}
#[test]
fn test_cpu_multiexp() {
    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let num_points = 131071;

    load_data(num_points, &mut points, &mut exponents);

    let v = Arc::new(points);
    let g = Arc::new(exponents);

    let pool = Worker::new();

    let now = Instant::now();
    let fast = multiexp(
        &pool,
        (g, 0),
        FullDensity,
        v
    ).wait().unwrap();
    let duration = now.elapsed();
    println!("{}.{:06}", duration.as_secs(), duration.subsec_micros());

    println!("{:?}", fast);

}*/