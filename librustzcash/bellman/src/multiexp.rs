use pairing::{
    CurveAffine,
    CurveProjective,
    Engine,
    PrimeField,
    Field,
    PrimeFieldRepr
};
use std::sync::Arc;
use std::io;
use bit_vec::{self, BitVec};
use std::iter;
use futures::{Future};
use super::multicore::Worker;

use super::SynthesisError;

/// An object that builds a source of bases.
pub trait SourceBuilder<G: CurveAffine>: Send + Sync + 'static + Clone {
    type Source: Source<G>;

    fn new(self) -> Self::Source;
}

/// A source of bases, like an iterator.
pub trait Source<G: CurveAffine> {
    /// Parses the element from the source. Fails if the point is at infinity.
    fn add_assign_mixed(&mut self, to: &mut <G as CurveAffine>::Projective) -> Result<(), SynthesisError>;

    /// Skips `amt` elements from the source, avoiding deserialization.
    fn skip(&mut self, amt: usize) -> Result<(), SynthesisError>;
}

impl<G: CurveAffine> SourceBuilder<G> for (Arc<Vec<G>>, usize) {
    type Source = (Arc<Vec<G>>, usize);

    fn new(self) -> (Arc<Vec<G>>, usize) {
        (self.0.clone(), self.1)
    }
}

impl<G: CurveAffine> Source<G> for (Arc<Vec<G>>, usize) {
    fn add_assign_mixed(&mut self, to: &mut <G as CurveAffine>::Projective) -> Result<(), SynthesisError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "expected more bases from source").into());
        }

        if self.0[self.1].is_zero() {
            return Err(SynthesisError::UnexpectedIdentity)
        }

        to.add_assign_mixed(&self.0[self.1]);

        self.1 += 1;

        Ok(())
    }

    fn skip(&mut self, amt: usize) -> Result<(), SynthesisError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "expected more bases from source").into());
        }

        self.1 += amt;

        Ok(())
    }
}

pub trait QueryDensity {
    /// Returns whether the base exists.
    type Iter: Iterator<Item=bool>;

    fn iter(self) -> Self::Iter;
    fn get_query_size(self) -> Option<usize>;
}

#[derive(Clone)]
pub struct FullDensity;

impl AsRef<FullDensity> for FullDensity {
    fn as_ref(&self) -> &FullDensity {
        self
    }
}

impl<'a> QueryDensity for &'a FullDensity {
    type Iter = iter::Repeat<bool>;

    fn iter(self) -> Self::Iter {
        iter::repeat(true)
    }

    fn get_query_size(self) -> Option<usize> {
        None
    }
}

pub struct DensityTracker {
    bv: BitVec,
    total_density: usize
}

impl<'a> QueryDensity for &'a DensityTracker {
    type Iter = bit_vec::Iter<'a>;

    fn iter(self) -> Self::Iter {
        self.bv.iter()
    }

    fn get_query_size(self) -> Option<usize> {
        Some(self.bv.len())
    }
}

impl DensityTracker {
    pub fn new() -> DensityTracker {
        DensityTracker {
            bv: BitVec::new(),
            total_density: 0
        }
    }

    pub fn add_element(&mut self) {
        self.bv.push(false);
    }

    pub fn inc(&mut self, idx: usize) {
        if !self.bv.get(idx).unwrap() {
            self.bv.set(idx, true);
            self.total_density += 1;
        }
    }

    pub fn get_total_density(&self) -> usize {
        self.total_density
    }
}

fn multiexp_inner<Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<<G::Engine as Engine>::Fr as PrimeField>::Repr>>,
    mut skip: u32,
    c: u32,
    handle_trivial: bool
) -> Box<Future<Item=<G as CurveAffine>::Projective, Error=SynthesisError>>
    where for<'a> &'a Q: QueryDensity,
          D: Send + Sync + 'static + Clone + AsRef<Q>,
          G: CurveAffine,
          S: SourceBuilder<G>
{
    // Perform this region of the multiexp
    let this = {
        let bases = bases.clone();
        let exponents = exponents.clone();
        let density_map = density_map.clone();

        pool.compute(move || {
            // Accumulate the result
            let mut acc = G::Projective::zero();

            // Build a source for the bases
            let mut bases = bases.new();

            // Create space for the buckets
            let mut buckets = vec![<G as CurveAffine>::Projective::zero(); (1 << c) - 1];

            let zero = <G::Engine as Engine>::Fr::zero().into_repr();
            let one = <G::Engine as Engine>::Fr::one().into_repr();

            // Sort the bases into buckets
            for (&exp, density) in exponents.iter().zip(density_map.as_ref().iter()) {
                if density {
                    if exp == zero {
                        bases.skip(1)?;
                    } else if exp == one {
                        if handle_trivial {
                            bases.add_assign_mixed(&mut acc)?;
                        } else {
                            bases.skip(1)?;
                        }
                    } else {
                        let mut exp = exp;
                        exp.shr(skip);
                        let exp = exp.as_ref()[0] % (1 << c);

                        if exp != 0 {
                            bases.add_assign_mixed(&mut buckets[(exp - 1) as usize])?;
                        } else {
                            bases.skip(1)?;
                        }
                    }
                }
            }

            // Summation by parts
            // e.g. 3a + 2b + 1c = a +
            //                    (a) + b +
            //                    ((a) + b) + c
            let mut running_sum = G::Projective::zero();
            for exp in buckets.into_iter().rev() {
                running_sum.add_assign(&exp);
                acc.add_assign(&running_sum);
            }

            Ok(acc)
        })
    };

    skip += c;

    if skip >= <G::Engine as Engine>::Fr::NUM_BITS {
        // There isn't another region.
        Box::new(this)
    } else {
        // There's another region more significant. Calculate and join it with
        // this region recursively.
        Box::new(
            this.join(multiexp_inner(pool, bases, density_map, exponents, skip, c, false))
                .map(move |(this, mut higher)| {
                    for _ in 0..c {
                        higher.double();
                    }

                    higher.add_assign(&this);

                    higher
                })
        )
    }
}

fn multiexp_simple<G>(
    pool: &Worker,
    bases: Arc<Vec<G>>,
    exponents: Arc<Vec<<<G::Engine as Engine>::Fr as PrimeField>::Repr>>,
    mut skip: u32,
    chunk: u32
) -> Box<Future<Item=<G as CurveAffine>::Projective, Error=SynthesisError>>
    where G: CurveAffine
{
    // Perform this region of the multiexp
    let this = {
        let bases = bases.clone();
        let exponents = exponents.clone();
        

        pool.compute(move || {
            // Accumulate the result
            let mut acc = G::Projective::zero();

            // Build a source for the bases
            let mut end = 0;
            if skip + chunk > (bases.len() as u32){
                end = bases.len() as u32;
            } else {
                end = skip + chunk;
            }
            for i in skip..end {
                acc.add_assign(&bases[i as usize].mul(exponents[i as usize]));
            }

            Ok(acc)
        })
    };

    if skip + chunk > (bases.len() as u32){
                Box::new(this)
            } else {
                Box::new(
                this.join(multiexp_simple(pool, bases, exponents, skip+chunk, chunk))
                    .map(move |(this, mut other)| {

                        other.add_assign(&this);
                        other
                    })
                )
            }    
}

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
pub fn multiexp<Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<<G::Engine as Engine>::Fr as PrimeField>::Repr>>
) -> Box<Future<Item=<G as CurveAffine>::Projective, Error=SynthesisError>>
    where for<'a> &'a Q: QueryDensity,
          D: Send + Sync + 'static + Clone + AsRef<Q>,
          G: CurveAffine,
          S: SourceBuilder<G>
{
    let c = if exponents.len() < 32 {
        3u32
    } else {
        (f64::from(exponents.len() as u32)).ln().ceil() as u32
    };

    if let Some(query_size) = density_map.as_ref().get_query_size() {
        // If the density map has a known query size, it should not be
        // inconsistent with the number of exponents.

        assert!(query_size == exponents.len());
    } 
    
    /*let count_bases = bases.clone();
    let count_exp = exponents.clone();
    let mut count_bases = count_bases.new();

    if density_map.as_ref().get_query_size().is_some() {
        let mut idx = 0;
        let mut ones_density = 0;
        for bit in density_map.as_ref().iter() {
            let mut acc = G::Projective::zero();
            if bit {
                count_bases.add_assign_mixed(&mut acc).unwrap();
                println!("{:?} {:?}", acc, exponents[idx]);
                ones_density += 1;
            }
            idx += 1;
        }
        println!("Points: {}", ones_density);
    } else {
        for &exp in count_exp.iter() {
            let mut acc = G::Projective::zero();
            count_bases.add_assign_mixed(&mut acc).unwrap();
            println!("{:?} {:?}", acc, exp);
        }
        println!("Points: {}", exponents.len());
    }*/
    multiexp_inner(pool, bases, density_map, exponents, 0, c, true)
}

#[test]
fn test_with_bls12() {
    fn naive_multiexp<G: CurveAffine>(
        bases: Arc<Vec<G>>,
        exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>
    ) -> G::Projective
    {
        assert_eq!(bases.len(), exponents.len());

        let mut acc = G::Projective::zero();

        for (base, exp) in bases.iter().zip(exponents.iter()) {
            acc.add_assign(&base.mul(*exp));
        }

        acc
    }

    use rand::{self, Rand};
    use pairing::bls12_381::Bls12;

    const SAMPLES: usize = 1 << 14;

    let rng = &mut rand::thread_rng();
    let v = Arc::new((0..SAMPLES).map(|_| <Bls12 as Engine>::Fr::rand(rng).into_repr()).collect::<Vec<_>>());
    let g = Arc::new((0..SAMPLES).map(|_| <Bls12 as Engine>::G1::rand(rng).into_affine()).collect::<Vec<_>>());

    let naive = naive_multiexp(g.clone(), v.clone());

    let pool = Worker::new();

    let fast = multiexp(
        &pool,
        (g, 0),
        FullDensity,
        v
    ).wait().unwrap();

    assert_eq!(naive, fast);
}

use std::io::BufReader;
use std::io::BufRead;
use std::fs::File;
use std::mem::transmute;
use std::time::Instant;

use pairing::bls12_381::Bls12;

type G1Affine = <Bls12 as Engine>::G1Affine;
type G1Projective = <Bls12 as Engine>::G1;
type FrRepr = <<Bls12 as Engine>::Fr as PrimeField>::Repr;
type Fr = <Bls12 as Engine>::Fr;
type Fq = <Bls12 as Engine>::Fq;

fn load_data(path: &str, num_elements: usize, points: &mut Vec<G1Affine>, exponents: &mut Vec<FrRepr>) {
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

#[inline(always)]
fn test_cpu_multiexp_pregen() {
    let test_set =  [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000];
    
    for num_points in test_set.iter() {

        let mut points = Vec::new();
        let mut exponents = Vec::new();

        let mut results = Vec::new();
        load_data(&"./src/point_exp_pairs.txt", *num_points, &mut points, &mut exponents);

        exponents.sort_unstable();
        for i in 0..exponents.len() {
            if i%20 != 0 {
                let mut e = exponents[i - i%200];
                exponents[i].sub_noborrow(& e);
            }
        }

        for i in 0..exponents.len() {
            if i%20 == 0 {
                exponents[i].0 = [0, 0, 0, 0];
            }
        }

        print!("Points: {}, ", *num_points);

        for _ in 0..1 {
            let g = Arc::new(points.clone());
            let v = Arc::new(exponents.clone());

            let pool = Worker::new();

            let now = Instant::now();
            let fast = multiexp(
                &pool,
                (g, 0),
                FullDensity,
                v
            ).wait().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            results.push(fast);
        }
        println!();

        for i in 1..results.len() {
            assert_eq!(results[i], results[i-1]);
        }
    }
}

#[inline(always)]
fn test_cpu_multiexp_vanilla() {
    let test_set =  [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000];
    
    for num_points in test_set.iter() {

        let mut points = Vec::new();
        let mut exponents = Vec::new();

        let mut results = Vec::new();
        let iterations = 1;
        load_data(&"./src/point_exp_pairs.txt", *num_points, &mut points, &mut exponents);


        print!("Points: {}, ", *num_points);

        for _ in 0..iterations {
            let g = Arc::new(points.clone());
            let v = Arc::new(exponents.clone());

            let pool = Worker::new();

            let now = Instant::now();
            let fast = multiexp(
                &pool,
                (g, 0),
                FullDensity,
                v
            ).wait().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            results.push(fast);
        }
        println!();

        for i in 1..results.len() {
            assert_eq!(results[i], results[i-1]);
        }
    }
}

pub fn run_tests() {
    //test_gpu_multiexp_pippenger_spread();
    //test_gpu_multiexp_pippenger_step();
    //test_gpu_reduce_cstyle().unwrap();
    //test_gpu_multiexp_simple_cstyle().unwrap();
    // test_gpu_multiexp_smart();
    // test_gpu_multiexp_pippenger_spread();
    // println!("Vanilla test");
    // test_cpu_multiexp_vanilla();

    // test_cpu_simple_dump();

    // println!("131071 dump:");
    // test_cpu_multiexp_dump();

    // println!("131071 dump, lower three quarters:");
    // test_cpu_multiexp_lower_three_quarters();

    // println!("131071 dump, lower half:");
    // test_cpu_multiexp_lower_half();

    // println!("131071 dump, 20 split");
    // test_cpu_multiexp_pregen();

    // println!("Simple multiexp GPU - entire exponent");
    // test_gpu_multiexp_simple();

    /*println!("Simple multiexp GPU - 1/2 exponent");
    test_gpu_multiexp_simple_lower_half();

    println!("Simple multiexp GPU - 1/4 exponent");
    test_gpu_multiexp_simple_lower_quarter();*/

    // println!("Smart multiexp GPU - entire exponent");
    // test_gpu_multiexp_smart();

    test_gpu_double();

    //println!("Smart multiexp GPU - entire exponent no local reduction large");
    //test_gpu_multiexp_smart_no_red_large();

    //println!("Smart multiexp GPU - entire exponent no local reduction");
    //test_gpu_multiexp_smart_no_red();

    /*println!("Smart multiexp GPU - 1/2 exponent");
    test_gpu_multiexp_smart_lower_half();

    println!("Smart multiexp GPU - 1/4 exponent");
    test_gpu_multiexp_smart_lower_quarter();*/

    //::groth16::test_proof();
}

#[inline(always)]
#[allow(dead_code, unused_variables, unused_mut)]
fn test_simple() -> ocl::core::Result<()> {
    use std::ffi::CString;
    use ocl::{core, flags};
    use ocl::enums::ArgVal;
    use ocl::builders::ContextProperties;

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims..
    let platform_id = core::default_platform()?;
    let device_ids = core::get_device_ids(&platform_id, None, None)?;
    let device_id = device_ids[0];
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties),
        &[device_id], None, None)?;
    let src_cstring = CString::new(src)?;
    let program = core::create_program_with_source(&context, &[src_cstring])?;
    core::build_program(&program, Some(&[device_id]), &CString::new("")?,
        None, None)?;
    let queue = core::create_command_queue(&context, &device_id, None)?;


    let dims = [1 << 20, 1, 1];

    // (2) Create a `Buffer`:
    let mut vec = vec![0.0f32; dims[0]];
    let buffer = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE |
        flags::MEM_COPY_HOST_PTR, dims[0], Some(&vec))? };

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = core::create_kernel(&program, "add")?;
    core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buffer))?;
    core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&3.14f32))?;

    // (4) Run the kernel:
    unsafe { core::enqueue_kernel(&queue, &kernel, 1, None, &dims,
        None, None::<core::Event>, None::<&mut core::Event>)?; }

    // (5) Read results from the device into a vector:
    unsafe { core::enqueue_read_buffer(&queue, &buffer, true, 0, &mut vec,
        None::<core::Event>, None::<&mut core::Event>)?; }

    // Print an element:
    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}
#[inline(always)]
fn test_cpu_multiexp_lower_half() {
    use std::io::Write;
    use std::io::stdout;
    let test_set =  [131071];
    
    for num_points in test_set.iter() {

        let mut points = Vec::new();
        let mut exponents = Vec::new();

        //let mut results = Vec::new();
        load_data(&"./src/131071_dump.txt", *num_points, &mut points, &mut exponents);

        //println!("Points: {}, ", *num_points);
        for i in 0..exponents.len() {
            exponents[i].0[3] = 0;
            exponents[i].0[2] = 0;
        }
        //for _ in 0..1 {
            let g = Arc::new(points.clone());
            let v = Arc::new(exponents.clone());

            let pool = Worker::new();

            let now = Instant::now();
            let fast = multiexp(
                &pool,
                (g, 0),
                FullDensity,
                v
            ).wait().unwrap();
            let duration = now.elapsed();
            println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
            //println!("{:?}", fast);
            // results.push(fast);
        //}
        println!();

        /*for i in 1..results.len() {
            assert_eq!(results[i], results[i-1]);
        }*/
    }
}

#[inline(always)]
fn test_cpu_multiexp_lower_three_quarters() {
    use std::io::Write;
    use std::io::stdout;
    let test_set =  [131071];
    
    for num_points in test_set.iter() {

        let mut points = Vec::new();
        let mut exponents = Vec::new();

        //let mut results = Vec::new();
        load_data(&"./src/131071_dump.txt", *num_points, &mut points, &mut exponents);

        //println!("Points: {}, ", *num_points);
        for i in 0..exponents.len() {
            exponents[i].0[3] = 0;
        }
        //for _ in 0..1 {
            let g = Arc::new(points.clone());
            let v = Arc::new(exponents.clone());

            let pool = Worker::new();

            let now = Instant::now();
            let fast = multiexp(
                &pool,
                (g, 0),
                FullDensity,
                v
            ).wait().unwrap();
            let duration = now.elapsed();
            println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
            //println!("{:?}", fast);
            // results.push(fast);
        //}
        println!();

        /*for i in 1..results.len() {
            assert_eq!(results[i], results[i-1]);
        }*/
    }
}

#[inline(always)]
fn test_cpu_multiexp_dump() {
    use std::io::Write;
    use std::io::stdout;
    let test_set =  [131071];
    
    for num_points in test_set.iter() {

        let mut points = Vec::new();
        let mut exponents = Vec::new();

        //let mut results = Vec::new();
        load_data(&"./src/131071_dump.txt", *num_points, &mut points, &mut exponents);

        //println!("Points: {}, ", *num_points);

        //for _ in 0..1 {
            let g = Arc::new(points.clone());
            let v = Arc::new(exponents.clone());

            let pool = Worker::new();

            let now = Instant::now();
            let fast = multiexp(
                &pool,
                (g, 0),
                FullDensity,
                v
            ).wait().unwrap();
            let duration = now.elapsed();
            println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
            //println!("{:?}", fast);
            // results.push(fast);
        //}
        println!();

        /*for i in 1..results.len() {
            assert_eq!(results[i], results[i-1]);
        }*/
    }
}

#[inline(always)]
fn test_cpu_simple_dump() {
    use std::io::Write;
    use std::io::stdout;
    let test_set =  [131071];
    
    for num_points in test_set.iter() {

        let mut points = Vec::new();
        let mut exponents = Vec::new();

        //let mut results = Vec::new();
        load_data(&"./src/131071_dump.txt", *num_points, &mut points, &mut exponents);

        //println!("Points: {}, ", *num_points);

        //for _ in 0..1 {
            let g = Arc::new(points.clone());
            let v = Arc::new(exponents.clone());

            let pool = Worker::new();
            println!("HERE");
            let now = Instant::now();
            let fast = multiexp_simple(
                &pool,
                g,
                v,
                0,
                5000
            ).wait().unwrap();
            let duration = now.elapsed();
            println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
            //println!("{:?}", fast);
            // results.push(fast);
        //}
        println!();

        /*for i in 1..results.len() {
            assert_eq!(results[i], results[i-1]);
        }*/
    }
}

#[test]
fn test_gpu_reduce() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let group_set = [2, 4, 8, 16, 32, 64, 128, 256];
    let iterations = 20;

    let num_points = 132000;
    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);

    let mut points_gpu = Vec::new();
    // let exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*3*FQ_WIDTH)
                                .build().unwrap();

    println!("STARTED READING IN THE POINTS!");
    for i in 0..num_points {
        unsafe {
            let point: [u64; 3*FQ_WIDTH] = transmute(points[i].into_projective());

            for &num in point.iter() {
                points_gpu.push(num);
            }
        }
    }
    println!("FINISHED READING IN THE POINTS!");

    for &group_size in group_set.iter() {
        print!("Group size: {}, ", group_size);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&length)
                        .arg_local::<u64>(group_size * FQ_WIDTH * 3)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }

                length = (length + (group_size as u32) - 1) / (group_size as u32);
                dims = (dims + group_size - 1) / group_size;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_points.cmd().read(&mut result).enq().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p);
            }
        }
        println!();
    }

    for i in 1..results.len() {
        assert_eq!(results[i-1], results[i]);
    }
}

#[test]
fn test_gpu_reduce_global() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel, EventList};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let group_set = [/*1, 2, 4, 8, 16,*/ 32, 64, 128, 256];
    let iterations = 5;

    let num_points = 1320000;
    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);

    let mut total = G1Projective::zero();

    for &point in points.iter() {
        total.add_assign(&point.into_projective());
    }

    results.push(total.into_affine());
    let mut points_gpu = Vec::new();
    // let exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*3*FQ_WIDTH)
                                .build().unwrap();

    println!("STARTED READING IN THE POINTS!");
    for i in 0..num_points {
        unsafe {
            let point: [u64; 3*FQ_WIDTH] = transmute(points[i].into_projective());

            for &num in point.iter() {
                points_gpu.push(num);
            }
        }
    }
    println!("FINISHED READING IN THE POINTS!");

    for &group_size in group_set.iter() {
        print!("Group size: {}, ", group_size);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

            

            while length > 1 {
                
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }

                queue.finish();
                length = (length + 1) / 2;
                dims = (((length as usize) + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_points.cmd().read(&mut result).enq().unwrap();
            queue.finish().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p.into_affine());
            }
        }
        println!();
    }

    for i in 1..results.len() {
        assert_eq!(results[i-1], results[i]);
    }
}

#[test]
fn test_gpu_reduce_global_lengths() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel, EventList};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 1;
    
    let point_set = [1, 2, 3, 32, 64, 128, 31, 33, 131];
    
    for &num_points in point_set.iter() {
        let mut points = Vec::new();
        let mut exponents = Vec::new();

        let mut results = Vec::new();
        load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);

        let mut total = G1Projective::zero();

        for &point in points.iter() {
            total.add_assign(&point.into_projective());
        }

        results.push(total.into_affine());
        let mut points_gpu = Vec::new();
        // let exps_gpu = Vec::new();

        let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

        let platform = Platform::default();
        let device = Device::first(platform).unwrap();

        let context = Context::builder()
                            .platform(platform)
                            .devices(device.clone())
                            .build().unwrap();

        let program = Program::builder()
                            .devices(device)
                            .src(opencl_string)
                            .build(&context).unwrap();

        let queue = Queue::new(&context, device, None).unwrap();
        
        let buffer_points = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*3*FQ_WIDTH)
                                    .build().unwrap();

        println!("STARTED READING IN THE POINTS!");
        for i in 0..num_points {
            unsafe {
                let point: [u64; 3*FQ_WIDTH] = transmute(points[i].into_projective());

                for &num in point.iter() {
                    points_gpu.push(num);
                }
            }
        }
        println!("FINISHED READING IN THE POINTS!");

        let group_size = 32;

        print!("Number of points: {}, ", num_points);
        let now = Instant::now();

        buffer_points.cmd().write(&points_gpu).enq().unwrap();

        let mut length = num_points as u32;
        let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

        

        while length > 1 {
            let kernel = Kernel::builder()
                    .program(&program)
                    .name("projective_reduce_step_global")
                    .global_work_size(dims)
                    .arg(&buffer_points)
                    .arg(&length)
                    .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size)
                    .enq().unwrap();
            }

            queue.finish().unwrap();
            length = (length + 1) / 2;
            dims = (((length as usize) + group_size - 1) / group_size) * group_size;
        }

        let mut result = vec![0u64; 3*FQ_WIDTH];

        buffer_points.cmd().read(&mut result).enq().unwrap();
        queue.finish().unwrap();
        let duration = now.elapsed();
        print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

        let mut result_cpu = G1Projective::zero();

        for &point in points.iter() {
            result_cpu.add_assign_mixed(&point);
        }

        let mut result_arr = [0u64; 3*FQ_WIDTH];
        for i in 0..result.len() {
            result_arr[i] = result[i];
        }
        unsafe {
            let p: G1Projective = transmute(result_arr);
            assert_eq!(p, result_cpu);
        }
        
        println!();

    }
}

#[inline(always)]
fn test_gpu_reduce_cstyle() -> ocl::core::Result<()> {
    use std::ffi::CString;
    use ocl::{core, flags};
    use ocl::enums::ArgVal;
    use ocl::builders::ContextProperties;
    use std::fs::read_to_string;

    use ocl::{Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 3;

    let test_set = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000];
    let group_set = [8, 16, 32, 64, 128, 256];
    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();

    //let mut points_gpu = Vec::new();
    //let mut exps_gpu = Vec::new();
    let mut results_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform_id = core::default_platform()?;
    let device_ids = core::get_device_ids(&platform_id, None, None)?;
    let device_id = device_ids[0];
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties),
        &[device_id], None, None)?;
    let src_cstring = CString::new(opencl_string)?;
    let program = core::create_program_with_source(&context, &[src_cstring])?;
    core::build_program(&program, Some(&[device_id]), &CString::new("")?,
        None, None)?;
    let queue = core::create_command_queue(&context, &device_id, None)?;

    for &group_size in group_set.iter() {
        for &num_points in test_set.iter() {

            let buffer_points = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE, 
            num_points*(2*FQ_WIDTH+1), None::<&[u64]>)? };

            let buffer_exponents = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE, 
            num_points*FR_WIDTH, None::<&[u64]>)? };

            let buffer_results = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE, 
            num_points*FQ_WIDTH*3, None::<&[u64]>)? };

            points.clear();
            exponents.clear();
            //points_gpu.clear();
            //exps_gpu.clear();
            results_gpu.clear();
            load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
            for i in 0..num_points {
                unsafe {
                    let point: [u64; 3*FQ_WIDTH] = transmute(points[i].into_projective());


                    for &num in point.iter() {
                        results_gpu.push(num);
                    }

                }
            }

            //let group_size = 128;
            //let group_size_exp = 128;
            println!("Points: {}, Group exp {}", num_points, group_size);
            for _ in 0..iterations {
                
                let now = Instant::now();
                unsafe {
                    //core::enqueue_write_buffer(&queue, &buffer_points, true, 0, &points_gpu, None::<core::Event>, None::<&mut core::Event>)?;
                    //core::enqueue_write_buffer(&queue, &buffer_exponents, true, 0, &exps_gpu, None::<core::Event>, None::<&mut core::Event>)?;
                    core::enqueue_write_buffer(&queue, &buffer_results, true, 0, &results_gpu, None::<core::Event>, None::<&mut core::Event>)?;
                }

                let mut length = num_points as u32;
                
                let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

                while length > 1 {
                    let kernel = core::create_kernel(&program, "projective_reduce_step_global")?;
                    core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buffer_results))?;
                    core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&length))?;
                    let dims_arr = [dims, 1, 1];
                    unsafe { core::enqueue_kernel(&queue, &kernel, 1, None, &dims_arr,
                Some([group_size, 1, 1]), None::<core::Event>, None::<&mut core::Event>)?; }

                    length = (length + 1) / 2;
                    dims = (dims + 1) / 2;
                    dims = ((dims + group_size - 1) / group_size) * group_size;
                }

                let mut result = vec![0u64; 3*FQ_WIDTH];

                unsafe { core::enqueue_read_buffer(&queue, &buffer_results, true, 0, &mut result,
            None::<core::Event>, None::<&mut core::Event>)?; }

                let duration = now.elapsed();
                println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

                let mut result_arr = [0u64; 3*FQ_WIDTH];
                for i in 0..result.len() {
                    result_arr[i] = result[i];
                }
                unsafe {
                    let p: G1Projective = transmute(result_arr);
                    results.push(p);
                }
            }
            println!();
        
            for i in 1..results.len() {
                assert_eq!(results[i-1], results[i]);
            }
            results.clear();
        }
    }
    Ok(())
}

#[inline(always)]
fn test_gpu_multiexp_simple_cstyle() -> ocl::core::Result<()> {
    use std::ffi::CString;
    use ocl::{core, flags};
    use ocl::enums::ArgVal;
    use ocl::builders::ContextProperties;
    use std::fs::read_to_string;

    use ocl::{Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 3;

    let test_set = [131071];
    let group_set = [64, 128, 256];
    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();

    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform_id = core::default_platform()?;
    let device_ids = core::get_device_ids(&platform_id, None, None)?;
    let device_id = device_ids[0];
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties),
        &[device_id], None, None)?;
    let src_cstring = CString::new(opencl_string)?;
    let program = core::create_program_with_source(&context, &[src_cstring])?;
    core::build_program(&program, Some(&[device_id]), &CString::new("-save-temps")?,
        None, None)?;
    let queue = core::create_command_queue(&context, &device_id, None)?;

    for &group_size_exp in group_set.iter() {
        for &num_points in test_set.iter() {

            let buffer_points = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE, 
            num_points*(2*FQ_WIDTH+1), None::<&[u64]>)? };

            let buffer_exponents = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE, 
            num_points*FR_WIDTH, None::<&[u64]>)? };

            let buffer_results = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE, 
            num_points*FQ_WIDTH*3, None::<&[u64]>)? };

            points.clear();
            exponents.clear();
            points_gpu.clear();
            exps_gpu.clear();
            load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
            for i in 0..num_points {
                unsafe {
                    let point: (Fq, Fq, bool) = transmute(points[i]);

                    let point_0: [u64; FQ_WIDTH] = transmute(point.0);
                    let point_1: [u64; FQ_WIDTH] = transmute(point.1);

                    for &num in point_0.iter() {
                        points_gpu.push(num);
                    }

                    for &num in point_1.iter() {
                        points_gpu.push(num);
                    }

                    points_gpu.push(point.2 as u64);

                    let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

                    for &e in exp.iter() {
                        exps_gpu.push(e);
                    }
                }
            }

            let group_size = 64;
            //let group_size_exp = 128;
            println!("Points: {}, Group exp {}", num_points, group_size_exp);
            for _ in 0..iterations {
                

                unsafe {
                    core::enqueue_write_buffer(&queue, &buffer_points, true, 0, &points_gpu, None::<core::Event>, None::<&mut core::Event>)?;
                    core::enqueue_write_buffer(&queue, &buffer_exponents, true, 0, &exps_gpu, None::<core::Event>, None::<&mut core::Event>)?;
                }

                let mut length = num_points as u32;
                let mut dims = ((num_points + group_size_exp - 1) / group_size_exp) * group_size_exp;

                let kernel = core::create_kernel(&program, "test_affine_mul_binary")?;
                core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buffer_points))?;
                core::set_kernel_arg(&kernel, 1, ArgVal::mem(&buffer_exponents))?;
                core::set_kernel_arg(&kernel, 2, ArgVal::mem(&buffer_results))?;
                core::set_kernel_arg(&kernel, 3, ArgVal::scalar(&length))?;
                
                let dims_arr = [dims, 1, 1];
                let now = Instant::now();
                unsafe { core::enqueue_kernel(&queue, &kernel, 1, None, &dims_arr,
                Some([group_size_exp, 1, 1]), None::<core::Event>, None::<&mut core::Event>)?; }
                
                let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

                while length > 1 {
                    core::finish(&queue)?;
                    let kernel = core::create_kernel(&program, "projective_reduce_step_global")?;
                    core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buffer_results))?;
                    core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&length))?;
                    let dims_arr = [dims, 1, 1];
                    unsafe { core::enqueue_kernel(&queue, &kernel, 1, None, &dims_arr,
                Some([group_size, 1, 1]), None::<core::Event>, None::<&mut core::Event>)?; }

                    length = (length + 1) / 2;
                    dims = (dims + 1) / 2;
                    dims = ((dims + group_size - 1) / group_size) * group_size;
                }

                let mut result = vec![0u64; 3*FQ_WIDTH];
                core::finish(&queue)?;
                unsafe { core::enqueue_read_buffer(&queue, &buffer_results, true, 0, &mut result,
            None::<core::Event>, None::<&mut core::Event>)?; }

                let duration = now.elapsed();
                println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

                let mut result_arr = [0u64; 3*FQ_WIDTH];
                for i in 0..result.len() {
                    result_arr[i] = result[i];
                }
                unsafe {
                    let p: G1Projective = transmute(result_arr);
                    results.push(p);
                }
            }
            println!();
        
            for i in 1..results.len() {
                assert_eq!(results[i-1], results[i]);
            }
            results.clear();
        }
    }
    Ok(())
}

#[inline(always)]
fn test_gpu_multiexp_simple() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let test_set = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000];

    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();

    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    println!("Devices: {}", Device::list_all(platform).unwrap().len());
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    println!("Preparing to build!");
    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    println!("Kernel built!");
    let queue = Queue::new(&context, device, None).unwrap();
    
    for &num_points in test_set.iter() {
        let buffer_points = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*(2*FQ_WIDTH+1))
                                    .build().unwrap();

        let buffer_exponents = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FR_WIDTH)
                                    .build().unwrap();

        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FQ_WIDTH*3)
                                    .build().unwrap();

        points.clear();
        exponents.clear();
        points_gpu.clear();
        exps_gpu.clear();
        load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
        for i in 0..num_points {
            unsafe {
                let point: (Fq, Fq, bool) = transmute(points[i]);

                let point_0: [u64; FQ_WIDTH] = transmute(point.0);
                let point_1: [u64; FQ_WIDTH] = transmute(point.1);

                for &num in point_0.iter() {
                    points_gpu.push(num);
                }

                for &num in point_1.iter() {
                    points_gpu.push(num);
                }

                points_gpu.push(point.2 as u64);

                let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

                for &e in exp.iter() {
                    exps_gpu.push(e);
                }
            }
        }

        let group_size = 128;
        let group_size_exp = 32;
        println!("Points: {}, ", num_points);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size_exp - 1) / group_size_exp) * group_size_exp;

            let kernel = Kernel::builder()
                        .program(&program)
                        .name("test_affine_mul_binary")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                length = (length + 1) / 2;
                dims = (dims + 1) / 2;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            queue.finish().unwrap();
            let duration = now.elapsed();
            println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p);
            }
        }
        println!();
    
        for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }
        results.clear();
    }
}

#[inline(always)]
fn test_gpu_double() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 5;

    let test_set = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000, 2000000, 4000000];

    let mut points = Vec::new();
    let mut exponents = Vec::new();

    //let mut results = Vec::new();

    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    println!("Devices: {}", Device::list_all(platform).unwrap().len());
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    println!("Preparing to build!");
    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .cmplr_opt("-save-temps")
                        .build(&context).unwrap();

    println!("Kernel built!");
    let queue = Queue::new(&context, device, None).unwrap();
    
    for &num_points in test_set.iter() {
        let buffer_points = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*(3*FQ_WIDTH))
                                    .build().unwrap();

        let buffer_exponents = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FR_WIDTH)
                                    .build().unwrap();

        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FQ_WIDTH*3)
                                    .build().unwrap();

        points.clear();
        exponents.clear();
        points_gpu.clear();
        exps_gpu.clear();
        load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
        for i in 0..num_points {
            unsafe {
                let point: ([u64; 3*FQ_WIDTH]) = transmute(points[i].into_projective());


                for &num in point.iter() {
                    points_gpu.push(num);
                }


                let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

                for &e in exp.iter() {
                    exps_gpu.push(e);
                }
            }
        }

        let group_size = 128;
        let group_size_exp = 32;
        println!("Points: {}, ", num_points);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            //buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size_exp - 1) / group_size_exp) * group_size_exp;

            let kernel = Kernel::builder()
                        .program(&program)
                        .name("double_kernel_test")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&length)
                        .arg(&buffer_results)
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_points.cmd().read(&mut result).enq().unwrap();
            queue.finish().unwrap();
            let duration = now.elapsed();
            println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];

        }
        println!();
    
        //results.clear();
    }
}

#[inline(always)]
fn test_gpu_multiexp_simple_lower_half() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let test_set = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000];

    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();

    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    println!("Devices: {}", Device::list_all(platform).unwrap().len());
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();
    println!("Preparing to build!");
    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    println!("Kernel built!");
    let queue = Queue::new(&context, device, None).unwrap();
    
    for &num_points in test_set.iter() {
        let buffer_points = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*(2*FQ_WIDTH+1))
                                    .build().unwrap();

        let buffer_exponents = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FR_WIDTH)
                                    .build().unwrap();

        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FQ_WIDTH*3)
                                    .build().unwrap();

        points.clear();
        exponents.clear();
        points_gpu.clear();
        exps_gpu.clear();
        load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
        for i in 0..num_points {
            unsafe {
                let point: (Fq, Fq, bool) = transmute(points[i]);

                let point_0: [u64; FQ_WIDTH] = transmute(point.0);
                let point_1: [u64; FQ_WIDTH] = transmute(point.1);

                for &num in point_0.iter() {
                    points_gpu.push(num);
                }

                for &num in point_1.iter() {
                    points_gpu.push(num);
                }

                points_gpu.push(point.2 as u64);

                let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

                for &e in exp.iter() {
                    exps_gpu.push(e);
                }
            }
        }

        let group_size = 128;
        let group_size_exp = 32;
        print!("Points: {}, ", num_points);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size_exp - 1) / group_size_exp) * group_size_exp;

            let kernel = Kernel::builder()
                        .program(&program)
                        .name("test_affine_mul_binary_lower_half")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                length = (length + 1) / 2;
                dims = (dims + 1) / 2;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            queue.finish().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p);
            }
        }
        println!();
    
        for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }
        results.clear();
    }
}

#[inline(always)]
fn test_gpu_multiexp_simple_lower_quarter() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let test_set = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 132000, 200000, 500000, 1000000];

    let mut points = Vec::new();
    let mut exponents = Vec::new();

    let mut results = Vec::new();

    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    for &num_points in test_set.iter() {
        let buffer_points = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*(2*FQ_WIDTH+1))
                                    .build().unwrap();

        let buffer_exponents = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FR_WIDTH)
                                    .build().unwrap();

        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len(num_points*FQ_WIDTH*3)
                                    .build().unwrap();

        points.clear();
        exponents.clear();
        points_gpu.clear();
        exps_gpu.clear();
        load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
        for i in 0..num_points {
            unsafe {
                let point: (Fq, Fq, bool) = transmute(points[i]);

                let point_0: [u64; FQ_WIDTH] = transmute(point.0);
                let point_1: [u64; FQ_WIDTH] = transmute(point.1);

                for &num in point_0.iter() {
                    points_gpu.push(num);
                }

                for &num in point_1.iter() {
                    points_gpu.push(num);
                }

                points_gpu.push(point.2 as u64);

                let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

                for &e in exp.iter() {
                    exps_gpu.push(e);
                }
            }
        }

        let group_size = 128;
        let group_size_exp = 32;
        print!("Points: {}, ", num_points);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size_exp - 1) / group_size_exp) * group_size_exp;

            let kernel = Kernel::builder()
                        .program(&program)
                        .name("test_affine_mul_binary_lower_quarter")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            let mut dims = ((num_points + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                length = (length + 1) / 2;
                dims = (dims + 1) / 2;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            queue.finish().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p);
            }
        }
        println!();
    
        for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }
        results.clear();
    }
}

#[test]
fn test_gpu_exp_simple() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 20;

    let group_set = [16, 32, 64, 128, 256];

    let num_points = 10000;
    let mut points = Vec::new();
    let mut exponents = Vec::new();

    //let mut results = Vec::new();

    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    println!("COMPILATION STARTED!");
    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    let buffer_results = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FQ_WIDTH*3)
                                .build().unwrap();

    println!("COMPILATION FINISHED!");

    println!("DATA LOAD STARTED!");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);

    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }

    println!("DATA LOAD FINISHED!");
    
    for &group_size_exp in group_set.iter() {
        print!("Group: {}, ", group_size_exp);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + group_size_exp - 1) / group_size_exp) * group_size_exp;

            let kernel = Kernel::builder()
                        .program(&program)
                        .name("test_affine_mul_binary")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            queue.finish().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
            
        }
        println!();
    }

}

#[test]
fn test_gpu_multiexp_chunking() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [/*2, 5,*/ 10, 13, 17, 20, 21, 23];

    let num_points = 131;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 64;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3)
                                    .build().unwrap();

        println!("Chunk: {}, ", chunk);
        let now = Instant::now();

        buffer_points.cmd().write(&points_gpu).enq().unwrap();
        buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

        let mut length = num_points as u32;
        let mut dims = ((num_points + chunk - 1) / chunk) * group_size_exp;
        let chunk_gpu = chunk as u32;
        let kernel = Kernel::builder()
                    .program(&program)
                    .name("affine_mulexp_smart")
                    .global_work_size(dims)
                    .arg(&buffer_points)
                    .arg(&buffer_exponents)
                    .arg(&length)
                    .arg(&chunk_gpu)
                    .arg(&buffer_results)
                    
                    .build().unwrap();

        unsafe {
            kernel.cmd()
                .queue(&queue)
                .global_work_offset(kernel.default_global_work_offset())
                .global_work_size(dims)
                .local_work_size(group_size_exp)
                .enq().unwrap();
        }
        
    
        let mut result = vec![0u64; 3*FQ_WIDTH * ((num_points + chunk - 1) / chunk)];

        buffer_results.cmd().read(&mut result).enq().unwrap();
        let duration = now.elapsed();
        println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

        results.clear();
        for j in 0..result.len()/(3*FQ_WIDTH) {
            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result_arr.len() {
                result_arr[i] = result[3*FQ_WIDTH*j + i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p);
                // println!("{}",j);
            }
        }

        let mut results_cpu = Vec::new();

        for i in 0..points.len() {
            if i%chunk == 0 {
                results_cpu.push(points[i].mul(exponents[i]));
            } else {
                results_cpu.index_mut(i/chunk).add_assign(&points[i].mul(exponents[i]));
            }
        }
        assert_eq!(results.len(), results_cpu.len());
        println!("{:?}", results.len());
        for i in 0..results.len() {
            assert_eq!(results[i], results_cpu[i]);
        }
        println!();
    }
}


#[inline(always)]
fn test_gpu_multiexp_smart() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [/*2, 5,*/ 10, 20, 40, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 64;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3)
                                    .build().unwrap();

        print!("Chunk: {}, ", chunk);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + chunk - 1) / chunk) * group_size_exp;
            let chunk_gpu = chunk as u32;
            let kernel = Kernel::builder()
                        .program(&program)
                        .name("affine_mulexp_smart")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&length)
                        .arg(&chunk_gpu)
                        .arg(&buffer_results)
                        
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            

            let group_size = 128;
            length = ((num_points + chunk - 1) / chunk) as u32;
            let mut dims = ((length + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                // queue.finish();
                length = (length + 1) / 2;
                dims = (dims + 1) / 2;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p.into_affine());
            }
        }
        println!();
        
        
    }

    for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }
}

#[inline(always)]
fn test_gpu_multiexp_smart_no_red() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 64;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3*group_size_exp)
                                    .build().unwrap();

        print!("Chunk: {}, ", chunk);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + chunk - 1) / chunk) * group_size_exp;
            let chunk_gpu = chunk as u32;
            let kernel = Kernel::builder()
                        .program(&program)
                        .name("affine_mulexp_smart_no_red")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&length)
                        .arg(&chunk_gpu)
                        .arg(&buffer_results)
                        
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            

            let group_size = 64;
            length = (((num_points + chunk - 1) / chunk) * group_size_exp) as u32;
            let mut dims = length;

            while length > 64 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_pippinger_reduction")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                
                length = ((length / group_size_exp as u32) + 1) / 2 * (group_size_exp as u32);
                dims = length;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH*64];
            let mut result_arr = [0u64; 3*FQ_WIDTH*64];

            buffer_results.cmd().read(&mut result).enq().unwrap();

            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            let mut partial_sum = G1Projective::zero();
            let mut sum = G1Projective::zero();
            
            unsafe {
                let mut result_cast: [G1Projective; 64] = transmute(result_arr);
                for i in (0..63).rev() {
                    partial_sum.add_assign(&result_cast[i]);
                    sum.add_assign(&partial_sum.clone());
                }
                results.push(sum.into_affine());
            }
            
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
        }
        println!();
        
        
    }

    /*for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }*/
}



#[inline(always)]
fn test_gpu_multiexp_smart_no_red_large() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 32;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3*group_size_exp)
                                    .build().unwrap();

        print!("Chunk: {}, ", chunk);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + chunk - 1) / chunk) * group_size_exp;
            let chunk_gpu = chunk as u32;
            let kernel = Kernel::builder()
                        .program(&program)
                        .name("affine_mulexp_smart_no_red_large")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&length)
                        .arg(&chunk_gpu)
                        .arg(&buffer_results)
                        
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            

            let group_size = 64;
            length = (((num_points + chunk - 1) / chunk) * group_size_exp) as u32;
            let mut dims = length;

            /*while length > 64 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_pippinger_reduction")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                
                length = ((length / group_size_exp as u32) + 1) / 2 * (group_size_exp as u32);
                dims = length;
            }*/

            let mut result = vec![0u64; 3*FQ_WIDTH*64];
            let mut result_arr = [0u64; 3*FQ_WIDTH*64];

            buffer_results.cmd().read(&mut result).enq().unwrap();

            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            let mut partial_sum = G1Projective::zero();
            let mut sum = G1Projective::zero();
            
            unsafe {
                let mut result_cast: [G1Projective; 64] = transmute(result_arr);
                for i in (0..63).rev() {
                    partial_sum.add_assign(&result_cast[i]);
                    sum.add_assign(&partial_sum.clone());
                }
                results.push(sum.into_affine());
            }
            
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
        }
        println!();
        
        
    }

    /*for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }*/
}

#[inline(always)]
fn test_gpu_multiexp_pippenger_spread() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [64, 128, 256, 500, 1000, 1500, 2000, 3000, 4000, 5000, 5500, 6000, 7500, 8000, 8500, 9000, 9500, 10000, 11000, 12000, 13000, 14000, 15000, 20000, 25000, 30000];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 256;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3*256)
                                    .build().unwrap();

        print!("Chunk: {}, ", chunk);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + chunk - 1) / chunk) * group_size_exp;
            let chunk_gpu = chunk as u32;
            let kernel = Kernel::builder()
                        .program(&program)
                        .name("pippenger_spread")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&length)
                        .arg(&chunk_gpu)
                        .arg(&buffer_results)
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            
            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p.into_affine());
            }
        }
        println!();  
    }    
}


#[inline(always)]
fn test_gpu_multiexp_pippenger_step() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [/*2, 5,*/ 10, 20, 30, 33, 35, 37, 40, 42, 45, 50, 60];//, 70, 80, 90, 100, 125, 150, 175, 200, 250, 500, 1000];
    let group_set = [32, 64, 96, 128, 256];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    for &group_size_exp in group_set.iter() {

        for &chunk in chunk_size.iter() {
            let buffer_results = Buffer::<u64>::builder()
                                        .queue(queue.clone())
                                        .flags(flags::MEM_READ_WRITE)
                                        .len((num_points+chunk-1)/chunk*FQ_WIDTH*3)
                                        .build().unwrap();

            print!("Group: {} Chunk: {}, ", group_size_exp, chunk);
            for _ in 0..iterations {
                let now = Instant::now();

                buffer_points.cmd().write(&points_gpu).enq().unwrap();
                buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

                let mut length = num_points as u32;
                let mut dims = (((num_points + chunk - 1) / chunk) + group_size_exp - 1) / group_size_exp * group_size_exp;
                let chunk_gpu = chunk as u32;
                let kernel = Kernel::builder()
                            .program(&program)
                            .name("pippenger_step_first")
                            .global_work_size(dims)
                            .arg(&buffer_points)
                            .arg(&buffer_exponents)
                            .arg(&length)
                            .arg(&chunk_gpu)
                            .arg(&buffer_results)
                            
                            .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size_exp)
                        .enq().unwrap();
                }
                
                for offset in (0..=62).rev() {
                    let offset_gpu = offset as u32;

                    let kernel = Kernel::builder()
                                .program(&program)
                                .name("pippenger_step_general")
                                .global_work_size(dims)
                                .arg(&buffer_points)
                                .arg(&buffer_exponents)
                                .arg(&length)
                                .arg(&chunk_gpu)
                                .arg(&offset_gpu)
                                .arg(&buffer_results)
                                .build().unwrap();

                    unsafe {
                        kernel.cmd()
                            .queue(&queue)
                            .global_work_offset(kernel.default_global_work_offset())
                            .global_work_size(dims)
                            .local_work_size(group_size_exp)
                            .enq().unwrap();
                    }
                }
                

                let group_size = 128;
                length = ((num_points + chunk - 1) / chunk) as u32;
                let mut dims = ((length + group_size - 1) / group_size) * group_size;

                while length > 1 {
                    let kernel = Kernel::builder()
                            .program(&program)
                            .name("projective_reduce_step_global")
                            .global_work_size(dims)
                            .arg(&buffer_results)
                            .arg(&length)
                            .build().unwrap();

                    unsafe {
                        kernel.cmd()
                            .queue(&queue)
                            .global_work_offset(kernel.default_global_work_offset())
                            .global_work_size(dims)
                            .local_work_size(group_size)
                            .enq().unwrap();
                    }
                    // queue.finish();
                    length = (length + 1) / 2;
                    dims = (dims + 1) / 2;
                    dims = ((dims + group_size - 1) / group_size) * group_size;
                }

                let mut result = vec![0u64; 3*FQ_WIDTH];

                buffer_results.cmd().read(&mut result).enq().unwrap();
                let duration = now.elapsed();
                print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

                let mut result_arr = [0u64; 3*FQ_WIDTH];
                for i in 0..result.len() {
                    result_arr[i] = result[i];
                }
                unsafe {
                    let p: G1Projective = transmute(result_arr);
                    results.push(p.into_affine());
                }
            }
            println!();
            
            
        }

        /*for i in 1..results.len() {
                assert_eq!(results[i-1], results[i]);
            }*/
    }
}

#[inline(always)]
fn test_gpu_multiexp_smart_lower_half() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [/*2, 5,*/ 10, 20, 40, 50, 100, 200, 500, 1000];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 32;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3)
                                    .build().unwrap();

        print!("Chunk: {}, ", chunk);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((num_points + chunk - 1) / chunk) * group_size_exp;
            let chunk_gpu = chunk as u32;
            let kernel = Kernel::builder()
                        .program(&program)
                        .name("affine_mulexp_smart_lower_half")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&length)
                        .arg(&chunk_gpu)
                        .arg(&buffer_results)
                        
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            

            let group_size = 128;
            length = ((num_points + chunk - 1) / chunk) as u32;
            let mut dims = ((length + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                // queue.finish();
                length = (length + 1) / 2;
                dims = (dims + 1) / 2;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p.into_affine());
            }
        }
        println!();
        
        
    }

    for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }
}

#[inline(always)]
fn test_gpu_multiexp_smart_lower_quarter() {
    use std::fs::read_to_string;

    use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel};

    use rand::{Rand, SeedableRng, XorShiftRng};
    use std::ops::IndexMut;
    use std::mem;

    const FQ_WIDTH: usize = 6;
    const FR_WIDTH: usize = 4;

    let iterations = 2;

    let chunk_size = [/*2, 5,*/ 10, 20, 40, 50, 100, 200, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1500, 1700, 2000, 2500];

    let num_points = 131071;
    let mut points = Vec::new();
    let mut exponents = Vec::new();
    let mut results = Vec::new();


    let mut points_gpu = Vec::new();
    let mut exps_gpu = Vec::new();

    let opencl_string = read_to_string("./src/bls12-381.cl").unwrap();

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    let context = Context::builder()
                        .platform(platform)
                        .devices(device.clone())
                        .build().unwrap();

    let program = Program::builder()
                        .devices(device)
                        .src(opencl_string)
                        .build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();
    
    
    let buffer_points = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*(2*FQ_WIDTH+1))
                                .build().unwrap();

    let buffer_exponents = Buffer::<u64>::builder()
                                .queue(queue.clone())
                                .flags(flags::MEM_READ_WRITE)
                                .len(num_points*FR_WIDTH)
                                .build().unwrap();

    

    points.clear();
    exponents.clear();
    points_gpu.clear();
    exps_gpu.clear();
    println!("LD");
    load_data(&"./src/point_exp_pairs.txt", num_points, &mut points, &mut exponents);
    for i in 0..num_points {
        unsafe {
            let point: (Fq, Fq, bool) = transmute(points[i]);

            let point_0: [u64; FQ_WIDTH] = transmute(point.0);
            let point_1: [u64; FQ_WIDTH] = transmute(point.1);

            for &num in point_0.iter() {
                points_gpu.push(num);
            }

            for &num in point_1.iter() {
                points_gpu.push(num);
            }

            points_gpu.push(point.2 as u64);

            let exp: [u64; FR_WIDTH] = transmute(exponents[i]);

            for &e in exp.iter() {
                exps_gpu.push(e);
            }
        }
    }
    println!("DL");

    let group_size_exp = 64;
    let subgroup_size_exp = 16;

    for &chunk in chunk_size.iter() {
        let buffer_results = Buffer::<u64>::builder()
                                    .queue(queue.clone())
                                    .flags(flags::MEM_READ_WRITE)
                                    .len((num_points+chunk-1)/chunk*FQ_WIDTH*3)
                                    .build().unwrap();

        print!("Chunk: {}, ", chunk);
        for _ in 0..iterations {
            let now = Instant::now();

            buffer_points.cmd().write(&points_gpu).enq().unwrap();
            buffer_exponents.cmd().write(&exps_gpu).enq().unwrap();

            let mut length = num_points as u32;
            let mut dims = ((((num_points + chunk - 1) / chunk) + 3) / 4) * group_size_exp;
            let chunk_gpu = chunk as u32;
            let kernel = Kernel::builder()
                        .program(&program)
                        .name("affine_mulexp_smart_quarter")
                        .global_work_size(dims)
                        .arg(&buffer_points)
                        .arg(&buffer_exponents)
                        .arg(&length)
                        .arg(&chunk_gpu)
                        .arg(&buffer_results)
                        
                        .build().unwrap();

            unsafe {
                kernel.cmd()
                    .queue(&queue)
                    .global_work_offset(kernel.default_global_work_offset())
                    .global_work_size(dims)
                    .local_work_size(group_size_exp)
                    .enq().unwrap();
            }
            
            

            let group_size = 128;
            length = ((num_points + chunk - 1) / chunk) as u32;
            let mut dims = ((length + group_size - 1) / group_size) * group_size;

            while length > 1 {
                let kernel = Kernel::builder()
                        .program(&program)
                        .name("projective_reduce_step_global")
                        .global_work_size(dims)
                        .arg(&buffer_results)
                        .arg(&length)
                        .build().unwrap();

                unsafe {
                    kernel.cmd()
                        .queue(&queue)
                        .global_work_offset(kernel.default_global_work_offset())
                        .global_work_size(dims)
                        .local_work_size(group_size)
                        .enq().unwrap();
                }
                // queue.finish();
                length = (length + 1) / 2;
                dims = (dims + 1) / 2;
                dims = ((dims + group_size - 1) / group_size) * group_size;
            }

            let mut result = vec![0u64; 3*FQ_WIDTH];

            buffer_results.cmd().read(&mut result).enq().unwrap();
            let duration = now.elapsed();
            print!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());

            let mut result_arr = [0u64; 3*FQ_WIDTH];
            for i in 0..result.len() {
                result_arr[i] = result[i];
            }
            unsafe {
                let p: G1Projective = transmute(result_arr);
                results.push(p.into_affine());
            }
        }
        println!();
        
        
    }

    for i in 1..results.len() {
            assert_eq!(results[i-1], results[i]);
        }
}