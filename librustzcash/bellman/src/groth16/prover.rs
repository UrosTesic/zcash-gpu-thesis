use rand::Rng;

use std::sync::Arc;
use std::time::Instant;

use futures::Future;

use pairing::{
    Engine,
    PrimeField,
    Field,
    CurveProjective,
    CurveAffine
};

use super::{
    ParameterSource,
    Proof
};

use ::{
    SynthesisError,
    Circuit,
    ConstraintSystem,
    LinearCombination,
    Variable,
    Index
};

use ::domain::{
    EvaluationDomain,
    Scalar
};

use ::multiexp::{
    DensityTracker,
    FullDensity,
    multiexp
};

use ::multicore::{
    Worker
};

fn eval<E: Engine>(
    lc: &LinearCombination<E>,
    mut input_density: Option<&mut DensityTracker>,
    mut aux_density: Option<&mut DensityTracker>,
    input_assignment: &[E::Fr],
    aux_assignment: &[E::Fr]
) -> E::Fr
{
    let mut acc = E::Fr::zero();

    for &(index, coeff) in lc.0.iter() {
        let mut tmp;

        match index {
            Variable(Index::Input(i)) => {
                tmp = input_assignment[i];
                if let Some(ref mut v) = input_density {
                    v.inc(i);
                }
            },
            Variable(Index::Aux(i)) => {
                tmp = aux_assignment[i];
                if let Some(ref mut v) = aux_density {
                    v.inc(i);
                }
            }
        }

        if coeff == E::Fr::one() {
           acc.add_assign(&tmp);
        } else {
           tmp.mul_assign(&coeff);
           acc.add_assign(&tmp);
        }
    }

    acc
}

struct ProvingAssignment<E: Engine> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: Vec<Scalar<E>>,
    b: Vec<Scalar<E>>,
    c: Vec<Scalar<E>>,

    // Assignments of variables
    input_assignment: Vec<E::Fr>,
    aux_assignment: Vec<E::Fr>
}

impl<E: Engine> ConstraintSystem<E> for ProvingAssignment<E> {
    type Root = Self;

    fn alloc<F, A, AR>(
        &mut self,
        _: A,
        f: F
    ) -> Result<Variable, SynthesisError>
        where F: FnOnce() -> Result<E::Fr, SynthesisError>, A: FnOnce() -> AR, AR: Into<String>
    {
        self.aux_assignment.push(f()?);
        self.a_aux_density.add_element();
        self.b_aux_density.add_element();

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(
        &mut self,
        _: A,
        f: F
    ) -> Result<Variable, SynthesisError>
        where F: FnOnce() -> Result<E::Fr, SynthesisError>, A: FnOnce() -> AR, AR: Into<String>
    {
        self.input_assignment.push(f()?);
        self.b_input_density.add_element();

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(
        &mut self,
        _: A,
        a: LA,
        b: LB,
        c: LC
    )
        where A: FnOnce() -> AR, AR: Into<String>,
              LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
              LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
              LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());

        self.a.push(Scalar(eval(
            &a,
            // Inputs have full density in the A query
            // because there are constraints of the
            // form x * 0 = 0 for each input.
            None,
            Some(&mut self.a_aux_density),
            &self.input_assignment,
            &self.aux_assignment
        )));
        self.b.push(Scalar(eval(
            &b,
            Some(&mut self.b_input_density),
            Some(&mut self.b_aux_density),
            &self.input_assignment,
            &self.aux_assignment
        )));
        self.c.push(Scalar(eval(
            &c,
            // There is no C polynomial query,
            // though there is an (beta)A + (alpha)B + C
            // query for all aux variables.
            // However, that query has full density.
            None,
            None,
            &self.input_assignment,
            &self.aux_assignment
        )));
    }

    fn push_namespace<NR, N>(&mut self, _: N)
        where NR: Into<String>, N: FnOnce() -> NR
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self)
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

pub fn create_random_proof<E, C, R, P: ParameterSource<E>>(
    circuit: C,
    params: P,
    rng: &mut R
) -> Result<Proof<E>, SynthesisError>
    where E: Engine, C: Circuit<E>, R: Rng
{
    let r = rng.gen();
    let s = rng.gen();

    create_proof::<E, C, P>(circuit, params, r, s)
}

pub fn create_proof<E, C, P: ParameterSource<E>>(
    circuit: C,
    mut params: P,
    r: E::Fr,
    s: E::Fr
) -> Result<Proof<E>, SynthesisError>
    where E: Engine, C: Circuit<E>
{
    let mut prover = ProvingAssignment {
        a_aux_density: DensityTracker::new(),
        b_input_density: DensityTracker::new(),
        b_aux_density: DensityTracker::new(),
        a: vec![],
        b: vec![],
        c: vec![],
        input_assignment: vec![],
        aux_assignment: vec![]
    };

    prover.alloc_input(|| "", || Ok(E::Fr::one()))?;

    circuit.synthesize(&mut prover)?;

    for i in 0..prover.input_assignment.len() {
        prover.enforce(|| "",
            |lc| lc + Variable(Index::Input(i)),
            |lc| lc,
            |lc| lc,
        );
    }

    println!("Prover data A:");
    for &item in prover.a.iter() {
        println!("{:?}", item);
    }
    println!("Prover data B:");
    for &item in prover.b.iter() {
        println!("{:?}", item);
    }
    println!("Prover data C:");
    for &item in prover.c.iter() {
        println!("{:?}", item);
    }

    let worker = Worker::new();

    let vk = params.get_vk(prover.input_assignment.len())?;

    let now_fft = Instant::now();
    let now_exp;

    let h = {
        let mut a = EvaluationDomain::from_coeffs(prover.a)?;
        let mut b = EvaluationDomain::from_coeffs(prover.b)?;
        let mut c = EvaluationDomain::from_coeffs(prover.c)?;
        a.ifft(&worker);
        a.coset_fft(&worker);
        b.ifft(&worker);
        b.coset_fft(&worker);
        c.ifft(&worker);
        c.coset_fft(&worker);

        a.mul_assign(&worker, &b);
        drop(b);
        a.sub_assign(&worker, &c);
        drop(c);
        a.divide_by_z_on_coset(&worker);
        a.icoset_fft(&worker);
        let mut a = a.into_coeffs();
        let a_len = a.len() - 1;
        a.truncate(a_len);

        let end_fft = now_fft.elapsed();

        /*for &elem in a.iter() {
            println!("{}", elem.0.into_repr());
        }*/
        println!("{}",a_len);
        println!("Proof generation FFT took {}_{} microseconds.", end_fft.as_secs(), end_fft.subsec_micros());
        now_exp = Instant::now();

        // TODO: parallelize if it's even helpful
        let a = Arc::new(a.into_iter().map(|s| s.0.into_repr()).collect::<Vec<_>>());

        multiexp(&worker, params.get_h(a.len())?, FullDensity, a)
    };

    // TODO: parallelize if it's even helpful
    let input_assignment = Arc::new(prover.input_assignment.into_iter().map(|s| s.into_repr()).collect::<Vec<_>>());
    let aux_assignment = Arc::new(prover.aux_assignment.into_iter().map(|s| s.into_repr()).collect::<Vec<_>>());

    let l = multiexp(&worker, params.get_l(aux_assignment.len())?, FullDensity, aux_assignment.clone());

    let a_aux_density_total = prover.a_aux_density.get_total_density();

    let (a_inputs_source, a_aux_source) = params.get_a(input_assignment.len(), a_aux_density_total)?;

    let a_inputs = multiexp(&worker, a_inputs_source, FullDensity, input_assignment.clone());
    let a_aux = multiexp(&worker, a_aux_source, Arc::new(prover.a_aux_density), aux_assignment.clone());

    let b_input_density = Arc::new(prover.b_input_density);
    let b_input_density_total = b_input_density.get_total_density();
    let b_aux_density = Arc::new(prover.b_aux_density);
    let b_aux_density_total = b_aux_density.get_total_density();

    let (b_g1_inputs_source, b_g1_aux_source) = params.get_b_g1(b_input_density_total, b_aux_density_total)?;

    let b_g1_inputs = multiexp(&worker, b_g1_inputs_source, b_input_density.clone(), input_assignment.clone());
    let b_g1_aux = multiexp(&worker, b_g1_aux_source, b_aux_density.clone(), aux_assignment.clone());

    let (b_g2_inputs_source, b_g2_aux_source) = params.get_b_g2(b_input_density_total, b_aux_density_total)?;
    
    let b_g2_inputs = multiexp(&worker, b_g2_inputs_source, b_input_density, input_assignment);
    let b_g2_aux = multiexp(&worker, b_g2_aux_source, b_aux_density, aux_assignment);

    if vk.delta_g1.is_zero() || vk.delta_g2.is_zero() {
        // If this element is zero, someone is trying to perform a
        // subversion-CRS attack.
        return Err(SynthesisError::UnexpectedIdentity);
    }

    let mut g_a = vk.delta_g1.mul(r);
    g_a.add_assign_mixed(&vk.alpha_g1);
    let mut g_b = vk.delta_g2.mul(s);
    g_b.add_assign_mixed(&vk.beta_g2);
    let mut g_c;
    {
        let mut rs = r;
        rs.mul_assign(&s);

        g_c = vk.delta_g1.mul(rs);
        g_c.add_assign(&vk.alpha_g1.mul(s));
        g_c.add_assign(&vk.beta_g1.mul(r));
    }
    let mut a_answer = a_inputs.wait()?;
    a_answer.add_assign(&a_aux.wait()?);
    g_a.add_assign(&a_answer);
    a_answer.mul_assign(s);
    g_c.add_assign(&a_answer);

    let mut b1_answer = b_g1_inputs.wait()?;
    b1_answer.add_assign(&b_g1_aux.wait()?);
    let mut b2_answer = b_g2_inputs.wait()?;
    b2_answer.add_assign(&b_g2_aux.wait()?);

    g_b.add_assign(&b2_answer);
    b1_answer.mul_assign(r);
    g_c.add_assign(&b1_answer);
    g_c.add_assign(&h.wait()?);
    g_c.add_assign(&l.wait()?);

    let end_exp = now_exp.elapsed();
    println!("Proof generation exponentiation took {}_{} microseconds.", end_exp.as_secs(), end_exp.subsec_micros());

    Ok(Proof {
        a: g_a.into_affine(),
        b: g_b.into_affine(),
        c: g_c.into_affine()
    })
}


pub fn test_proof() {
    use std::io::BufReader;
    use std::io::BufRead;
    use std::fs::File;
    use std::mem::transmute;
    use std::time::Instant;

    use pairing::bls12_381::Bls12;

    type G1Affine = <Bls12 as Engine>::G1Affine;
    type G1Projective = <Bls12 as Engine>::G1;
    type G2Affine = <Bls12 as Engine>::G2Affine;
    type G2Projective = <Bls12 as Engine>::G2;
    type FrRepr = <<Bls12 as Engine>::Fr as PrimeField>::Repr;
    type Fr = <Bls12 as Engine>::Fr;
    type Fq = <Bls12 as Engine>::Fq;
    type Scalar = ::domain::Scalar<Bls12>;

    let file = File::open("./src/dump_all.txt").unwrap();
    let reader = BufReader::new(&file);

    let mut g1_u64 = [0u64; 18];
    let mut g2_u64 = [0u64; 36];
    let mut exponent_u64 = [0u64; 4];

    let mut a_vec = Vec::new();
    let mut b_vec = Vec::new();
    let mut c_vec = Vec::new();

    let mut h_exp = Vec::new();
    let mut h_points = Vec::new();

    let mut l_exp = Vec::new();
    let mut l_points = Vec::new();

    let mut a_inp_exp = Vec::new();
    let mut a_inp_points = Vec::new();

    let mut a_aux_exp = Vec::new();
    let mut a_aux_points = Vec::new();

    let mut b_inp_exp = Vec::new();
    let mut b_inp_points = Vec::new();

    let mut b_aux_exp = Vec::new();
    let mut b_aux_points = Vec::new();

    let mut b_inp_g2_exp = Vec::new();
    let mut b_inp_g2_points = Vec::new();

    let mut b_aux_g2_exp = Vec::new();
    let mut b_aux_g2_points = Vec::new();

    let fft_format_string = "Scalar(Fr(FrRepr([{d}, {d}, {d}, {d}])))";
    let g1_format_string = "G1 {{ x: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), y: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), z: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])) }} FrRepr([{d}, {d}, {d}, {d}])";
    let g2_format_string = "G2 {{ x: Fq2 {{ c0: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), c1: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])) }}, y: Fq2 {{ c0: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), c1: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])) }}, z: Fq2 {{ c0: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])), c1: Fq(FqRepr([{d}, {d}, {d}, {d}, {d}, {d}])) }} }} FrRepr([{d}, {d}, {d}, {d}])";
    for (num, line) in reader.lines().enumerate() {
        // FFT A Scalar
        if num >= 3 && num <= 98787 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, fft_format_string, u64, u64, u64, u64);

            exponent_u64[0] = temp.0.unwrap();
            exponent_u64[1] = temp.1.unwrap();
            exponent_u64[2] = temp.2.unwrap();
            exponent_u64[3] = temp.3.unwrap();

            unsafe {
                let exp: Scalar = transmute(exponent_u64);
                a_vec.push(exp);
            }
        // FFT B Scalar
        } else if num >= 98789 && num <= 197573 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, fft_format_string, u64, u64, u64, u64);

            exponent_u64[0] = temp.0.unwrap();
            exponent_u64[1] = temp.1.unwrap();
            exponent_u64[2] = temp.2.unwrap();
            exponent_u64[3] = temp.3.unwrap();

            unsafe {
                let exp: Scalar = transmute(exponent_u64);
                b_vec.push(exp);
            }
        // FFT C Scalar
        } else if num >= 197575 && num <= 296359 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, fft_format_string, u64, u64, u64, u64);

            exponent_u64[0] = temp.0.unwrap();
            exponent_u64[1] = temp.1.unwrap();
            exponent_u64[2] = temp.2.unwrap();
            exponent_u64[3] = temp.3.unwrap();

            unsafe {
                let exp: Scalar = transmute(exponent_u64);
                c_vec.push(exp);
            }
        // ME H G1
        } else if num >= 296362 && num <= 427432 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g1_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g1_u64[0] = temp.0.unwrap();
            g1_u64[1] = temp.1.unwrap();
            g1_u64[2] = temp.2.unwrap();
            g1_u64[3] = temp.3.unwrap();
            g1_u64[4] = temp.4.unwrap();
            g1_u64[5] = temp.5.unwrap();
            g1_u64[6] = temp.6.unwrap();
            g1_u64[7] = temp.7.unwrap();
            g1_u64[8] = temp.8.unwrap();
            g1_u64[9] = temp.9.unwrap();
            g1_u64[10] = temp.10.unwrap();
            g1_u64[11] = temp.11.unwrap();
            g1_u64[12] = temp.12.unwrap();
            g1_u64[13] = temp.13.unwrap();
            g1_u64[14] = temp.14.unwrap();
            g1_u64[15] = temp.15.unwrap();
            g1_u64[16] = temp.16.unwrap();
            g1_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(g1_u64);
                let exp: FrRepr = transmute(exponent_u64);

                h_points.push(proj.into_affine());
                h_exp.push(exp);
            }
        // ME L G1
        } else if num >= 427434 && num <= 526071 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g1_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g1_u64[0] = temp.0.unwrap();
            g1_u64[1] = temp.1.unwrap();
            g1_u64[2] = temp.2.unwrap();
            g1_u64[3] = temp.3.unwrap();
            g1_u64[4] = temp.4.unwrap();
            g1_u64[5] = temp.5.unwrap();
            g1_u64[6] = temp.6.unwrap();
            g1_u64[7] = temp.7.unwrap();
            g1_u64[8] = temp.8.unwrap();
            g1_u64[9] = temp.9.unwrap();
            g1_u64[10] = temp.10.unwrap();
            g1_u64[11] = temp.11.unwrap();
            g1_u64[12] = temp.12.unwrap();
            g1_u64[13] = temp.13.unwrap();
            g1_u64[14] = temp.14.unwrap();
            g1_u64[15] = temp.15.unwrap();
            g1_u64[16] = temp.16.unwrap();
            g1_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(g1_u64);
                let exp: FrRepr = transmute(exponent_u64);

                l_points.push(proj.into_affine());
                l_exp.push(exp);
            }
        // ME A INP G1
        } else if num >= 526073 && num <= 526080 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g1_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g1_u64[0] = temp.0.unwrap();
            g1_u64[1] = temp.1.unwrap();
            g1_u64[2] = temp.2.unwrap();
            g1_u64[3] = temp.3.unwrap();
            g1_u64[4] = temp.4.unwrap();
            g1_u64[5] = temp.5.unwrap();
            g1_u64[6] = temp.6.unwrap();
            g1_u64[7] = temp.7.unwrap();
            g1_u64[8] = temp.8.unwrap();
            g1_u64[9] = temp.9.unwrap();
            g1_u64[10] = temp.10.unwrap();
            g1_u64[11] = temp.11.unwrap();
            g1_u64[12] = temp.12.unwrap();
            g1_u64[13] = temp.13.unwrap();
            g1_u64[14] = temp.14.unwrap();
            g1_u64[15] = temp.15.unwrap();
            g1_u64[16] = temp.16.unwrap();
            g1_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(g1_u64);
                let exp: FrRepr = transmute(exponent_u64);

                a_inp_points.push(proj.into_affine());
                a_inp_exp.push(exp);
            }
        // ME A AUX G1
        } else if num >= 526082 && num <= 611463 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g1_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g1_u64[0] = temp.0.unwrap();
            g1_u64[1] = temp.1.unwrap();
            g1_u64[2] = temp.2.unwrap();
            g1_u64[3] = temp.3.unwrap();
            g1_u64[4] = temp.4.unwrap();
            g1_u64[5] = temp.5.unwrap();
            g1_u64[6] = temp.6.unwrap();
            g1_u64[7] = temp.7.unwrap();
            g1_u64[8] = temp.8.unwrap();
            g1_u64[9] = temp.9.unwrap();
            g1_u64[10] = temp.10.unwrap();
            g1_u64[11] = temp.11.unwrap();
            g1_u64[12] = temp.12.unwrap();
            g1_u64[13] = temp.13.unwrap();
            g1_u64[14] = temp.14.unwrap();
            g1_u64[15] = temp.15.unwrap();
            g1_u64[16] = temp.16.unwrap();
            g1_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(g1_u64);
                let exp: FrRepr = transmute(exponent_u64);

                a_aux_points.push(proj.into_affine());
                a_aux_exp.push(exp);
            }
        // ME B G1 INPUT
        } else if num >= 611465 && num <= 611465 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g1_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g1_u64[0] = temp.0.unwrap();
            g1_u64[1] = temp.1.unwrap();
            g1_u64[2] = temp.2.unwrap();
            g1_u64[3] = temp.3.unwrap();
            g1_u64[4] = temp.4.unwrap();
            g1_u64[5] = temp.5.unwrap();
            g1_u64[6] = temp.6.unwrap();
            g1_u64[7] = temp.7.unwrap();
            g1_u64[8] = temp.8.unwrap();
            g1_u64[9] = temp.9.unwrap();
            g1_u64[10] = temp.10.unwrap();
            g1_u64[11] = temp.11.unwrap();
            g1_u64[12] = temp.12.unwrap();
            g1_u64[13] = temp.13.unwrap();
            g1_u64[14] = temp.14.unwrap();
            g1_u64[15] = temp.15.unwrap();
            g1_u64[16] = temp.16.unwrap();
            g1_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(g1_u64);
                let exp: FrRepr = transmute(exponent_u64);

                b_inp_points.push(proj.into_affine());
                b_inp_exp.push(exp);
            }
        // ME B G1 AUX
        } else if num >= 611467 && num <= 672765{
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g1_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g1_u64[0] = temp.0.unwrap();
            g1_u64[1] = temp.1.unwrap();
            g1_u64[2] = temp.2.unwrap();
            g1_u64[3] = temp.3.unwrap();
            g1_u64[4] = temp.4.unwrap();
            g1_u64[5] = temp.5.unwrap();
            g1_u64[6] = temp.6.unwrap();
            g1_u64[7] = temp.7.unwrap();
            g1_u64[8] = temp.8.unwrap();
            g1_u64[9] = temp.9.unwrap();
            g1_u64[10] = temp.10.unwrap();
            g1_u64[11] = temp.11.unwrap();
            g1_u64[12] = temp.12.unwrap();
            g1_u64[13] = temp.13.unwrap();
            g1_u64[14] = temp.14.unwrap();
            g1_u64[15] = temp.15.unwrap();
            g1_u64[16] = temp.16.unwrap();
            g1_u64[17] = temp.17.unwrap();

            exponent_u64[0] = temp.18.unwrap();
            exponent_u64[1] = temp.19.unwrap();
            exponent_u64[2] = temp.20.unwrap();
            exponent_u64[3] = temp.21.unwrap();
            
            unsafe {
                let proj: G1Projective = transmute(g1_u64);
                let exp: FrRepr = transmute(exponent_u64);

                b_aux_points.push(proj.into_affine());
                b_aux_exp.push(exp);
            }
        // ME B G2 INPUT
        } else if num >= 672767 && num <= 672767 {
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g2_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

            g2_u64[0] = temp.0.unwrap();
            g2_u64[1] = temp.1.unwrap();
            g2_u64[2] = temp.2.unwrap();
            g2_u64[3] = temp.3.unwrap();
            g2_u64[4] = temp.4.unwrap();
            g2_u64[5] = temp.5.unwrap();
            g2_u64[6] = temp.6.unwrap();
            g2_u64[7] = temp.7.unwrap();
            g2_u64[8] = temp.8.unwrap();
            g2_u64[9] = temp.9.unwrap();
            g2_u64[10] = temp.10.unwrap();
            g2_u64[11] = temp.11.unwrap();
            g2_u64[12] = temp.12.unwrap();
            g2_u64[13] = temp.13.unwrap();
            g2_u64[14] = temp.14.unwrap();
            g2_u64[15] = temp.15.unwrap();
            g2_u64[16] = temp.16.unwrap();
            g2_u64[17] = temp.17.unwrap();
            g2_u64[18] = temp.18.unwrap();
            g2_u64[19] = temp.19.unwrap();
            g2_u64[20] = temp.20.unwrap();
            g2_u64[21] = temp.21.unwrap();
            g2_u64[22] = temp.22.unwrap();
            g2_u64[23] = temp.23.unwrap();
            g2_u64[24] = temp.24.unwrap();
            g2_u64[25] = temp.25.unwrap();
            g2_u64[26] = temp.26.unwrap();
            g2_u64[27] = temp.27.unwrap();
            g2_u64[28] = temp.28.unwrap();
            g2_u64[29] = temp.29.unwrap();
            g2_u64[30] = temp.30.unwrap();
            g2_u64[31] = temp.31.unwrap();
            g2_u64[32] = temp.32.unwrap();
            g2_u64[33] = temp.33.unwrap();
            g2_u64[34] = temp.34.unwrap();
            g2_u64[35] = temp.35.unwrap();

            exponent_u64[0] = temp.36.unwrap();
            exponent_u64[1] = temp.37.unwrap();
            exponent_u64[2] = temp.38.unwrap();
            exponent_u64[3] = temp.39.unwrap();
            
            unsafe {
                let proj: G2Projective = transmute(g2_u64);
                let exp: FrRepr = transmute(exponent_u64);

                b_inp_g2_points.push(proj.into_affine());
                b_inp_g2_exp.push(exp);
            }
        // ME B G2 AUX
        } else if num >= 672769 && num <= 734067{
            let line = line.unwrap();
            let temp = scan_fmt!(&line, g2_format_string, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);
            
            g2_u64[0] = temp.0.unwrap();
            g2_u64[1] = temp.1.unwrap();
            g2_u64[2] = temp.2.unwrap();
            g2_u64[3] = temp.3.unwrap();
            g2_u64[4] = temp.4.unwrap();
            g2_u64[5] = temp.5.unwrap();
            g2_u64[6] = temp.6.unwrap();
            g2_u64[7] = temp.7.unwrap();
            g2_u64[8] = temp.8.unwrap();
            g2_u64[9] = temp.9.unwrap();
            g2_u64[10] = temp.10.unwrap();
            g2_u64[11] = temp.11.unwrap();
            g2_u64[12] = temp.12.unwrap();
            g2_u64[13] = temp.13.unwrap();
            g2_u64[14] = temp.14.unwrap();
            g2_u64[15] = temp.15.unwrap();
            g2_u64[16] = temp.16.unwrap();
            g2_u64[17] = temp.17.unwrap();
            g2_u64[18] = temp.18.unwrap();
            g2_u64[19] = temp.19.unwrap();
            g2_u64[20] = temp.20.unwrap();
            g2_u64[21] = temp.21.unwrap();
            g2_u64[22] = temp.22.unwrap();
            g2_u64[23] = temp.23.unwrap();
            g2_u64[24] = temp.24.unwrap();
            g2_u64[25] = temp.25.unwrap();
            g2_u64[26] = temp.26.unwrap();
            g2_u64[27] = temp.27.unwrap();
            g2_u64[28] = temp.28.unwrap();
            g2_u64[29] = temp.29.unwrap();
            g2_u64[30] = temp.30.unwrap();
            g2_u64[31] = temp.31.unwrap();
            g2_u64[32] = temp.32.unwrap();
            g2_u64[33] = temp.33.unwrap();
            g2_u64[34] = temp.34.unwrap();
            g2_u64[35] = temp.35.unwrap();

            exponent_u64[0] = temp.36.unwrap();
            exponent_u64[1] = temp.37.unwrap();
            exponent_u64[2] = temp.38.unwrap();
            exponent_u64[3] = temp.39.unwrap();
            
            unsafe {
                let proj: G2Projective = transmute(g2_u64);
                let exp: FrRepr = transmute(exponent_u64);

                b_aux_g2_points.push(proj.into_affine());
                b_aux_g2_exp.push(exp);
            }
        } else if num >=734068 {
            break;
        }       
    }
    for _ in 0..1000 {
        let mut a_vec_l = a_vec.clone();
        let mut b_vec_l = b_vec.clone();
        let mut c_vec_l = c_vec.clone();

        let mut h_exp_l = h_exp.clone();
        let mut h_points_l = h_points.clone();

        let mut l_exp_l = l_exp.clone();
        let mut l_points_l = l_points.clone();

        let mut a_inp_exp_l = a_inp_exp.clone();
        let mut a_inp_points_l = a_inp_points.clone();

        let mut a_aux_exp_l = a_aux_exp.clone();
        let mut a_aux_points_l = a_aux_points.clone();

        let mut b_inp_exp_l = b_inp_exp.clone();
        let mut b_inp_points_l = b_inp_points.clone();

        let mut b_aux_exp_l = b_aux_exp.clone();
        let mut b_aux_points_l = b_aux_points.clone();

        let mut b_inp_g2_exp_l = b_inp_g2_exp.clone();
        let mut b_inp_g2_points_l = b_inp_g2_points.clone();

        let mut b_aux_g2_exp_l = b_aux_g2_exp.clone();
        let mut b_aux_g2_points_l = b_aux_g2_points.clone();


        let now = Instant::now();
        let worker = Worker::new();

        let mut a = EvaluationDomain::from_coeffs(a_vec_l).unwrap();
        let mut b = EvaluationDomain::from_coeffs(b_vec_l).unwrap();
        let mut c = EvaluationDomain::from_coeffs(c_vec_l).unwrap();
        a.ifft(&worker);
        a.coset_fft(&worker);
        b.ifft(&worker);
        b.coset_fft(&worker);
        c.ifft(&worker);
        c.coset_fft(&worker);

        a.mul_assign(&worker, &b);
        drop(b);
        a.sub_assign(&worker, &c);
        drop(c);
        a.divide_by_z_on_coset(&worker);
        a.icoset_fft(&worker);
        let mut a = a.into_coeffs();
        let a_len = a.len() - 1;
        a.truncate(a_len);
        
        let r1 = multiexp(&worker, (Arc::new(h_points_l), 0), FullDensity, Arc::new(h_exp_l));
        let r2 = multiexp(&worker, (Arc::new(l_points_l), 0), FullDensity, Arc::new(l_exp_l));
        let r3 = multiexp(&worker, (Arc::new(a_inp_points_l), 0), FullDensity, Arc::new(a_inp_exp_l));
        let r4 = multiexp(&worker, (Arc::new(a_aux_points_l), 0), FullDensity, Arc::new(a_aux_exp_l));
        let r5 = multiexp(&worker, (Arc::new(b_inp_points_l), 0), FullDensity, Arc::new(b_inp_exp_l));
        let r6 = multiexp(&worker, (Arc::new(b_aux_points_l), 0), FullDensity, Arc::new(b_aux_exp_l));
        let r7 = multiexp(&worker, (Arc::new(b_inp_g2_points_l), 0), FullDensity, Arc::new(b_inp_g2_exp_l));
        let r8 = multiexp(&worker, (Arc::new(b_aux_g2_points_l), 0), FullDensity, Arc::new(b_aux_g2_exp_l));

        r1.wait().unwrap();
        r2.wait().unwrap();
        r3.wait().unwrap();
        r4.wait().unwrap();
        r5.wait().unwrap();
        r6.wait().unwrap();
        r7.wait().unwrap();
        r8.wait().unwrap();

        let duration = now.elapsed();
        println!("{}.{:06}, ", duration.as_secs(), duration.subsec_micros());
    }
}
