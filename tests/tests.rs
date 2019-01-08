extern crate ocl;
extern crate pairing;
extern crate rand;

use ocl::{flags, Platform, Device, Context, Queue, Program,
Buffer, Kernel};
use rand::random;
use std::fs::read_to_string;

#[test]
fn test_sbb() {
    let dims = 100;

    let mut a = vec![0u64; dims];
    let mut b = vec![0u64; dims];

    let mut borrow = vec![0u64; dims];
    let mut borrow_gpu = vec![0u64; dims];

    let mut c = vec![0u64; dims];
    let mut c_gpu = vec![0u64; dims];

    for i in 0..dims {
        a[i] = random::<u64>();
        b[i] = random::<u64>();

        borrow[i] = (random::<u8>() as u64) & 0x01;
        borrow_gpu[i] = borrow[i];
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_b = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_borrow = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_WRITE)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_c = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a).enq().unwrap();
    buffer_b.cmd().write(&b).enq().unwrap();
    buffer_borrow.cmd().write(&borrow).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_sbb")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_borrow)
        .arg(&buffer_c)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_c.cmd().read(&mut c_gpu).enq().unwrap();
    buffer_borrow.cmd().read(&mut borrow_gpu).enq().unwrap();

    for i in 0..dims {
        c[i] = pairing::sbb(a[i], b[i], &mut borrow[i]);
        assert_eq!(c[i], c_gpu[i]);
        assert_eq!(borrow[i], borrow_gpu[i]);
    }
}

#[test]
fn test_adc() {
    let dims = 100;

    let mut a = vec![0u64; dims];
    let mut b = vec![0u64; dims];

    let mut carry = vec![0u64; dims];
    let mut carry_gpu = vec![0u64; dims];

    let mut c = vec![0u64; dims];
    let mut c_gpu = vec![0u64; dims];

    for i in 0..dims {
        a[i] = random::<u64>();
        b[i] = random::<u64>();

        carry[i] = (random::<u8>() as u64) & 0x01;
        carry_gpu[i] = carry[i];
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_b = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_carry = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_WRITE)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_c = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a).enq().unwrap();
    buffer_b.cmd().write(&b).enq().unwrap();
    buffer_carry.cmd().write(&carry_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_adc")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_carry)
        .arg(&buffer_c)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_c.cmd().read(&mut c_gpu).enq().unwrap();
    buffer_carry.cmd().read(&mut carry_gpu).enq().unwrap();

    for i in 0..dims {
        c[i] = pairing::adc(a[i], b[i], &mut carry[i]);
        assert_eq!(c[i], c_gpu[i]);
        assert_eq!(carry[i], carry_gpu[i]);
    }
}

#[test]
fn test_mac_with_carry() {
    let dims = 100;

    let mut a = vec![0u64; dims];
    let mut b = vec![0u64; dims];

    let mut carry = vec![0u64; dims];
    let mut carry_gpu = vec![0u64; dims];

    let mut c = vec![0u64; dims];

    let mut result = vec![0u64; dims];
    let mut result_gpu = vec![0u64; dims];

    for i in 0..dims {
        a[i] = random::<u64>();
        b[i] = random::<u64>();
        c[i] = random::<u64>();

        carry[i] = (random::<u8>() as u64) & 0x01;
        carry_gpu[i] = carry[i];
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_b = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_carry = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_WRITE)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_c = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    let buffer_result = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a).enq().unwrap();
    buffer_b.cmd().write(&b).enq().unwrap();
    buffer_c.cmd().write(&c).enq().unwrap();
    buffer_carry.cmd().write(&carry_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_mac_with_carry")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_c)
        .arg(&buffer_carry)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();
    buffer_carry.cmd().read(&mut carry_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = pairing::mac_with_carry(a[i], b[i], c[i], &mut carry[i]);
        assert_eq!(result[i], result_gpu[i]);
        assert_eq!(carry[i], carry_gpu[i]);
    }
}

#[test]
fn test_is_odd() {
    use pairing::bls12_381::*;
    use pairing::PrimeFieldRepr;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];

    for i in 0..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();


    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_is_odd")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = a[i].is_odd() as i8;
        assert_eq!(result[i], result_gpu[i]);
    }
}

#[test]
fn test_is_even() {
    use pairing::bls12_381::*;
    use pairing::PrimeFieldRepr;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];

    for i in 0..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();


    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_is_even")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = a[i].is_even() as i8;
        assert_eq!(result[i], result_gpu[i]);
    }
}

#[test]
fn test_is_zero() {
    use pairing::bls12_381::*;
    use pairing::PrimeFieldRepr;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];

    a.push(FrRepr::from(0));
    a_gpu.push(0);
    a_gpu.push(0);
    a_gpu.push(0);
    a_gpu.push(0);

    for i in 1..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();


    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_is_zero")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = a[i].is_zero() as i8;
        assert_eq!(result[i], result_gpu[i]);
    }
}

#[test]
fn test_cmp() {
    use pairing::bls12_381::*;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut b = Vec::new();
    let mut b_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];


    for i in 0..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);

        b.push(FrRepr::rand(&mut rng));

        b_gpu.push(b[i].0[0]);
        b_gpu.push(b[i].0[1]);
        b_gpu.push(b[i].0[2]);
        b_gpu.push(b[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();

    let buffer_b = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();


    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_cmp")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = match a.cmp(&b) {
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Greater => 1
        };
        assert_eq!(result[i], result_gpu[i]);
    }
}

#[test]
fn test_lt() {
    use pairing::bls12_381::*;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut b = Vec::new();
    let mut b_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];


    for i in 0..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);

        b.push(FrRepr::rand(&mut rng));

        b_gpu.push(b[i].0[0]);
        b_gpu.push(b[i].0[1]);
        b_gpu.push(b[i].0[2]);
        b_gpu.push(b[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();

    let buffer_b = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();


    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();
    buffer_b.cmd().write(&b_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_lt")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = a[i].lt(&b[i]) as i8;
        assert_eq!(result[i], result_gpu[i]);
    }
}

#[test]
fn test_eq() {
    use pairing::bls12_381::*;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut b = Vec::new();
    let mut b_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];


    for i in 0..dims/2 {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);

        b.push(FrRepr::rand(&mut rng));

        b_gpu.push(b[i].0[0]);
        b_gpu.push(b[i].0[1]);
        b_gpu.push(b[i].0[2]);
        b_gpu.push(b[i].0[3]);
    }

    for i in dims/2..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);

        b.push(a[i]);

        b_gpu.push(a[i].0[0]);
        b_gpu.push(a[i].0[1]);
        b_gpu.push(a[i].0[2]);
        b_gpu.push(a[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();

    let buffer_b = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();


    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();
    buffer_b.cmd().write(&b_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_eq")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = a[i].eq(&b[i]) as i8;
        assert_eq!(result[i], result_gpu[i]);
    }
}

#[test]
fn test_is_valid() {
    use pairing::bls12_381::*;
    use pairing::PrimeField;
    use rand::{Rand, SeedableRng, XorShiftRng};

    let mut rng = XorShiftRng::from_seed([0x5dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);
    let dims = 100;

    let mut a = Vec::new();
    let mut a_gpu = Vec::new();

    let mut result = vec![0i8; dims];
    let mut result_gpu = vec![0i8; dims];


    for i in 0..dims {
        a.push(FrRepr::rand(&mut rng));

        a_gpu.push(a[i].0[0]);
        a_gpu.push(a[i].0[1]);
        a_gpu.push(a[i].0[2]);
        a_gpu.push(a[i].0[3]);
    }


    let opencl_string = read_to_string("./src/primitives.cl").unwrap();

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
    

    let buffer_a = Buffer::<u64>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_READ_ONLY)
                                 .len(dims*4)
                                 .build().unwrap();



    let buffer_result = Buffer::<i8>::builder()
                                 .queue(queue.clone())
                                 .flags(flags::MEM_WRITE_ONLY)
                                 .len(dims)
                                 .build().unwrap();

    buffer_a.cmd().write(&a_gpu).enq().unwrap();

    let dims_gpu = dims as u32;

    let kernel = Kernel::builder()
        .program(&program)
        .name("test_is_valid")
        .global_work_size(dims)
        .arg(&buffer_a)
        .arg(&buffer_result)
        .arg(&dims_gpu)
        .build().unwrap();

    unsafe {
        kernel.cmd()
            .queue(&queue)
            .global_work_offset(kernel.default_global_work_offset())
            .global_work_size(dims)
            .local_work_size(10)
            .enq().unwrap();
    }

    buffer_result.cmd().read(&mut result_gpu).enq().unwrap();

    for i in 0..dims {
        result[i] = a[i].lt(&Fr::char()) as i8;
        assert_eq!(result[i], result_gpu[i]);
    }
}