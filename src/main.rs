//! This is the result of prototyping a domain specific compressor for CNF files. This code was not
//! written for maintainability and needs quite a bit of clean up and documentation.
use std::{
    convert::TryInto,
    io::{self, Read, Write},
    path::PathBuf,
};

#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd};
#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, FromRawHandle};

use flussab_cnf::cnf;
use structopt::StructOpt;

const BUF_SIZE: usize = 1024 * 128;

const MAGIC: &[u8; 9] = b"\x07cnfpack\n";

type Result<T, E = Box<dyn std::error::Error>> = std::result::Result<T, E>;

/// Encoder and decoder for the Cnfpack format.
///
/// Converts between the text based DIMACS CNF file format and the compressed binary Cnfpack format.
#[derive(StructOpt, Debug)]
struct Opt {
    /// Read a Cnfpack file and write a DIMACS CNF file. Default for the `uncnfpack` program.
    #[structopt(short, long, conflicts_with = "normalize", conflicts_with = "encode")]
    decode: bool,

    /// Read a DIMACS CNF file and write a Cnfpack file. Default for the `cnfpack` program.
    #[structopt(short, long, conflicts_with = "normalize", conflicts_with = "decode")]
    encode: bool,

    /// Read a DIMACS CNF file and write a normalized DIMACS CNF file.
    ///
    /// This produces the same output as encoding followed by decoding. The output is a minimal size
    /// CNF file where every clause, including the final one is terminated by a `0` and a newline.
    #[structopt(short, long)]
    normalize: bool,

    /// Input file to use.
    ///
    /// Reads from stdin by default or when specifying `-`.
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,

    /// Output file to use.
    ///
    /// Writes to stdout by default or when specifying `-`.
    #[structopt(parse(from_os_str))]
    output: Option<PathBuf>,

    /// Zstd compression level (1-22).
    #[structopt(short, long, default_value = "19")]
    zstd_level: u32,

    /// Delta threshold (0-100).
    ///
    /// Heuristic used to select between delta and direct encoding of literals. A value of `0`
    /// disables delta encoding, a value of `100` forces delta encoding.
    #[structopt(short = "t", long, default_value = "80")]
    delta_threshold: u32,

    /// Delta window (2-256).
    ///
    /// Maximum between literals that are delta encoded. Smaller values are faster, some inputs
    /// require larger values to compress well. For some inputs though, smaller values do result in
    /// slightly smaller files.
    #[structopt(short = "w", long, default_value = "256")]
    delta_window: u32,
}

struct EncodeOptions {
    zstd_level: u32,
    delta_threshold: u32,
    delta_window: u32,
}

fn main() {
    match main_err() {
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("error: {}", err);
            std::process::exit(1);
        }
    }
}

fn main_err() -> Result<i32, Box<dyn std::error::Error>> {
    let mut opt = Opt::from_args();

    if !opt.encode && !opt.decode && !opt.normalize {
        if let Some(progname) = std::env::args_os()
            .next()
            .map(PathBuf::from)
            .as_ref()
            .and_then(|progname| progname.file_name())
        {
            if progname.to_string_lossy().starts_with("un") {
                opt.decode = true;
            } else {
                opt.encode = true;
            }
        }
    }

    if opt
        .input
        .as_ref()
        .and_then(|path| path.as_os_str().to_str())
        == Some("-")
    {
        opt.input = None
    }

    if opt
        .output
        .as_ref()
        .and_then(|path| path.as_os_str().to_str())
        == Some("-")
    {
        opt.output = None
    }

    let mut input_tty = false;
    let input = if let Some(input) = &opt.input {
        Box::new(std::fs::File::open(input)?)
    } else {
        input_tty = atty::is(atty::Stream::Stdin);
        #[cfg(unix)]
        unsafe {
            Box::new(std::fs::File::from_raw_fd(std::io::stdin().as_raw_fd()))
        }
        #[cfg(windows)]
        unsafe {
            Box::new(std::fs::File::from_raw_handle(
                std::io::stdin().as_raw_handle(),
            ))
        }
    };

    let mut output_tty = false;
    let output = if let Some(output) = &opt.output {
        Box::new(std::fs::File::create(output)?)
    } else {
        output_tty = atty::is(atty::Stream::Stdout);
        #[cfg(unix)]
        unsafe {
            Box::new(std::fs::File::from_raw_fd(std::io::stdout().as_raw_fd()))
        }
        #[cfg(windows)]
        unsafe {
            Box::new(std::fs::File::from_raw_handle(
                std::io::stdout().as_raw_handle(),
            ))
        }
    };

    if input_tty && output_tty && std::env::args_os().len() == 1 {
        Opt::clap().print_help()?;
        println!();
        return Ok(1);
    }

    if opt.normalize {
        normalize(input, output)?;
    } else if opt.decode {
        if input_tty {
            return Err("cannot read binary data from the console".into());
        }
        decode(input, output)?;
    } else if opt.encode {
        if output_tty {
            return Err("will not output binary data to the console".into());
        }
        encode(
            input,
            output,
            EncodeOptions {
                zstd_level: opt.zstd_level,
                delta_threshold: opt.delta_threshold,
                delta_window: opt.delta_window,
            },
        )?;
    } else {
        unreachable!()
    }

    Ok(0)
}

pub struct ParsedCnf {
    clause_len: Vec<u8>,
    long_clause_len: Vec<u32>,
    lits: Vec<u32>,
    var_count: u32,
}

impl ParsedCnf {
    pub fn new(input: Box<dyn Read>) -> Result<Self> {
        let mut parser = cnf::Parser::<i32>::from_boxed_dyn_read(input, true)?;
        let mut clause_len: Vec<u8> = vec![];
        let mut long_clause_len: Vec<u32> = vec![];
        let mut lits: Vec<u32> = vec![];

        let mut var_count: u32 = parser
            .header()
            .map(|header| header.var_count as u32)
            .unwrap_or_default();

        while let Some(clause) = parser.next_clause()? {
            let len = clause.len();
            clause_len.push(len.min(255) as u8);
            if len >= 255 {
                if len > u32::MAX as usize {
                    return Err("clause longer than supported".into());
                }
                long_clause_len.push(len as u32);
            }

            for &lit in clause {
                let code = code_from_lit(lit);
                var_count = var_count.max((code >> 1) + 1);

                lits.push(code);
            }
        }

        Ok(Self {
            clause_len,
            long_clause_len,
            lits,
            var_count,
        })
    }
}

fn normalize(input: Box<dyn Read>, mut output: Box<dyn Write>) -> Result<()> {
    let ParsedCnf {
        clause_len,
        long_clause_len,
        lits,
        var_count,
    } = ParsedCnf::new(input)?;

    let mut long_clause_len = long_clause_len.into_iter();

    let mut lit_pos = 0;
    let mut output_buf = vec![];

    cnf::write_header(
        &mut output_buf,
        cnf::Header {
            var_count: var_count as usize,
            clause_count: clause_len.len(),
        },
    )?;

    for short_len in clause_len {
        let len = if short_len == 255 {
            long_clause_len.next().unwrap() as usize
        } else {
            short_len as usize
        };

        for _ in 0..len {
            let code = lits[lit_pos];
            lit_pos += 1;

            let lit = lit_from_code(code);
            itoap::write_to_vec(&mut output_buf, lit);
            output_buf.push(b' ');
            if output_buf.len() > BUF_SIZE {
                output.write_all(&output_buf)?;
                output_buf.clear();
            }
        }
        output_buf.extend(b"0\n");
    }

    output.write_all(output_buf.drain(..).as_slice())?;

    output.flush()?;

    Ok(())
}

fn u_from_s(value: i32) -> u32 {
    (value.wrapping_shl(1) ^ (value >> 31)) as u32
}

fn s_from_u(value: u32) -> i32 {
    let rot = value.wrapping_shl(31) as i32;
    (value >> 1) as i32 ^ (rot >> 31)
}

fn code_from_lit(lit: i32) -> u32 {
    u_from_s(lit) - 1
}

fn lit_from_code(code: u32) -> i32 {
    s_from_u(code + 1)
}

#[cfg(test)]
#[test]
fn test_lit_code() {
    assert_eq!(code_from_lit(1), 1);
    assert_eq!(code_from_lit(-1), 0);
    assert_eq!(code_from_lit(2), 3);
    assert_eq!(code_from_lit(-2), 2);

    for i in -100..=100 {
        if i == 0 {
            continue;
        }
        assert_eq!(lit_from_code(code_from_lit(i)), i);
    }

    for i in -100..=100 {
        assert_eq!(s_from_u(u_from_s(i)), i);
    }

    for i in i32::MIN..i32::MIN + 100 {
        assert_eq!(s_from_u(u_from_s(i)), i);
    }

    for i in i32::MAX - 100..=i32::MAX {
        assert_eq!(s_from_u(u_from_s(i)), i);
    }
}

fn encode_ints(values: &mut [u32], max_value: u32) -> &[u8] {
    if max_value < (1 << 8) {
        unsafe {
            let ptr = values.as_mut_ptr();
            let mut write = ptr as *mut u8;
            let mut read = ptr;

            for _ in 0..values.len() {
                *write = *read as u8;
                write = write.add(1);
                read = read.add(1);
            }

            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len())
        }
    } else if max_value < (1 << 16) {
        unsafe {
            let ptr = values.as_mut_ptr();
            let mut write = ptr as *mut u16;
            let mut read = ptr;

            for _ in 0..values.len() {
                *write = (*read).to_le() as u16;
                write = write.add(1);
                read = read.add(1);
            }

            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 2)
        }
    } else if max_value < (1 << 24) {
        unsafe {
            let ptr = values.as_mut_ptr();
            let mut write = ptr as *mut u8;
            let mut read = ptr;

            for _ in 0..values.len() {
                let [a, b, c, _] = (*read).to_le_bytes();
                *write = a;
                *write.add(1) = b;
                *write.add(2) = c;
                write = write.add(3);
                read = read.add(1);
            }

            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 3)
        }
    } else {
        for value in &mut *values {
            *value = value.to_le();
        }

        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }
}

fn encode(input: Box<dyn Read>, mut output: Box<dyn Write>, options: EncodeOptions) -> Result<()> {
    let ParsedCnf {
        clause_len,
        mut long_clause_len,
        mut lits,
        var_count,
    } = ParsedCnf::new(input)?;

    output.write_all(MAGIC)?;

    let mut output = zstd::Encoder::new(output, options.zstd_level as i32)?;

    output.write_all(&[0])?;

    output.write_all(&var_count.to_le_bytes())?;
    output.write_all(&(clause_len.len() as u64).to_le_bytes())?;
    output.write_all(&clause_len)?;
    drop(clause_len);
    output.write_all(encode_ints(&mut long_clause_len, u32::MAX))?;

    let max_lit_code = (var_count - 1) * 2 + 1;

    let shift = if max_lit_code < (1 << 8) {
        24
    } else if max_lit_code < (1 << 16) {
        16
    } else if max_lit_code < (1 << 24) {
        8
    } else {
        0
    };

    let delta_len = if options.delta_threshold == 0 {
        0
    } else {
        lits.len()
    };

    let mut offsets = vec![255u8; delta_len];
    let mut deltas = vec![u32::MAX; delta_len];

    for i in 0..delta_len.min(256) {
        offsets[i] = 255;
        deltas[i] = u_from_s((lits[i] << shift) as i32 >> shift);
    }

    const BLOCK: usize = 4;

    let window = options.delta_window.clamp(2, 256) as usize;

    for k0 in 0..(delta_len + BLOCK - 1) / BLOCK {
        let k_off = k0 * BLOCK;

        if k_off < 256 || k_off + BLOCK > delta_len {
            for i in (0..window).rev() {
                for k1 in 0..BLOCK {
                    let k = k_off + k1;
                    if k < i + 1 || k >= delta_len {
                        continue;
                    }
                    let a = lits[k - i - 1];
                    let b = lits[k];
                    let delta = u_from_s((b.wrapping_sub(a) << shift) as i32 >> shift);
                    if deltas[k] >= delta {
                        deltas[k] = delta;
                        offsets[k] = i as u8;
                    }
                }
            }
        } else {
            #[cfg(not(target_arch = "x86_64"))]
            for i in (0..window).rev() {
                for k1 in 0..BLOCK {
                    let k = k_off + k1;
                    let a = lits[k - i - 1];
                    let b = lits[k];
                    let delta = u_from_s((b.wrapping_sub(a) << shift) as i32 >> shift);
                    if deltas[k] >= delta {
                        deltas[k] = delta;
                        offsets[k] = i as u8;
                    }
                }
            }

            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{
                    _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_cmplt_epi32,
                    _mm_cvtsi128_si32, _mm_loadu_si128, _mm_or_si128, _mm_packs_epi16,
                    _mm_packs_epi32, _mm_set1_epi32, _mm_set_epi64x, _mm_sll_epi32, _mm_slli_epi32,
                    _mm_sra_epi32, _mm_srai_epi32, _mm_storeu_si128, _mm_sub_epi32, _mm_xor_si128,
                };

                let lits_ptr = lits.as_ptr();
                let delta_ptr = deltas.as_mut_ptr();
                let offsets_ptr = offsets.as_mut_ptr();
                let shift_vec = _mm_set_epi64x(0, shift as _);
                let sign_offset = _mm_set1_epi32(i32::MIN);

                let mut current_delta_vec = _mm_loadu_si128(delta_ptr.add(k_off) as *const _);
                let mut current_offsets =
                    u32::from_le_bytes(*(offsets_ptr.add(k_off) as *const [u8; 4]));

                let b_vec = _mm_loadu_si128(lits_ptr.add(k_off) as *const _);

                for i in (0..window).rev() {
                    let a_vec = _mm_loadu_si128(lits_ptr.add(k_off - i - 1) as *const _);

                    let delta_s_vec = _mm_sra_epi32(
                        _mm_sll_epi32(_mm_sub_epi32(b_vec, a_vec), shift_vec),
                        shift_vec,
                    );

                    let delta_vec = _mm_xor_si128(
                        _mm_srai_epi32::<31>(delta_s_vec),
                        _mm_slli_epi32::<1>(delta_s_vec),
                    );

                    let delta_offset_vec = _mm_add_epi32(delta_vec, sign_offset);
                    let current_delta_offset_vec = _mm_add_epi32(current_delta_vec, sign_offset);

                    let compare = _mm_cmplt_epi32(current_delta_offset_vec, delta_offset_vec);

                    current_delta_vec = _mm_or_si128(
                        _mm_and_si128(compare, current_delta_vec),
                        _mm_andnot_si128(compare, delta_vec),
                    );

                    let tmp = _mm_packs_epi32(compare, compare);

                    let compare_bytes = _mm_cvtsi128_si32(_mm_packs_epi16(tmp, tmp)) as u32;

                    let offset_word = i as u32 * 0x01010101;

                    current_offsets =
                        (current_offsets & compare_bytes) | (offset_word & !compare_bytes);
                }

                _mm_storeu_si128(delta_ptr.add(k_off) as *mut _, current_delta_vec);
                *(offsets_ptr.add(k_off) as *mut [u8; 4]) = current_offsets.to_le_bytes();
            }
        }
    }

    let mut hist = [0usize; 256];

    for &offset in &offsets {
        hist[offset as usize] += 1;
    }

    let mut entropy = 0.0;

    let total = offsets.len() as f32;

    for &count in &hist {
        if count > 0 {
            let prob = count as f32 / total;
            entropy -= prob * prob.log2();
        }
    }

    entropy /= (window as f32).log2();

    if entropy >= (options.delta_threshold as f32) / 100.0 {
        output.write_all(&[0])?;
        output.write_all(encode_ints(&mut lits, max_lit_code))?;
    } else {
        output.write_all(&[1])?;
        output.write_all(&offsets)?;
        output.write_all(encode_ints(&mut deltas, max_lit_code))?;
    }

    output.finish()?;

    Ok(())
}

fn decode_u64(read: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    read.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn decode_u32(read: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    read.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn decode_ints(read: &mut impl Read, len: usize, max_value: u32) -> io::Result<Vec<u32>> {
    let mut values = vec![0u32; len];
    if max_value < (1 << 8) {
        unsafe {
            read.read_exact(std::slice::from_raw_parts_mut(
                (values.as_mut_ptr() as *mut u8).add(len * 3),
                len,
            ))?;

            let ptr = values.as_mut_ptr();
            let mut write = ptr as *mut u32;
            let mut read = (ptr as *mut u8).add(len * 3);

            for _ in 0..len {
                *write = *read as u32;
                write = write.add(1);
                read = read.add(1);
            }
        }
    } else if max_value < (1 << 16) {
        unsafe {
            read.read_exact(std::slice::from_raw_parts_mut(
                (values.as_mut_ptr() as *mut u8).add(len * 2),
                len * 2,
            ))?;

            let ptr = values.as_mut_ptr();
            let mut write = ptr as *mut u32;
            let mut read = (ptr as *mut u16).add(len);

            for _ in 0..len {
                *write = *read as u32;
                write = write.add(1);
                read = read.add(1);
            }
        }
    } else if max_value < (1 << 24) {
        unsafe {
            read.read_exact(std::slice::from_raw_parts_mut(
                (values.as_mut_ptr() as *mut u8).add(len),
                len * 3,
            ))?;

            let ptr = values.as_mut_ptr();
            let mut write = ptr as *mut u32;
            let mut read = (ptr as *mut u8).add(len);

            for _ in 0..len {
                let a = *read;
                let b = *read.add(1);
                let c = *read.add(2);

                *write = u32::from_le_bytes([a, b, c, 0]);
                write = write.add(1);
                read = read.add(3);
            }
        }
    } else {
        unsafe {
            read.read_exact(std::slice::from_raw_parts_mut(
                values.as_mut_ptr() as *mut u8,
                values.len() * 4,
            ))?;
        }

        for value in &mut values {
            *value = value.to_le();
        }
    }
    Ok(values)
}

fn decode(mut input: Box<dyn Read>, mut output: Box<dyn Write>) -> Result<()> {
    let mut magic = *MAGIC;

    input.read_exact(&mut magic)?;

    if magic != *MAGIC {
        return Err("input file not in cnfpack format".into());
    }

    let mut input = zstd::Decoder::new(input)?;

    let mut version = [0];

    input.read_exact(&mut version)?;

    if version != [0] {
        return Err(format!("input file has unsupported cnfpack version {}", version[0]).into());
    }

    let mut output_buf = Vec::<u8>::with_capacity(BUF_SIZE + 128);

    let var_count = decode_u32(&mut input)?;
    let clause_count: usize = decode_u64(&mut input)?.try_into()?;

    let mut clause_len = vec![0u8; clause_count];

    input.read_exact(&mut clause_len)?;

    let long_clause_count: usize = clause_len.iter().map(|&len| (len == 255) as usize).sum();

    let mut lit_count: usize = clause_len.iter().map(|&len| len as usize).sum();

    let long_clause_len = decode_ints(&mut input, long_clause_count, u32::MAX)?;

    lit_count += long_clause_len
        .iter()
        .map(|&len| len as usize - 255)
        .sum::<usize>();

    let mut mode = [0];
    input.read_exact(&mut mode)?;
    let mode = mode[0];

    if mode >= 2 {
        return Err(format!("input file has unsupported literal encoding {}", mode).into());
    }

    let mut offsets = vec![0u8; lit_count];

    if mode == 1 {
        input.read_exact(&mut offsets)?;
    }

    let max_lit_code = (var_count - 1) * 2 + 1;

    let shift = if max_lit_code < (1 << 8) {
        24
    } else if max_lit_code < (1 << 16) {
        16
    } else if max_lit_code < (1 << 24) {
        8
    } else {
        0
    };

    let mask = u32::MAX >> shift;

    let mut deltas = decode_ints(&mut input, lit_count, max_lit_code)?;

    let mut lit_pos = 0;

    let mut long_clause_len = long_clause_len.into_iter();

    cnf::write_header(
        &mut output_buf,
        cnf::Header {
            var_count: var_count as usize,
            clause_count,
        },
    )?;

    for short_len in clause_len {
        let len = if short_len == 255 {
            long_clause_len.next().unwrap() as usize
        } else {
            short_len as usize
        };

        for _ in 0..len {
            let code;
            if mode == 1 {
                let offset = offsets[lit_pos] as usize;
                let delta = deltas[lit_pos];

                let base = if lit_pos < offset + 1 {
                    0
                } else {
                    deltas[lit_pos - 1 - offset]
                };

                code = base.wrapping_add(s_from_u(delta) as u32) & mask;

                deltas[lit_pos] = code;
            } else {
                code = deltas[lit_pos];
            }

            lit_pos += 1;

            let lit = lit_from_code(code);
            itoap::write_to_vec(&mut output_buf, lit);
            output_buf.push(b' ');
            if output_buf.len() > BUF_SIZE {
                output.write_all(&output_buf)?;
                output_buf.clear();
            }
        }
        output_buf.extend(b"0\n");
    }

    output.write_all(output_buf.drain(..).as_slice())?;

    output.flush()?;

    Ok(())
}
