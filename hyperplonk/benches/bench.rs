// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use std::{fs::File, io, time::Instant};

use ark_bls12_381::{Bls12_381, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Write};
use ark_std::test_rng;
use hyperplonk::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK, LogaHyperPlonkSNARK,
};
use subroutines::{
    pcs::{
        prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
        PolynomialCommitmentScheme,
    },
    poly_iop::PolyIOP,
};

const SUPPORTED_SIZE: usize = 16;
const MIN_NUM_VARS: usize = 5;
const MAX_NUM_VARS: usize = 15;
const MIN_CUSTOM_DEGREE: usize = 1;
const MAX_CUSTOM_DEGREE: usize = 32;
const HIGH_DEGREE_TEST_NV: usize = 10;

fn main() -> Result<(), HyperPlonkErrors> {
    let thread = rayon::current_num_threads();
    println!("start benchmark with #{} threads", thread);
    let mut rng = test_rng();
    let pcs_srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, SUPPORTED_SIZE)?;
    for arg in std::env::args() {
        if arg == "vanilla" {
            bench_vanilla_plonk(&pcs_srs, thread)?;
            println!();
        } else if arg == "jelly" {
            bench_jellyfish_plonk(&pcs_srs, thread)?;
            println!();
        } else if arg == "high" {
            for degree in MIN_CUSTOM_DEGREE..=MAX_CUSTOM_DEGREE {
                bench_high_degree_plonk(&pcs_srs, degree, thread)?;
                println!();
            }
            println!();
        }
    }

    Ok(())
}

fn _read_srs() -> Result<MultilinearUniversalParams<Bls12_381>, io::Error> {
    let mut f = File::open("srs.params")?;
    Ok(MultilinearUniversalParams::<Bls12_381>::deserialize_compressed_unchecked(&mut f).unwrap())
}

fn _write_srs(pcs_srs: &MultilinearUniversalParams<Bls12_381>) {
    let mut f = File::create("srs.params").unwrap();
    pcs_srs.serialize_uncompressed(&mut f).unwrap();
}

fn bench_vanilla_plonk(
    pcs_srs: &MultilinearUniversalParams<Bls12_381>,
    thread: usize,
) -> Result<(), HyperPlonkErrors> {
    let filename_loga = format!("logalk vanilla threads {}.txt", thread);
    let filename_plk = format!("plk vanilla threads {}.txt", thread);
    let mut file_loga = File::create(filename_loga).unwrap();
    let mut file_plk = File::create(filename_plk).unwrap();
    let vanilla_gate = CustomizedGates::vanilla_plonk_gate();
    for nv in MIN_NUM_VARS..=MAX_NUM_VARS {
        println!("Vanilla Plonk with {nv} variables:");
        bench_mock_circuit_zkp_helper_with_loga_lk(&mut file_loga, nv, &vanilla_gate, pcs_srs)?;
        bench_mock_circuit_zkp_helper_with_plk(&mut file_plk, nv, &vanilla_gate, pcs_srs)?;
    }

    Ok(())
}

fn bench_jellyfish_plonk(
    pcs_srs: &MultilinearUniversalParams<Bls12_381>,
    thread: usize,
) -> Result<(), HyperPlonkErrors> {
    let filename_loga = format!("logalk jellyfish threads {}.txt", thread);
    let filename_plk = format!("plk jellyfish threads {}.txt", thread);
    let mut file_loga = File::create(filename_loga).unwrap();
    let mut file_plk = File::create(filename_plk).unwrap();
    let jf_gate = CustomizedGates::jellyfish_turbo_plonk_gate();
    for nv in MIN_NUM_VARS..=MAX_NUM_VARS {
        println!("Jellyfish with {} variables", nv);
        bench_mock_circuit_zkp_helper_with_loga_lk(&mut file_loga, nv, &jf_gate, pcs_srs)?;
        bench_mock_circuit_zkp_helper_with_plk(&mut file_plk, nv, &jf_gate, pcs_srs)?;
    }

    Ok(())
}

fn bench_high_degree_plonk(
    pcs_srs: &MultilinearUniversalParams<Bls12_381>,
    degree: usize,
    thread: usize,
) -> Result<(), HyperPlonkErrors> {
    let filename_loga = format!("logalk high degree {} thread {}.txt", degree, thread);
    let filename_plk = format!("plk high degree {} thread {}.txt", degree, thread);
    let mut file_loga = File::create(filename_loga).unwrap();
    let mut file_plk = File::create(filename_plk).unwrap();

    let vanilla_gate = CustomizedGates::mock_gate(2, degree);
    println!("Custom gate of degree {}", degree);

    bench_mock_circuit_zkp_helper_with_loga_lk(
        &mut file_loga,
        HIGH_DEGREE_TEST_NV,
        &vanilla_gate,
        pcs_srs,
    )?;
    bench_mock_circuit_zkp_helper_with_plk(
        &mut file_plk,
        HIGH_DEGREE_TEST_NV,
        &vanilla_gate,
        pcs_srs,
    )?;

    Ok(())
}

fn bench_mock_circuit_zkp_helper_with_loga_lk(
    file: &mut File,
    nv: usize,
    gate: &CustomizedGates,
    pcs_srs: &MultilinearUniversalParams<Bls12_381>,
) -> Result<(), HyperPlonkErrors> {
    let repetition = if nv < 10 {
        5
    } else if nv < 20 {
        3
    } else {
        1
    };

    // LOGACIRCUIT AND LOGA SNARK
    //==========================================================
    let logacircuit = MockCircuit::<Fr>::new(1 << nv, gate, gate, true);
    assert!(logacircuit.is_satisfied());
    let index = logacircuit.index;
    //==========================================================
    // generate pk and vks
    let start = Instant::now();
    for _ in 0..(repetition - 1) {
        let (_pk, _vk) = <PolyIOP<Fr> as LogaHyperPlonkSNARK<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::preprocess(&index, pcs_srs)?;
    }
    let (pk, vk) =
        <PolyIOP<Fr> as LogaHyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::preprocess(
            &index, pcs_srs,
        )?;
    let key_extraction = start.elapsed().as_micros() / repetition as u128;
    //==========================================================
    // generate a proof
    let start = Instant::now();
    for _ in 0..(repetition - 1) {
        let _proof = <PolyIOP<Fr> as LogaHyperPlonkSNARK<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::prove(&pk, &logacircuit.public_inputs, &logacircuit.witnesses)?;
    }
    let proof =
        <PolyIOP<Fr> as LogaHyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::prove(
            &pk,
            &logacircuit.public_inputs,
            &logacircuit.witnesses,
        )?;
    let proving = start.elapsed().as_micros() / repetition as u128;
    file.write_all(format!("{} {}\n", nv, proving).as_ref())
        .unwrap();

    //==========================================================
    // verify a proof
    let start = Instant::now();
    for _ in 0..repetition {
        let verify = <PolyIOP<Fr> as LogaHyperPlonkSNARK<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::verify(&vk, &logacircuit.public_inputs, &proof)?;
        assert!(verify);
    }
    let verifying = start.elapsed().as_micros() / repetition as u128;
    println!(
        "{:2} variables [loga]: {:9} us, {:9} us, {:9} us",
        nv, key_extraction, proving, verifying
    );
    Ok(())
}

fn bench_mock_circuit_zkp_helper_with_plk(
    file: &mut File,
    nv: usize,
    gate: &CustomizedGates,
    pcs_srs: &MultilinearUniversalParams<Bls12_381>,
) -> Result<(), HyperPlonkErrors> {
    let repetition = if nv < 10 {
        5
    } else if nv < 20 {
        3
    } else {
        1
    };

    //==========================================================
    let circuit = MockCircuit::<Fr>::new(1 << nv, gate, gate, false);
    assert!(circuit.is_satisfied());
    let index = circuit.index;
    //==========================================================
    // generate pk and vks
    let start = Instant::now();
    for _ in 0..(repetition - 1) {
        let (_pk, _vk) = <PolyIOP<Fr> as HyperPlonkSNARK<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::preprocess(&index, pcs_srs)?;
    }
    let (pk, vk) =
        <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::preprocess(
            &index, pcs_srs,
        )?;
    let key_extraction = start.elapsed().as_micros() / repetition as u128;
    //==========================================================
    // generate a proof
    let start = Instant::now();
    for _ in 0..(repetition - 1) {
        let _proof =
            <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::prove(
                &pk,
                &circuit.public_inputs,
                &circuit.witnesses,
            )?;
    }
    let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::prove(
        &pk,
        &circuit.public_inputs,
        &circuit.witnesses,
    )?;
    let proving = start.elapsed().as_micros() / repetition as u128;
    file.write_all(format!("{} {}\n", nv, proving).as_ref())
        .unwrap();

    //==========================================================
    // verify a proof
    let start = Instant::now();
    for _ in 0..repetition {
        let verify =
            <PolyIOP<Fr> as HyperPlonkSNARK<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::verify(
                &vk,
                &circuit.public_inputs,
                &proof,
            )?;
        assert!(verify);
    }
    let verifying = start.elapsed().as_micros() / repetition as u128;
    println!(
        "{:2} variables [p-lk]: {:9} us, {:9} us, {:9} us",
        nv, key_extraction, proving, verifying
    );
    Ok(())
}
