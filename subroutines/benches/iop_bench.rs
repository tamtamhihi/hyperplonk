// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use arithmetic::{identity_permutation_mles, VPAuxInfo, VirtualPolynomial};
use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::test_rng;
use std::{marker::PhantomData, sync::Arc, time::Instant};
use subroutines::{
    pcs::{prelude::MultilinearKzgPCS, PolynomialCommitmentScheme},
    poly_iop::prelude::{
        PermutationCheck, PolyIOP, PolyIOPErrors, ProductCheck, SumCheck, ZeroCheck,
    },
    LookupCheck, PlookupCheck,
};

type Kzg = MultilinearKzgPCS<Bls12_381>;

fn main() -> Result<(), PolyIOPErrors> {
    for argument in std::env::args() {
        let arg = argument.to_lowercase();
        match arg.as_str() {
            "sum" => bench_sum_check()?,
            "zero" => bench_zero_check()?,
            "prod" => bench_prod_check()?,
            "perm" => bench_permutation_check()?,
            "lk" => bench_lookup_check()?,
            "plk" => bench_plookup_check()?,
            _ => continue,
        };
    }
    Ok(())
}

fn bench_sum_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();
    for degree in 2..4 {
        for nv in 4..25 {
            let repetition = if nv < 10 {
                100
            } else if nv < 20 {
                50
            } else {
                10
            };

            let (poly, asserted_sum) =
                VirtualPolynomial::rand(nv, (degree, degree + 1), 2, &mut rng)?;
            let poly_info = poly.aux_info.clone();
            let proof = {
                let start = Instant::now();
                for _ in 0..repetition {
                    let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                    let _proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?;
                }

                println!(
                    "sum check proving time for {} variables and {} degree: {} ns",
                    nv,
                    degree,
                    start.elapsed().as_nanos() / repetition as u128
                );
                let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                <PolyIOP<Fr> as SumCheck<Fr>>::prove(&poly, &mut transcript)?
            };

            {
                let start = Instant::now();

                for _ in 0..repetition {
                    let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
                    let _subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
                        asserted_sum,
                        &proof,
                        &poly_info,
                        &mut transcript,
                    )?;
                }
                println!(
                    "sum check verification time for {} variables and {} degree: {} ns",
                    nv,
                    degree,
                    start.elapsed().as_nanos() / repetition as u128
                );
            }

            println!("====================================");
        }
    }
    Ok(())
}

fn bench_zero_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();
    for degree in 2..4 {
        for nv in 4..20 {
            let repetition = if nv < 10 {
                100
            } else if nv < 20 {
                50
            } else {
                10
            };

            let poly = VirtualPolynomial::rand_zero(nv, (degree, degree + 1), 2, &mut rng)?;
            let poly_info = poly.aux_info.clone();
            let proof = {
                let start = Instant::now();
                let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;
                let proof = <PolyIOP<Fr> as ZeroCheck<Fr>>::prove(&poly, &mut transcript)?;

                println!(
                    "zero check proving time for {} variables and {} degree: {} ns",
                    nv,
                    degree,
                    start.elapsed().as_nanos() / repetition as u128
                );
                proof
            };

            {
                let start = Instant::now();
                let mut transcript = <PolyIOP<Fr> as ZeroCheck<Fr>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;
                let _zero_subclaim =
                    <PolyIOP<Fr> as ZeroCheck<Fr>>::verify(&proof, &poly_info, &mut transcript)?;
                println!(
                    "zero check verification time for {} variables and {} degree: {} ns",
                    nv,
                    degree,
                    start.elapsed().as_nanos() / repetition as u128
                );
            }

            println!("====================================");
        }
    }
    Ok(())
}

fn bench_permutation_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    for nv in 4..20 {
        let srs = Kzg::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = Kzg::trim(&srs, None, Some(nv + 1))?;

        let repetition = if nv < 10 {
            100
        } else if nv < 20 {
            50
        } else {
            10
        };

        let ws = vec![Arc::new(DenseMultilinearExtension::rand(nv, &mut rng))];

        // identity map
        let perms = identity_permutation_mles(nv, 1);

        let proof =
            {
                let start = Instant::now();
                let mut transcript =
                    <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;

                let (proof, _q_x, _frac_poly) = <PolyIOP<Fr> as PermutationCheck<
                    Bls12_381,
                    Kzg,
                >>::prove(
                    &pcs_param, &ws, &ws, &perms, &mut transcript
                )?;

                println!(
                    "permutation check proving time for {} variables: {} ns",
                    nv,
                    start.elapsed().as_nanos() / repetition as u128
                );
                proof
            };

        {
            let poly_info = VPAuxInfo {
                max_degree: 2,
                num_variables: nv,
                phantom: PhantomData::default(),
            };

            let start = Instant::now();
            let mut transcript =
                <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let _perm_check_sum_claim = <PolyIOP<Fr> as PermutationCheck<Bls12_381, Kzg>>::verify(
                &proof,
                &poly_info,
                &mut transcript,
            )?;
            println!(
                "permutation check verification time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
        }

        println!("====================================");
    }

    Ok(())
}

fn bench_prod_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    for nv in 4..20 {
        let srs = Kzg::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = Kzg::trim(&srs, None, Some(nv + 1))?;

        let repetition = if nv < 10 {
            100
        } else if nv < 20 {
            50
        } else {
            10
        };

        let f: DenseMultilinearExtension<Fr> = DenseMultilinearExtension::rand(nv, &mut rng);
        let mut g = f.clone();
        g.evaluations.reverse();
        let fs = vec![Arc::new(f)];
        let gs = vec![Arc::new(g)];

        let proof = {
            let start = Instant::now();
            let mut transcript = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;

            let (proof, _prod_x, _frac_poly) =
                <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::prove(
                    &pcs_param,
                    &fs,
                    &gs,
                    &mut transcript,
                )?;

            println!(
                "product check proving time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
            proof
        };

        {
            let poly_info = VPAuxInfo {
                max_degree: 2,
                num_variables: nv,
                phantom: PhantomData::default(),
            };

            let start = Instant::now();
            let mut transcript = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let _perm_check_sum_claim = <PolyIOP<Fr> as ProductCheck<Bls12_381, Kzg>>::verify(
                &proof,
                &poly_info,
                &mut transcript,
            )?;
            println!(
                "product check verification time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
        }

        println!("====================================");
    }

    Ok(())
}

fn bench_lookup_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    for nv in 4..20 {
        let srs = Kzg::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = Kzg::trim(&srs, None, Some(nv + 1))?;

        let repetition = if nv < 10 {
            5
        } else if nv < 20 {
            2
        } else {
            1
        };

        let half_n = 1 << (nv - 1);
        let half_table = DenseMultilinearExtension::<Fr>::rand(nv - 1, &mut rng);
        let mut table = half_table.evaluations;
        table.append(&mut vec![table[half_n - 1]; half_n]);
        let preprocessed_table =
            <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::preprocess_table(nv, &table)?;

        let lookups = (0..half_n)
            .map(|i| vec![table[i]; 2])
            .collect::<Vec<_>>()
            .concat();
        let f = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
            nv, lookups,
        ));

        let proof = {
            let start = Instant::now();

            for _ in 0..repetition {
                let mut transcript =
                    <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;

                let (_, _, _, _) = <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::prove(
                    &pcs_param,
                    &f,
                    &preprocessed_table,
                    &mut transcript,
                )?;
            }
            println!(
                "lookup check proving time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
            let mut transcript = <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let (proof, _, _, _) = <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::prove(
                &pcs_param,
                &f,
                &preprocessed_table,
                &mut transcript,
            )?;
            proof
        };

        {
            let zc_aux_info: VPAuxInfo<Fr> = VPAuxInfo {
                max_degree: 2,
                num_variables: nv,
                phantom: PhantomData::default(),
            };
            let sc_aux_info: VPAuxInfo<Fr> = VPAuxInfo {
                max_degree: 1,
                num_variables: nv,
                phantom: PhantomData::default(),
            };

            let start = Instant::now();
            for _ in 0..repetition {
                let mut transcript =
                    <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;
                let _perm_check_sum_claim = <PolyIOP<Fr> as LookupCheck<Bls12_381, Kzg>>::verify(
                    &proof,
                    &zc_aux_info,
                    &sc_aux_info,
                    &mut transcript,
                )?;
            }
            println!(
                "lookup check verification time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
        }

        println!("====================================");
    }

    Ok(())
}

fn bench_plookup_check() -> Result<(), PolyIOPErrors> {
    let mut rng = test_rng();

    for nv in 4..15 {
        let srs = Kzg::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = Kzg::trim(&srs, None, Some(nv + 1))?;

        let repetition = if nv < 10 {
            5
        } else if nv < 20 {
            2
        } else {
            1
        };

        let half_n = 1 << (nv - 1);
        let half_table = DenseMultilinearExtension::<Fr>::rand(nv - 1, &mut rng);
        let mut table = half_table.evaluations;
        table.append(&mut vec![table[half_n - 1]; half_n - 1]);
        let preprocessed_table =
            <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::preprocess_table(nv, &table)?;

        let lookups = (0..half_n)
            .map(|i| vec![table[i]; 2])
            .collect::<Vec<_>>()
            .concat();
        let f = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
            nv, lookups,
        ));

        let proof = {
            let start = Instant::now();

            for _ in 0..repetition {
                let mut transcript =
                    <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;

                let (_, _, _) = <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::prove(
                    &pcs_param,
                    &f,
                    &preprocessed_table,
                    &mut transcript,
                )?;
            }
            println!(
                "plookup check proving time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
            let mut transcript = <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::init_transcript();
            transcript.append_message(b"testing", b"initializing transcript for testing")?;
            let (proof, _, _) = <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::prove(
                &pcs_param,
                &f,
                &preprocessed_table,
                &mut transcript,
            )?;
            proof
        };

        {
            let aux_info: VPAuxInfo<Fr> = VPAuxInfo {
                max_degree: 2,
                num_variables: nv + 1,
                phantom: PhantomData::default(),
            };

            let start = Instant::now();
            for _ in 0..repetition {
                let mut transcript =
                    <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::init_transcript();
                transcript.append_message(b"testing", b"initializing transcript for testing")?;
                let _perm_check_sum_claim = <PolyIOP<Fr> as PlookupCheck<Bls12_381, Kzg>>::verify(
                    &proof,
                    &aux_info,
                    &mut transcript,
                )?;
            }
            println!(
                "plookup check verification time for {} variables: {} ns",
                nv,
                start.elapsed().as_nanos() / repetition as u128
            );
        }

        println!("====================================");
    }

    Ok(())
}
