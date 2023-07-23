use arithmetic::bit_decompose;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use dashmap::{DashMap, DashSet};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

use crate::poly_iop::errors::PolyIOPErrors;

pub(super) fn compute_h<F: PrimeField>(
    f: &Arc<DenseMultilinearExtension<F>>,
    t: &[F],
    h_t: &DashMap<F, usize>,
) -> Result<Vec<F>, PolyIOPErrors> {
    let nv = f.num_vars;
    assert!(
        (1 << nv) - 1 == t.len(),
        "lookup size and table size are incorrect"
    );

    let h_f = DashMap::new();

    f.evaluations
        .par_iter()
        .map(|num| -> Result<(), PolyIOPErrors> {
            if h_t.get(num).is_none() {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "Lookup value {num} is not in table"
                )));
            }
            *h_f.entry(*num).or_insert_with(|| 0) += 1;
            Ok(())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut evaluations: Vec<F> = vec![];
    let table_set = DashSet::new();
    for num in t.iter() {
        if !table_set.contains(num) {
            let h_t_val: usize = *h_t.get(num).unwrap();
            if let Some(h_f_val) = h_f.get(num) {
                evaluations.append(&mut vec![*num; h_t_val + *h_f_val]);
            } else {
                evaluations.append(&mut vec![*num; h_t_val]);
            }
            table_set.insert(*num);
        }
    }

    Ok(evaluations)
}

pub(super) fn get_primitive_polynomial(n: usize) -> Result<Vec<usize>, PolyIOPErrors> {
    if n < 2 || n > 20 {
        return Err(PolyIOPErrors::InvalidProof(
            "Only support primitive polynomial whose degree >= 2 and <= 20".to_string(),
        ));
    }
    let primitive_poly = match n {
        2 => vec![1, 1, 1],
        3 => vec![1, 0, 1, 1],
        4 => vec![1, 0, 0, 1, 1],
        5 => vec![1, 0, 0, 1, 0, 1],
        6 => vec![1, 0, 0, 0, 0, 1, 1],
        7 => vec![1, 0, 0, 0, 0, 0, 1, 1],
        8 => vec![1, 0, 0, 0, 1, 1, 1, 0, 1],
        9 => vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        10 => vec![1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        11 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        12 => vec![1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        13 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        14 => vec![1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        15 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        16 => vec![1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        17 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        18 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        19 => vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        20 => vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        ],
        _ => vec![],
    };
  
    Ok(primitive_poly)
}

/// g_mu(b1,...,b_mu) = (b_mu, b1', ... , b_{mu-1}')
///
pub(super) fn next_element(
    cur_num: usize,
    nv: usize,
    s: &[usize], // primitive polynomial
) -> usize {
    let bit_sequence = bit_decompose(cur_num as u64, nv);
    let s_bool = s[1..nv].par_iter().map(|x| *x == 1).collect::<Vec<bool>>();

    let sign = bit_sequence[0];
    let next_num = project(
        &[
            transform(&bit_sequence[1..nv], &s_bool, sign).as_ref(),
            [bit_sequence[0]].as_ref(),
        ]
        .concat(),
    ) as usize;

    next_num
}

pub(super) fn transform(a: &[bool], s: &[bool], sign: bool) -> Vec<bool> {
    a.par_iter()
        .zip(s.par_iter())
        .map(|(&a_val, &s_val)| if s_val { a_val ^ sign } else { a_val })
        .collect::<Vec<bool>>()
}

/// Project a little endian binary vector into an integer.
pub(super) fn project(input: &[bool]) -> u64 {
    let mut res = 0;
    for &e in input.iter().rev() {
        res <<= 1;
        res += e as u64;
    }
    res
}

pub fn embed<F: PrimeField>(
    poly_evals: &[F], //2^num_vars - 1
    nv: usize,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolyIOPErrors> {
    assert!(
        poly_evals.len() == (1 << nv) - 1,
        "Embedded evaluations must be in form of pow(2,nv) - 1"
    );
    let s = get_primitive_polynomial(nv)?;
    let mut embedded_poly_evals: Vec<F> = vec![F::zero(); 1 << nv];

    let mut res = vec![];
    let mut cur_element: usize = 1 << (nv - 1);

    res.push(cur_element);

    for item in poly_evals.iter() {
        embedded_poly_evals[cur_element] = *item;
        cur_element = next_element(cur_element, nv, &s);
        res.push(cur_element);
    }

    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        nv,
        embedded_poly_evals,
    )))
}

/// poly_delta(X1,X2...,Xnv) = Xnv * poly(1,X1',...,X_{nv-1}')
///             + (1 - Xnv) * poly(0,X1,...,X_{nv-1})
/// where Xi' := 1 - Xi (if i in S), and Xi' := Xi otherwise.
pub(super) fn compute_poly_delta<F: PrimeField>(
    poly: &Arc<DenseMultilinearExtension<F>>,
    nv: usize,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolyIOPErrors> {
    let s = get_primitive_polynomial(nv)?;
    let s_bool = s[1..nv].iter().map(|x| *x == 1).collect::<Vec<bool>>();

    let poly_evals = &poly.evaluations;

    let mut evaluations: Vec<F> = vec![];
    evaluations.push(poly_evals[0]);

    for i in 1..(1 << nv) {
        let bit_sequence = bit_decompose(i as u64, nv);
        let sign = bit_sequence[0];
        let x0 = project(&[bit_sequence[1..nv].as_ref(), [false].as_ref()].concat()) as usize;
        let x1 = project(
            &[
                transform(&bit_sequence[1..nv], &s_bool, sign).as_ref(),
                [true].as_ref(),
            ]
            .concat(),
        ) as usize;

        let sign = bit_sequence[0];
        if !sign {
            evaluations.push(poly_evals[x0]);
        } else {
            evaluations.push(poly_evals[x1]);
        }
    }

    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        nv,
        evaluations,
    )))
}

#[cfg(test)]
mod test {
    use super::embed;
    use super::*;
    use ark_bls12_381::Fr;
    use ark_poly::MultilinearExtension;
    use ark_std::test_rng;

    #[test]
    fn test_embed() -> Result<(), PolyIOPErrors> {
        let nv = 3;
        let poly_evals = (1..8).map(Fr::from).collect::<Vec<_>>();

        let poly_embed = embed(&poly_evals, nv)?;

        let poly_embed_evals = &poly_embed.evaluations;

        for x in poly_evals.iter() {
            if !poly_embed_evals.contains(x) {
                return Err(PolyIOPErrors::InvalidParameters(
                    "wrong embedding".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn test_compute_poly_delta() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        let nv = 3;
        let poly = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let poly_evals = &poly.evaluations;

        let poly_delta = compute_poly_delta(&poly, nv)?;
        let poly_delta_evals = &poly_delta.evaluations;

        if poly_evals.len() != poly_delta_evals.len() {
            return Err(PolyIOPErrors::InvalidParameters(
                "wrong embedding: Result poly not have the same size".to_string(),
            ));
        }
        for x in poly_evals {
            if !poly_delta_evals.contains(x) {
                return Err(PolyIOPErrors::InvalidParameters(
                    "wrong embedding: Result poly not equal to Original poly".to_string(),
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn test_compute_h() -> Result<(), PolyIOPErrors> {
        let nv = 3;

        // generate the table, whose each element is distinct
        let table = (1..(1 << nv)).map(Fr::from).collect::<Vec<_>>();
        let h_t = DashMap::new();
        for num in table.iter() {
            *h_t.entry(*num).or_insert_with(|| 0) += 1;
        }

        let half_nv = 1 << (nv - 1);
        let lookups = (0..half_nv)
            .map(|i| vec![table[i]; 2])
            .collect::<Vec<_>>()
            .concat();

        let f = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv,
            lookups.clone(),
        ));

        let h = compute_h(&f, &table, &h_t)?;
        if h.len() != lookups.len() + table.len() {
            return Err(PolyIOPErrors::InvalidParameters(
                "wrong computing h: Incorrect result vector size".to_string(),
            ));
        }

        let mut lookups_len_expected = 0;
        let mut cur_table_value = table[0] - Fr::from(1);
        for x in h {
            if x == cur_table_value {
                lookups_len_expected += 1;
            } else {
                cur_table_value = x;
            }
        }

        if lookups_len_expected != lookups.len() {
            return Err(PolyIOPErrors::InvalidParameters(
                "wrong computing h: Incorrect result vector size".to_string(),
            ));
        }

        Ok(())
    }
}
