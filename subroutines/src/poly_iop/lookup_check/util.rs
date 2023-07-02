use crate::poly_iop::errors::PolyIOPErrors;
use arithmetic::VirtualPolynomial;
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use dashmap::DashMap;
#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[cfg(not(feature = "parallel"))]
use ark_std::collections::HashMap;

/// Compute normalized multiplicity polynomial:
///        m(x) = m_f(x) / m_t(x),
/// where m_f(x) = count of value t(x) in lookup f
/// and m_t(x) = count of value t(x) in table t.
pub(super) fn compute_multiplicity_poly<F: PrimeField>(
    f: &Arc<DenseMultilinearExtension<F>>,
    t: &Arc<DenseMultilinearExtension<F>>,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolyIOPErrors> {
    assert!(
        f.num_vars == t.num_vars,
        "lookup and table should have equal size"
    );
    let num_vars = f.num_vars;

    #[cfg(feature = "parallel")]
    return {
        let h_f = DashMap::<F, F>::new();
        let h_t = DashMap::<F, F>::new();

        // Count number of occurences of each elements
        t.evaluations.par_iter().for_each(|num| {
            *h_t.entry(*num).or_insert_with(F::zero) += F::one();
        });
        f.evaluations
            .par_iter()
            .map(|num| -> Result<(), PolyIOPErrors> {
                if h_t.get(num).is_none() {
                    return Err(PolyIOPErrors::InvalidProof(format!(
                        "Lookup value {num} is not in table"
                    )));
                }
                *h_f.entry(*num).or_insert_with(F::zero) += F::one();
                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let m_evals = t
            .evaluations
            .par_iter()
            .map(|value| {
                if let Some(h_f_val) = h_f.get(value) {
                    *h_f_val / *h_t.get(value).unwrap()
                } else {
                    F::zero()
                }
            })
            .collect::<Vec<F>>();
        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, m_evals,
        )))
    };

    #[cfg(not(feature = "parallel"))]
    {
        let mut h_f = HashMap::new();
        let mut h_t = HashMap::new();

        // Count number of occurences of each elements
        for num in t.iter() {
            *h_t.entry(*num).or_insert_with(F::zero) += F::one();
        }
        for num in f.iter() {
            if h_t.get(num).is_none() {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "Lookup value {num} is not in table"
                )));
            }
            *h_f.entry(*num).or_insert_with(F::zero) += F::one();
        }

        let m_evals = t
            .iter()
            .map(|value| {
                if let Some(h_f_val) = h_f.get(value) {
                    *h_f_val / *h_t.get(value).unwrap()
                } else {
                    F::zero()
                }
            })
            .collect::<Vec<F>>();

        Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, m_evals,
        )))
    }
}

/// Calculate A(x) = m(x) / (beta + t(x)).
pub(super) fn compute_a<F: PrimeField>(
    m: &Arc<DenseMultilinearExtension<F>>,
    t: &Arc<DenseMultilinearExtension<F>>,
    beta: &F,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolyIOPErrors> {
    assert!(
        m.num_vars() == t.num_vars(),
        "multiplicity and table should have same num vars"
    );

    let num_vars = m.num_vars();

    // Denominator evaluations
    #[cfg(feature = "parallel")]
    let mut denom_evals = t
        .evaluations
        .par_iter()
        .map(|eval| *eval + beta)
        .collect::<Vec<_>>();
    #[cfg(not(feature = "parallel"))]
    let mut denom_evals = t.iter().map(|eval| *eval + beta).collect::<Vec<F>>();
    batch_inversion(&mut denom_evals);

    #[cfg(feature = "parallel")]
    let a_evals = m
        .evaluations
        .par_iter()
        .zip(denom_evals.par_iter())
        .map(|(m_eval, denom_eval)| *m_eval * *denom_eval)
        .collect::<Vec<F>>();
    #[cfg(not(feature = "parallel"))]
    let a_evals = m
        .iter()
        .zip(denom_evals.iter())
        .map(|(m_eval, denom_eval)| *m_eval * *denom_eval)
        .collect::<Vec<F>>();

    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars, a_evals,
    )))
}

/// Calculate B(x) = 1 / (beta + f())
pub(super) fn compute_b<F: PrimeField>(
    f: &Arc<DenseMultilinearExtension<F>>,
    beta: &F,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolyIOPErrors> {
    let num_vars = f.num_vars();

    // Calculate denominator evaluations.
    #[cfg(feature = "parallel")]
    let mut denom_evals = f
        .evaluations
        .par_iter()
        .map(|eval| *eval + beta)
        .collect::<Vec<F>>();
    #[cfg(not(feature = "parallel"))]
    let mut denom_evals = f.iter().map(|eval| *eval + beta).collect::<Vec<F>>();
    batch_inversion(&mut denom_evals);

    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        denom_evals,
    )))
}

/// Compute A(x) * (beta + t(x)) - m(x) + alpha * (B(x) * (beta + f(x)) - 1)
pub fn build_pq_virtual<F: PrimeField>(
    a: &Arc<DenseMultilinearExtension<F>>,
    b: &Arc<DenseMultilinearExtension<F>>,
    f: &Arc<DenseMultilinearExtension<F>>,
    t: &Arc<DenseMultilinearExtension<F>>,
    m: &Arc<DenseMultilinearExtension<F>>,
    alpha: &F,
    beta: &F,
) -> Result<VirtualPolynomial<F>, PolyIOPErrors> {
    let num_vars = a.num_vars();
    let mut vp = VirtualPolynomial::new(num_vars);

    vp.add_mle_list([Arc::clone(a)], *beta)?;
    vp.add_mle_list([Arc::clone(a), Arc::clone(t)], F::one())?;
    vp.add_mle_list([Arc::clone(m)], -F::one())?;

    vp.add_mle_list([Arc::clone(b)], *alpha * beta)?;
    vp.add_mle_list([Arc::clone(b), Arc::clone(f)], *alpha)?;

    vp.add_mle_list(
        [Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![*alpha; 1 << num_vars],
        ))],
        -F::one(),
    )?;

    Ok(vp)
}

/// Compute virtual polynomial
///     p(x) = A(x) * (beta + t(x)) - m(x)
#[allow(dead_code)]
pub fn build_p_virtual<F: PrimeField>(
    a: &Arc<DenseMultilinearExtension<F>>,
    t: &Arc<DenseMultilinearExtension<F>>,
    m: &Arc<DenseMultilinearExtension<F>>,
    beta: &F,
) -> Result<VirtualPolynomial<F>, PolyIOPErrors> {
    assert!(
        a.num_vars() == t.num_vars() && t.num_vars == m.num_vars(),
        "All polynomials must have the same number of variables"
    );
    let num_vars = a.num_vars();
    let mut vp = VirtualPolynomial::new(num_vars);

    vp.add_mle_list(vec![Arc::clone(a)], *beta)?;
    vp.add_mle_list(vec![Arc::clone(a), Arc::clone(t)], F::one())?;
    vp.add_mle_list(vec![Arc::clone(m)], -F::one())?;

    Ok(vp)
}

/// Compute virtual polynomial
///     q(x) = B(x) * (beta + f(x)) - 1
#[allow(dead_code)]
pub fn build_q_virtual<F: PrimeField>(
    b: &Arc<DenseMultilinearExtension<F>>,
    f: &Arc<DenseMultilinearExtension<F>>,
    beta: &F,
) -> Result<VirtualPolynomial<F>, PolyIOPErrors> {
    assert!(
        b.num_vars() == f.num_vars,
        "All polynomial must have same the same number of variables"
    );

    let num_vars = b.num_vars();
    let mut vp = VirtualPolynomial::new(num_vars);

    vp.add_mle_list(vec![Arc::clone(b)], *beta)?;
    vp.add_mle_list(vec![Arc::clone(b), Arc::clone(f)], F::one())?;
    vp.add_mle_list(
        vec![Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars,
            vec![F::one(); 1 << num_vars],
        ))],
        -F::one(),
    )?;

    Ok(vp)
}

/// Compute virtual polynomial
///     L(x) = A(x) - B(x)
pub fn build_l_virtual<F: PrimeField>(
    a: &Arc<DenseMultilinearExtension<F>>,
    b: &Arc<DenseMultilinearExtension<F>>,
) -> Result<VirtualPolynomial<F>, PolyIOPErrors> {
    assert!(
        a.num_vars() == b.num_vars,
        "All polynomial must have same the same number of variables"
    );

    let mut vp = VirtualPolynomial::new_from_mle(a, F::one());
    vp.add_mle_list(vec![Arc::clone(b)], -F::one())?;

    Ok(vp)
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::PrimeField;
    use ark_std::{test_rng, UniformRand};
    use std::convert::From;

    #[test]
    fn test_compute_multiplicity_poly() -> Result<(), PolyIOPErrors> {
        let nv = 3;
        let f = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
            nv,
            convert_usize_to_field(&[1, 1, 1, 2, 2, 3, 3, 4]),
        ));
        let t = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
            nv,
            convert_usize_to_field(&[1, 2, 3, 4, 4, 4, 4, 4]),
        ));

        let m = compute_multiplicity_poly(&f, &t)?;
        let mut expected_m = vec![Fr::from(3), Fr::from(2), Fr::from(2)];
        expected_m.append(&mut vec![Fr::from(1) / Fr::from(5); 5]);
        assert_eq!(
            m.to_evaluations(),
            expected_m,
            "multiplicity poly is incorrect, expected {:?}, found {expected_m:?}",
            m.to_evaluations(),
        );
        Ok(())
    }

    #[test]
    fn test_compute_a_poly() -> Result<(), PolyIOPErrors> {
        let nv = 3;
        let mut rng = test_rng();

        let t = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let m = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let beta = Fr::rand(&mut rng);

        let a = compute_a(&m, &t, &beta)?;

        for i in 0..1 << nv {
            assert_eq!(
                a.evaluations[i],
                m.evaluations[i] / (beta + t.evaluations[i]),
                "a poly is not correct at position {i}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_compute_b_poly() -> Result<(), PolyIOPErrors> {
        let nv = 3;
        let mut rng = test_rng();

        let f = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let beta = Fr::rand(&mut rng);

        let b = compute_b(&f, &beta)?;

        for i in 0..1 << nv {
            assert_eq!(
                b.evaluations[i],
                Fr::from(1) / (beta + f.evaluations[i]),
                "b poly is not correct at position {i}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_compute_p_vp() -> Result<(), PolyIOPErrors> {
        let nv = 3;
        let mut rng = test_rng();

        let a = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let m = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let t = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));

        let beta = Fr::rand(&mut rng);

        let p = build_p_virtual(&a, &t, &m, &beta)?.to_mle()?;

        for i in 0..1 << nv {
            assert_eq!(
                p.evaluations[i],
                a.evaluations[i] * (beta + t.evaluations[i]) - m.evaluations[i],
                "p virtual poly is not correct at position {i}"
            );
        }

        Ok(())
    }

    #[test]
    fn test_compute_q_vp() -> Result<(), PolyIOPErrors> {
        let nv = 3;
        let mut rng = test_rng();

        let b = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));
        let f = Arc::new(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng));

        let beta = Fr::rand(&mut rng);

        let q = build_q_virtual(&b, &f, &beta)?.to_mle()?;

        for i in 0..1 << nv {
            assert_eq!(
                q.evaluations[i],
                b.evaluations[i] * (beta + f.evaluations[i]) - Fr::from(1),
                "q virtual poly is not correct at position {i}"
            );
        }

        Ok(())
    }

    fn convert_usize_to_field<F: PrimeField>(v: &[usize]) -> Vec<F> {
        v.iter()
            .map(|value| F::from(*value as u128))
            .collect::<Vec<_>>()
    }
}
