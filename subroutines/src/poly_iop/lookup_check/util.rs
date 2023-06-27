use crate::poly_iop::errors::PolyIOPErrors;
use arithmetic::VirtualPolynomial;
use ark_ff::{batch_inversion, PrimeField};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use std::{collections::HashMap, sync::Arc};

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

    let mut h_f = HashMap::new();
    let mut h_t = HashMap::new();

    // Count number of occurences of each elements
    for num in t.iter() {
        *h_t.entry(*num).or_insert(F::zero()) += F::one();
    }
    for num in f.iter() {
        if h_t.get(num).is_none() {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "Lookup value {num} is not in table"
            )));
        }
        *h_f.entry(*num).or_insert(F::zero()) += F::one();
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

/// Calculate A(x) = m(x) / (beta + t(x)).
pub(super) fn compute_A<F: PrimeField>(
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
    let mut denom_evals = t.iter().map(|eval| *eval + beta).collect::<Vec<F>>();
    batch_inversion(&mut denom_evals);

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
pub(super) fn compute_B<F: PrimeField>(
    f: &Arc<DenseMultilinearExtension<F>>,
    beta: &F,
) -> Result<Arc<DenseMultilinearExtension<F>>, PolyIOPErrors> {
    let num_vars = f.num_vars();

    // Calculate denominator evaluations.
    let mut denom_evals = f.iter().map(|eval| *eval + beta).collect::<Vec<F>>();
    batch_inversion(&mut denom_evals);

    Ok(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars,
        denom_evals,
    )))
}

/// Compute virtual polynomial
///     p(x) = A(x) * (beta + t(x)) - m(x)
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
