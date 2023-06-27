use crate::{PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme, SumCheck, ZeroCheck};
use arithmetic::VPAuxInfo;
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField, Zero};
use ark_poly::DenseMultilinearExtension;
use std::ops::Add;
use std::sync::Arc;
use transcript::IOPTranscript;
// git commit --no-verify -m ".."
mod util;
use self::util::*;

pub trait LookupCheck<E, PCS>: ZeroCheck<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type LookupCheckSubClaim;
    type LookupCheckProof;

    fn init_transcript() -> Self::Transcript;

    fn prove(
        pcs_param: &PCS::ProverParam,
        f: &Self::MultilinearExtension,
        t: &Self::MultilinearExtension,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::LookupCheckProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    >;

    fn verify(
        proof: &Self::LookupCheckProof,
        zc_aux_info: &VPAuxInfo<E::ScalarField>,
        sc_aux_info: &VPAuxInfo<E::ScalarField>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors>;
}

/// A lookup check subclaim consists of
/// - A zero check subclaim
/// - A value beta
pub struct LookupCheckSubClaim<F: PrimeField, ZC: ZeroCheck<F>> {
    pub zero_check_sub_claim: ZC::ZeroCheckSubClaim,

    pub sum_check_sub_claim: <ZC as SumCheck<F>>::SumCheckSubClaim,

    /// Challenges beta and alpha
    pub challenges: (F, F),
}

pub struct LookupCheckProof<E, PCS, ZC>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    ZC: ZeroCheck<E::ScalarField>,
{
    pub zc_proof: ZC::ZeroCheckProof,
    pub sc_proof: <ZC as SumCheck<E::ScalarField>>::SumCheckProof,
    pub f_comm: PCS::Commitment,
    pub m_comm: PCS::Commitment,
    pub a_comm: PCS::Commitment,
    pub b_comm: PCS::Commitment,
}

impl<E, PCS> LookupCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type LookupCheckSubClaim = LookupCheckSubClaim<E::ScalarField, Self>;
    type LookupCheckProof = LookupCheckProof<E, PCS, Self>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing LookupCheck transcript")
    }

    fn prove(
        pcs_param: &PCS::ProverParam,
        f: &Self::MultilinearExtension,
        t: &Self::MultilinearExtension,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::LookupCheckProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    > {
        // 1) Commit lookup & multiplicity polynomials.
        let f_comm = PCS::commit(pcs_param, f)?;
        transcript.append_serializable_element(b"f_comm", &f_comm)?;

        let m_poly = compute_multiplicity_poly(&f, &t)?;
        let m_comm = PCS::commit(pcs_param, &m_poly)?;
        transcript.append_serializable_element(b"m_comm", &m_comm)?;

        // 2) Create and commit polynomials A, B
        //      A(x) = m(x) / (beta + t(x))
        //      B(x) = 1 / (beta + f(x))
        let beta = transcript.get_and_append_challenge(b"beta")?;

        let a_poly = compute_A(&m_poly, &t, &beta)?;
        let b_poly = compute_B(&f, &beta)?;

        let a_comm = PCS::commit(pcs_param, &a_poly)?;
        let b_comm = PCS::commit(pcs_param, &b_poly)?;

        transcript.append_serializable_element(b"a_comm", &a_comm)?;
        transcript.append_serializable_element(b"b_comm", &b_comm)?;

        // 3) Build virtual polynomial p and q
        let mut p = build_p_virtual(&a_poly, &t, &m_poly, &beta)?;
        let mut q = build_q_virtual(&b_poly, &f, &beta)?;

        // 4) Batch ZeroCheck for p and q
        let alpha = transcript.get_and_append_challenge(b"alpha")?;
        let num_vars = f.num_vars;
        q.mul_by_mle(
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                vec![E::ScalarField::one(); 1 << num_vars],
            )),
            alpha,
        )?;
        p = p.add(&q);

        let zc_proof = <Self as ZeroCheck<E::ScalarField>>::prove(&p, transcript)?;

        // 5) SumCheck for L(x) = A(x) - B(x)
        let l = build_l_virtual(&a_poly, &b_poly)?;

        let sc_proof = <Self as SumCheck<E::ScalarField>>::prove(&l, transcript)?;

        Ok((
            LookupCheckProof {
                zc_proof,
                sc_proof,
                f_comm,
                m_comm,
                a_comm,
                b_comm,
            },
            m_poly,
            a_poly,
            b_poly,
        ))
    }

    fn verify(
        proof: &Self::LookupCheckProof,
        zc_aux_info: &VPAuxInfo<E::ScalarField>,
        sc_aux_info: &VPAuxInfo<E::ScalarField>,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors> {
        // update transcript and generate challenge
        transcript.append_serializable_element(b"f_comm", &proof.f_comm)?;
        transcript.append_serializable_element(b"m_comm", &proof.m_comm)?;
        let beta = transcript.get_and_append_challenge(b"beta")?;

        transcript.append_serializable_element(b"a_comm", &proof.a_comm)?;
        transcript.append_serializable_element(b"b_comm", &proof.b_comm)?;

        let alpha = transcript.get_and_append_challenge(b"alpha")?;
        let zc_sub_claim =
            <Self as ZeroCheck<E::ScalarField>>::verify(&proof.zc_proof, zc_aux_info, transcript)?;

        let sc_sub_claim = <Self as SumCheck<E::ScalarField>>::verify(
            E::ScalarField::zero(),
            &proof.sc_proof,
            sc_aux_info,
            transcript,
        )?;

        Ok(LookupCheckSubClaim {
            zero_check_sub_claim: zc_sub_claim,
            sum_check_sub_claim: sc_sub_claim,
            challenges: (beta, alpha),
        })
    }
}

#[cfg(test)]
mod test {
    use super::LookupCheck;
    use super::LookupCheckSubClaim;
    use crate::{
        pcs::{prelude::MultilinearKzgPCS, PolynomialCommitmentScheme},
        poly_iop::{errors::PolyIOPErrors, PolyIOP},
    };
    use arithmetic::VPAuxInfo;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use std::marker::PhantomData;
    use std::sync::Arc;

    fn test_lookup_check_helper<E, PCS>(
        f: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        t: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        pcs_param: &PCS::ProverParam,
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        // 1) Generate proof
        let mut transcript = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let (proof, m_poly, a_poly, b_poly) =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::prove(
                pcs_param,
                f,
                t,
                &mut transcript,
            )?;

        // 2) Verify the proof
        let mut transcript = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        let zc_aux_info: VPAuxInfo<E::ScalarField> = VPAuxInfo {
            max_degree: 3,
            num_variables: f.num_vars(),
            phantom: PhantomData::default(),
        };
        let sc_aux_info: VPAuxInfo<E::ScalarField> = VPAuxInfo {
            max_degree: 1,
            num_variables: f.num_vars(),
            phantom: PhantomData::default(),
        };

        let LookupCheckSubClaim {
            zero_check_sub_claim,
            sum_check_sub_claim,
            challenges,
        } = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS>>::verify(
            &proof,
            &zc_aux_info,
            &sc_aux_info,
            &mut transcript,
        )?;

        assert_eq!(
            a_poly.evaluate(&sum_check_sub_claim.point).unwrap()
                - b_poly.evaluate(&sum_check_sub_claim.point).unwrap(),
            sum_check_sub_claim.expected_evaluation,
            "sumcheck on A-B not satisfied",
        );

        // zero check: 2 ham p(A,m,t), q(B,f)
        // sum check: l = a - b

        // A(x) - B(x) = 0
        // m(x) / (beta + t(x)) - 1 / (beta + f(x)) = 0
        // Check zc subclaim
        // poly(x) == 0 for all x
        //      where poly = p - alpha*q
        //          p(x) = A(x) * (beta + t(x)) - m(x)
        //          q(x) = B(x) * (beta + f(x)) - 1
        // Batch openning for A(x), B(x), t(x), f(x) commitments
        // assert!(poly.evaluate(zero_check_sub_claim.point)? == zero_check_sub_claim.expected_evaluation,
        //     "wrong zero-check subclaim"
        // );

        // Check sc subclaim
        // sum_{x \in B\mu} poly = 0
        // where poly = L(x) = A(x) - B(x) == 0 for all x
        // Batch openning for A(x), B(x) commitments
        // assert!(
        //     poly.evaluate(&sum_check_sub_claim.point).unwrap() == sum_check_sub_claim.expected_evaluation,
        //     "wrong sum-check subclaim"
        // );

        Ok(())
    }

    fn test_lookup_check(num_vars: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        // 1) Generate the table, where the last half is padded.
        let half_n = 1 << (num_vars - 1);
        let half_table = DenseMultilinearExtension::<Fr>::rand(num_vars - 1, &mut rng);

        let mut table = half_table.evaluations;
        table.append(&mut vec![table[half_n - 1]; half_n]);

        let t = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_slice(
            num_vars, &table,
        ));

        // 2) Generate lookups
        let lookups = (0..half_n)
            .into_iter()
            .map(|i| vec![table[i]; 2])
            .collect::<Vec<_>>()
            .concat();
        let f = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
            num_vars, lookups,
        ));

        // 3) Generate srs
        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, num_vars)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(num_vars))?;

        // 4) Generate proof & verify proof
        test_lookup_check_helper::<Bls12_381, MultilinearKzgPCS<Bls12_381>>(&f, &t, &pcs_param)?;

        Ok(())
    }

    #[test]
    fn test_1() -> Result<(), PolyIOPErrors> {
        test_lookup_check(1)
    }

    #[test]
    fn test_10() -> Result<(), PolyIOPErrors> {
        test_lookup_check(10)
    }
}
