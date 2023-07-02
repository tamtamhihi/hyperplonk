use crate::{PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme, ProductCheck};
use arithmetic::merge_polynomials;
use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use std::sync::Arc;
use transcript::IOPTranscript;

pub mod utils;
use self::utils::{compute_h, compute_poly_delta, embed};
use super::structs::PreprocessedTable;
use dashmap::DashMap;

pub trait PlookupCheck<E, PCS>: ProductCheck<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type PlookupCheckSubClaim;
    type PlookupCheckProof;

    fn init_transcript() -> Self::Transcript;

    fn preprocess_table(
        nv: usize,
        table: &[E::ScalarField],
    ) -> Result<PreprocessedTable<E::ScalarField, usize>, PolyIOPErrors>;

    #[allow(clippy::type_complexity)]
    fn prove(
        pcs_param: &PCS::ProverParam,
        f: &Self::MultilinearExtension,
        preprocessed_table: &PreprocessedTable<E::ScalarField, usize>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::PlookupCheckProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    >;

    fn verify(
        proof: &Self::PlookupCheckProof,
        aux_infor: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PlookupCheckSubClaim, PolyIOPErrors>;
}

pub struct PlookupCheckSubClaim<E, PCS, PC>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    PC: ProductCheck<E, PCS>,
{
    pub product_check_subclaim: PC::ProductCheckSubClaim,

    pub challenges: (E::ScalarField, E::ScalarField),
}

pub struct PlookupCheckProof<E, PCS, PC>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    PC: ProductCheck<E, PCS>,
{
    pub product_check_proof: PC::ProductCheckProof,
    pub h_comm: PCS::Commitment,
    pub g1_comm: PCS::Commitment,
    pub g2_comm: PCS::Commitment,
    pub h_delta_comm: PCS::Commitment,
}

impl<E, PCS> PlookupCheck<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type PlookupCheckSubClaim = PlookupCheckSubClaim<E, PCS, Self>;
    type PlookupCheckProof = PlookupCheckProof<E, PCS, Self>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing PlookupCheck transcript")
    }

    fn preprocess_table(
        nv: usize,
        table: &[E::ScalarField],
    ) -> Result<PreprocessedTable<E::ScalarField, usize>, PolyIOPErrors> {
        if table.len() != (1 << nv) - 1 {
            return Err(PolyIOPErrors::InvalidParameters(
                "Table size is not in from of ".to_string(),
            ));
        };

        let t = embed(table, nv)?;

        let h_t = DashMap::<E::ScalarField, usize>::new();
        for val in table.iter() {
            *h_t.entry(*val).or_insert_with(|| 0) += 1;
        }
        Ok(PreprocessedTable {
            table: Vec::from(table),
            table_map: h_t,
            t,
        })
    }

    fn prove(
        pcs_param: &PCS::ProverParam,
        f: &Self::MultilinearExtension,
        preprocessed_table: &PreprocessedTable<E::ScalarField, usize>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<
        (
            Self::PlookupCheckProof,
            Self::MultilinearExtension,
            Self::MultilinearExtension,
        ),
        PolyIOPErrors,
    > {
        let num_vars = f.num_vars;

        // h is vector
        // num elements: 2^{mu + 1} - 1
        let h = compute_h(f, &preprocessed_table.table)?;

        // num vars: nv + 1
        let h_emb = embed(&h, num_vars + 1)?;
        let h_comm = PCS::commit(pcs_param, &h_emb)?;
        transcript.append_serializable_element(b"h_comm", &h_comm)?;

        // polynomial g1 = merge(f,t)
        // num vars: nv + 1
        let g1 = merge_polynomials(&[f.clone(), preprocessed_table.t.clone()])?;
        let g1_comm = PCS::commit(pcs_param, &g1)?;
        transcript.append_serializable_element(b"g1_comm", &g1_comm)?;

        // num vars: nv
        let t_delta = compute_poly_delta(&preprocessed_table.t, num_vars)?;

        // polynomial g2 = merge(f, t_delta)
        // num vars: nv + 1
        let g2 = merge_polynomials(&[f.clone(), t_delta])?;
        let g2_comm = PCS::commit(pcs_param, &g2)?;
        transcript.append_serializable_element(b"g2_comm", &g2_comm)?;

        // num vars: nv + 1
        let h_delta = compute_poly_delta(&h_emb, num_vars + 1)?;
        let h_delta_comm = PCS::commit(pcs_param, &h_delta)?;
        transcript.append_serializable_element(b"h_delta_comm", &h_delta_comm)?;

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        // run multiset check for
        // ([[g1]], [[g2]], [[h]],[[h_delta_mu_plus_one]]; (f,t,h)) in R_MSET^2

        // combine g1 evals with g2 evals
        // num evals: (1 << (nv + 1))
        let numerator_evals = g1
            .evaluations
            .iter()
            .zip(g2.evaluations.iter())
            .map(|(&g1_eval, &g2_eval)| g1_eval + beta * g2_eval + gamma)
            .collect::<Vec<E::ScalarField>>();
        let numerator = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars + 1,
            numerator_evals,
        ));

        // combine h_emb evals with h_delta evals
        // num evals: (1 << (nv + 1))
        let denominator_evals = h_emb
            .evaluations
            .iter()
            .zip(h_delta.evaluations.iter())
            .map(|(&h_emb_eval, &h_delta_eval)| h_emb_eval + beta * h_delta_eval + gamma)
            .collect::<Vec<E::ScalarField>>();

        let denominator = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars + 1,
            denominator_evals,
        ));

        let (proof, prod_poly, frac_poly) = <Self as ProductCheck<E, PCS>>::prove(
            pcs_param,
            &[numerator],
            &[denominator],
            transcript,
        )?;

        Ok((
            PlookupCheckProof {
                product_check_proof: proof,
                h_comm,
                g1_comm,
                g2_comm,
                h_delta_comm,
            },
            prod_poly,
            frac_poly,
        ))
    }

    fn verify(
        proof: &Self::PlookupCheckProof,
        aux_infor: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PlookupCheckSubClaim, PolyIOPErrors> {
        // update transcript and generate challenge
        transcript.append_serializable_element(b"h_comm", &proof.h_comm)?;
        transcript.append_serializable_element(b"g1_comm", &proof.g1_comm)?;
        transcript.append_serializable_element(b"g2_comm", &proof.g2_comm)?;
        transcript.append_serializable_element(b"h_delta_comm", &proof.h_delta_comm)?;

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        let product_check_subclaim = <Self as ProductCheck<E, PCS>>::verify(
            &proof.product_check_proof,
            aux_infor,
            transcript,
        )?;

        Ok(PlookupCheckSubClaim {
            product_check_subclaim,
            challenges: (beta, gamma),
        })
    }
}

#[cfg(test)]
mod test {

    use std::{marker::PhantomData, sync::Arc};

    use arithmetic::{evaluate_opt, VPAuxInfo};
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::Pairing;
    use ark_poly::DenseMultilinearExtension;
    use ark_std::test_rng;

    use crate::{
        MultilinearKzgPCS, PolyIOP, PolyIOPErrors, PolynomialCommitmentScheme, PreprocessedTable,
    };

    use super::PlookupCheck;

    fn test_plookup_check_helper<E, PCS>(
        pcs_param: &PCS::ProverParam,
        f: &Arc<DenseMultilinearExtension<E::ScalarField>>,
        preprocessed_table: &PreprocessedTable<E::ScalarField, usize>,
    ) -> Result<(), PolyIOPErrors>
    where
        E: Pairing,
        PCS: PolynomialCommitmentScheme<
            E,
            Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        >,
    {
        let mut transcript = <PolyIOP<E::ScalarField> as PlookupCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        // 1) Generate proof
        let (proof, prod_poly, _frac_poly) =
            <PolyIOP<E::ScalarField> as PlookupCheck<E, PCS>>::prove(
                pcs_param,
                f,
                preprocessed_table,
                &mut transcript,
            )?;

        // 2) Verify the proof
        let mut transcript = <PolyIOP<E::ScalarField> as PlookupCheck<E, PCS>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;

        // pc_aux_infor for poly g1, g2, h, h_delta
        // num_variables = num_vars + 1
        let pc_aux_infor: VPAuxInfo<E::ScalarField> = VPAuxInfo {
            max_degree: 2,
            num_variables: f.num_vars + 1,
            phantom: PhantomData::default(),
        };

        let plookup_check_subclaim = <PolyIOP<E::ScalarField> as PlookupCheck<E, PCS>>::verify(
            &proof,
            &pc_aux_infor,
            &mut transcript,
        )?;

        // check product subclaim
        if evaluate_opt(
            &prod_poly,
            &plookup_check_subclaim.product_check_subclaim.final_query.0,
        ) != plookup_check_subclaim.product_check_subclaim.final_query.1
        {
            return Err(PolyIOPErrors::InvalidVerifier("wrong subclaim".to_string()));
        };

        Ok(())
    }

    fn test_plookup_check(nv: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        // generate the table, whose each element is distinct
        let table = (1..(1 << nv)).map(Fr::from).collect::<Vec<_>>();

        let preprocessed_table = <PolyIOP<Fr> as PlookupCheck<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::preprocess_table(nv, &table)?;

        // generate lookups
        let half_nv = 1 << (nv - 1);
        let lookups = (0..half_nv)
            .map(|i| vec![table[i]; 2])
            .collect::<Vec<_>>()
            .concat();

        let f = Arc::new(DenseMultilinearExtension::<Fr>::from_evaluations_vec(
            nv, lookups,
        ));

        // generate srs
        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, nv + 1)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(srs, None, Some(nv + 1))?;

        // generate proof & verify proof
        test_plookup_check_helper::<Bls12_381, MultilinearKzgPCS<Bls12_381>>(
            &pcs_param,
            &f,
            &preprocessed_table,
        )?;

        Ok(())
    }

    #[test]
    fn test_4() -> Result<(), PolyIOPErrors> {
        test_plookup_check(4)
    }

    #[test]
    fn test_10() -> Result<(), PolyIOPErrors> {
        test_plookup_check(10)
    }

    #[test]
    fn test_15() -> Result<(), PolyIOPErrors> {
        test_plookup_check(15)
    }
}
