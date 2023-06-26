use crate::{PolyIOPErrors, PolynomialCommitmentScheme, ZeroCheck};
use ark_ec::pairing::Pairing;
use transcript::IOPTranscript;

pub trait LookupCheck<E, PCS>: ZeroCheck<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type LookupCheckSubClaim;
    type LookupCheckProof;

    fn prove(
        pcs_param: &PCS::ProverParam,
        f: &Self::MultilinearExtension,
        t: &Self::MultilinearExtension,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<Self::LookupCheckProof, PolyIOPErrors>;

    fn verifier(
        proof: &Self::LookupCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors>;
}
