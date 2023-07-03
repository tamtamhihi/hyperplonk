use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use dashmap::DashMap;
use std::sync::Arc;

use crate::PolynomialCommitmentScheme;

#[derive(Clone, Debug, Default)]
pub struct LogaPreprocessedTable<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub table: Vec<E::ScalarField>,
    pub table_map: DashMap<E::ScalarField, E::ScalarField>,
    pub t: Arc<DenseMultilinearExtension<E::ScalarField>>,
    pub t_comm: PCS::Commitment,
}

#[derive(Clone, Debug, Default)]
pub struct PlookupPreprocessedTable<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub table: Vec<E::ScalarField>,
    pub table_map: DashMap<E::ScalarField, usize>,
    pub t: Arc<DenseMultilinearExtension<E::ScalarField>>,
    pub t_comm: PCS::Commitment,

    pub t_delta: Arc<DenseMultilinearExtension<E::ScalarField>>,
    pub t_delta_comm: PCS::Commitment,
}
