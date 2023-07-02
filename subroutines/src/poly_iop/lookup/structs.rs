use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use dashmap::DashMap;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct PreprocessedTable<F: PrimeField, V> {
    pub table: Vec<F>,
    pub table_map: DashMap<F, V>,
    pub t: Arc<DenseMultilinearExtension<F>>,
}
