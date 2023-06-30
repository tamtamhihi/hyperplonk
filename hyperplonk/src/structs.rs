// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the HyperPlonk PolyIOP.

use crate::{custom_gate::CustomizedGates, prelude::HyperPlonkErrors, selectors::SelectorColumn};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::log2;
use std::sync::Arc;
use subroutines::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::prelude::{PermutationCheck, ZeroCheck},
    LookupCheck,
};

/// The proof for the HyperPlonk PolyIOP, consists of the following:
///   - the commitments to all witness MLEs
///   - a batch opening to all the MLEs at certain index
///   - the zero-check proof for checking custom gate-satisfiability
///   - the permutation-check proof for checking the copy constraints
#[derive(Clone, Debug, PartialEq)]
pub struct HyperPlonkProof<E, PC, LC, PCS>
where
    E: Pairing,
    PC: PermutationCheck<E, PCS>,
    LC: LookupCheck<E, PCS>,
    PCS: PolynomialCommitmentScheme<E>,
{
    // PCS commit for witnesses
    pub witness_commits: Vec<PCS::Commitment>,
    pub batch_openings: PCS::BatchProof,
    // =======================================================================
    // IOP proofs
    // =======================================================================
    // the custom gate zerocheck proof
    pub zero_check_proof: <PC as ZeroCheck<E::ScalarField>>::ZeroCheckProof,
    // the permutation check proof for copy constraints
    pub perm_check_proof: PC::PermutationProof,

    pub lookup_check_proof: LC::LookupCheckProof,
    pub lookup_zc_proof: <LC as ZeroCheck<E::ScalarField>>::ZeroCheckProof,
}

/// The HyperPlonk instance parameters, consists of the following:
///   - the number of constraints
///   - number of public input columns
///   - the customized gate function
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HyperPlonkParams {
    /// the number of constraints
    pub num_constraints: usize,
    /// number of public input
    // public input is only 1 column and is implicitly the first witness column.
    // this size must not exceed number of constraints.
    pub num_pub_input: usize,
    /// customized gate function
    pub gate_func: CustomizedGates,
    /// lookup gate functions, each corresponding to 1 table
    pub lk_gate_func: CustomizedGates,
}

impl HyperPlonkParams {
    /// Number of variables in a multilinear system
    pub fn num_variables(&self) -> usize {
        log2(self.num_constraints) as usize
    }

    /// number of selector columns
    pub fn num_selector_columns(&self) -> usize {
        self.gate_func.num_selector_columns()
    }

    /// number of witness columns
    pub fn num_witness_columns(&self) -> usize {
        self.gate_func.num_witness_columns()
    }

    /// number of lookup selector columns
    pub fn num_lk_selector_columns(&self) -> usize {
        self.lk_gate_func.num_selector_columns()
    }

    /// evaluate the identical polynomial
    pub fn eval_id_oracle<F: PrimeField>(&self, point: &[F]) -> Result<F, HyperPlonkErrors> {
        let len = self.num_variables() + (log2(self.num_witness_columns()) as usize);
        if point.len() != len {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "ID oracle point length = {}, expected {}",
                point.len(),
                len,
            )));
        }

        let mut res = F::zero();
        let mut base = F::one();
        for &v in point.iter() {
            res += base * v;
            base += base;
        }
        Ok(res)
    }
}

/// The HyperPlonk index, consists of the following:
///   - HyperPlonk parameters
///   - the wire permutation
///   - the selector vectors
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HyperPlonkIndex<F: PrimeField> {
    pub params: HyperPlonkParams,
    pub permutation: Vec<F>,
    pub selectors: Vec<SelectorColumn<F>>,
    pub lk_selectors: Vec<SelectorColumn<F>>,
    pub table: Vec<F>,
}

impl<F: PrimeField> HyperPlonkIndex<F> {
    /// Number of variables in a multilinear system
    pub fn num_variables(&self) -> usize {
        self.params.num_variables()
    }

    /// number of selector columns
    pub fn num_selector_columns(&self) -> usize {
        self.params.num_selector_columns()
    }

    pub fn num_lk_selector_columns(&self) -> usize {
        self.params.num_lk_selector_columns()
    }

    /// number of witness columns
    pub fn num_witness_columns(&self) -> usize {
        self.params.num_witness_columns()
    }
}

/// The HyperPlonk proving key, consists of the following:
///   - the hyperplonk instance parameters
///   - the preprocessed polynomials output by the indexer
///   - the commitment to the selectors and permutations
///   - the parameters for polynomial commitment
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HyperPlonkProvingKey<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    /// Hyperplonk instance parameters
    pub params: HyperPlonkParams,

    /// The preprocessed permutation polynomials
    pub permutation_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
    /// The preprocessed selector polynomials
    pub selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
    /// The preprocessed lookup selector polynomials
    pub lk_selector_oracles: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,
    /// The preprocessed table polynomial
    pub table_oracle: Arc<DenseMultilinearExtension<E::ScalarField>>,

    /// Commitments to the preprocessed permutation polynomials
    pub permutation_commitments: Vec<PCS::Commitment>,
    /// Commitments to the preprocessed selector polynomials
    pub selector_commitments: Vec<PCS::Commitment>,
    /// Commitments to the preprocessed lookup selector polynomials
    pub lk_selector_commitments: Vec<PCS::Commitment>,
    /// Commitments to the preprocessed table polynomial
    pub table_commitment: PCS::Commitment,

    /// The parameters for PCS commitment
    pub pcs_param: PCS::ProverParam,
}

/// The HyperPlonk verifying key, consists of the following:
///   - the hyperplonk instance parameters
///   - the commitments to the preprocessed polynomials output by the indexer
///   - the parameters for polynomial commitment
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HyperPlonkVerifyingKey<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    /// Hyperplonk instance parameters
    pub params: HyperPlonkParams,
    /// The parameters for PCS commitment
    pub pcs_param: PCS::VerifierParam,
    /// Permutation oracles' commitments
    pub perm_commitments: Vec<PCS::Commitment>,
    /// A commitment to the preprocessed selector polynomials
    pub selector_commitments: Vec<PCS::Commitment>,
    /// A commitment to the preprocessed lookup selector polynomials
    pub lk_selector_commitments: Vec<PCS::Commitment>,
    /// A commitment to the preprocessed table polynomial
    pub table_commitment: PCS::Commitment,
}
