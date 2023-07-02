// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements useful functions for the permutation check protocol.

use crate::poly_iop::errors::PolyIOPErrors;
use arithmetic::identity_permutation_mles;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Returns the evaluations of two list of MLEs:
/// - numerators = (a1, ..., ak)
/// - denominators = (b1, ..., bk)
///
///  where
///  - beta and gamma are challenges
///  - (f1, ..., fk), (g1, ..., gk),
///  - (s_id1, ..., s_idk), (perm1, ..., permk) are mle-s
///
/// - ai(x) is the MLE for `fi(x) + \beta s_id_i(x) + \gamma`
/// - bi(x) is the MLE for `gi(x) + \beta perm_i(x) + \gamma`
///
/// The caller is responsible for sanity-check
#[allow(clippy::type_complexity)]
pub(super) fn computer_nums_and_denoms<F: PrimeField>(
    beta: &F,
    gamma: &F,
    fxs: &[Arc<DenseMultilinearExtension<F>>],
    gxs: &[Arc<DenseMultilinearExtension<F>>],
    perms: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<
    (
        Vec<Arc<DenseMultilinearExtension<F>>>,
        Vec<Arc<DenseMultilinearExtension<F>>>,
    ),
    PolyIOPErrors,
> {
    let start = start_timer!(|| "compute numerators and denominators");

    let num_vars = fxs[0].num_vars;
    let s_ids = identity_permutation_mles::<F>(num_vars, fxs.len());

    #[cfg(feature = "parallel")]
    return {
        let numerators = fxs
            .par_iter()
            .zip(s_ids.par_iter())
            .map(|(f, s_id)| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    f.evaluations
                        .par_iter()
                        .zip(s_id.evaluations.par_iter())
                        .map(|(&f_ev, &s_id_ev)| f_ev + *beta * s_id_ev + gamma)
                        .collect::<Vec<_>>(),
                ))
            })
            .collect::<Vec<_>>();

        let denominators = gxs
            .par_iter()
            .zip(perms.par_iter())
            .map(|(g, perm)| {
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    num_vars,
                    g.evaluations
                        .par_iter()
                        .zip(perm.evaluations.par_iter())
                        .map(|(&g_ev, &perm_ev)| g_ev + *beta * perm_ev + gamma)
                        .collect::<Vec<_>>(),
                ))
            })
            .collect::<Vec<_>>();

        end_timer!(start);
        Ok((numerators, denominators))
    };

    #[cfg(not(feature = "parallel"))]
    {
        let mut numerators = vec![];
        let mut denominators = vec![];

        for l in 0..fxs.len() {
            let mut numerator_evals = vec![];
            let mut denominator_evals = vec![];

            for (&f_ev, (&g_ev, (&s_id_ev, &perm_ev))) in fxs[l]
                .iter()
                .zip(gxs[l].iter().zip(s_ids[l].iter().zip(perms[l].iter())))
            {
                let numerator = f_ev + *beta * s_id_ev + gamma;
                let denominator = g_ev + *beta * perm_ev + gamma;

                numerator_evals.push(numerator);
                denominator_evals.push(denominator);
            }
            numerators.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                numerator_evals,
            )));
            denominators.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars,
                denominator_evals,
            )));
        }

        end_timer!(start);
        Ok((numerators, denominators))
    }
}
