#![allow(clippy::reversed_empty_ranges)]

use crate::{Neighborhood, Neumann, Sim};
use ndarray::{azip, par_azip, s, Array2, ArrayView2, Zip};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

/// Represents the state of the simulation.
#[derive(Clone, Debug)]
pub struct SquareGrid<S>
where
    S: Sim<Neumann>,
{
    sim: S,
    cells: Array2<S::Cell>,
}

impl<S> SquareGrid<S>
where
    S: Sim<Neumann>,
    S::Cell: Send,
{
    /// Make a new grid with the given cells.
    pub fn new(sim: S, mut original_cells: Array2<S::Cell>) -> Self {
        let dims = original_cells.dim();
        let mut cells =
            Array2::from_shape_simple_fn((dims.0 + 2, dims.1 + 2), || sim.cell_padding());
        par_azip!((dest in &mut cells.slice_mut(s![1..-1, 1..-1]), cell in &mut original_cells) {
            std::mem::swap(dest, cell);
        });
        assert!(
            dims.0 >= 1 && dims.1 >= 1,
            "grid is empty, which isnt allowed"
        );
        Self { sim, cells }
    }
}

impl<S> SquareGrid<S>
where
    S: Sim<Neumann> + Send + Sync,
    S::Cell: Send + Sync,
    S::Diff: Send + Sync,
    S::Flow: Send,
{
    fn step(&mut self) {
        let diffs = self.compute_diffs();
        let flows = self.perform_egress(diffs.view());
    }

    fn compute_diffs(&self) -> Array2<S::Diff> {
        let mut diffs = Array2::from_shape_simple_fn(self.cells.dim(), || self.sim.diff_padding());
        par_azip!((diff in diffs.slice_mut(s![1..-1, 1..-1]), cell in self.cells.windows((3, 3))) {
            *diff = self.sim.compute(cell);
        });
        diffs
    }

    fn perform_egress(&mut self, diffs: ArrayView2<'_, S::Diff>) -> Array2<[S::Flow; 8]> {
        let mut flows = Array2::from_shape_simple_fn(self.cells.dim(), || {
            [
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
            ]
        });
        let sim = &self.sim;
        par_azip!((flow in flows.slice_mut(s![1..-1, 1..-1]), cell in self.cells.slice_mut(s![1..-1, 1..-1]), diffs in diffs.windows((3, 3))) {
            *flow = sim.egress(cell, diffs);
        });
        flows
    }
}
