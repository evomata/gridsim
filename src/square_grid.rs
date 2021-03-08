#![allow(clippy::reversed_empty_ranges)]

use crate::{Neumann, Sim};
use ndarray::{par_azip, s, Array2, ArrayView2, ArrayViewMut2};
use std::{
    cell::UnsafeCell,
    mem::{self, ManuallyDrop},
};
use itertools::Itertools;

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
        assert!(
            dims.0 >= 1 && dims.1 >= 1,
            "grid is empty, which isnt allowed"
        );
        let mut cells =
            Array2::from_shape_simple_fn((dims.0 + 2, dims.1 + 2), || sim.cell_padding());
        par_azip!((dest in &mut cells.slice_mut(s![1..-1, 1..-1]), cell in &mut original_cells) {
            mem::swap(dest, cell);
        });
        Self { sim, cells }
    }

    /// Get view of cells on the grid.
    pub fn cells(&self) -> ArrayView2<'_, S::Cell> {
        self.cells.slice(s![1..-1, 1..-1])
    }

    /// Get mutable view of cells on the grid.
    pub fn cells_mut(&mut self) -> ArrayViewMut2<'_, S::Cell> {
        self.cells.slice_mut(s![1..-1, 1..-1])
    }
}

#[cfg(feature = "use-rayon")]
impl<S> SquareGrid<S>
where
    S: Sim<Neumann> + Sync,
    S::Cell: Send + Sync,
    S::Diff: Send + Sync,
    S::Flow: Send,
{
    pub fn step_parallel(&mut self) {
        let diffs = self.compute_diffs();
        let flows = self.perform_egress(diffs.view());
        self.perform_ingress(flows);
    }

    fn compute_diffs(&self) -> Array2<S::Diff> {
        let mut diffs = Array2::from_shape_simple_fn(self.cells.dim(), || self.sim.diff_padding());
        par_azip!((diff in diffs.slice_mut(s![1..-1, 1..-1]), cell in self.cells.windows((3, 3))) {
            *diff = self.sim.compute(cell);
        });
        diffs
    }

    fn perform_egress(
        &mut self,
        diffs: ArrayView2<'_, S::Diff>,
    ) -> Array2<ManuallyDrop<UnsafeCell<[S::Flow; 8]>>> {
        let mut flows = Array2::from_shape_simple_fn(self.cells.dim(), || {
            ManuallyDrop::new(UnsafeCell::new([
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
                self.sim.flow_padding(),
            ]))
        });
        let sim = &self.sim;
        par_azip!((flow in flows.slice_mut(s![1..-1, 1..-1]), cell in self.cells.slice_mut(s![1..-1, 1..-1]), diffs in diffs.windows((3, 3))) {
            *flow.get_mut() = sim.egress(cell, diffs);
        });

        unsafe fn exchange_chunk<T>(chunk: ArrayViewMut2<'_, ManuallyDrop<UnsafeCell<[T; 8]>>>) {
            let top_left = &mut *chunk[(0, 0)].get();
            let top_right = &mut *chunk[(0, 1)].get();
            let bottom_left = &mut *chunk[(1, 0)].get();
            let bottom_right = &mut *chunk[(1, 1)].get();
            // Left to right
            mem::swap(&mut bottom_left[0], &mut bottom_right[4]);
            // Top to bottom
            mem::swap(&mut top_right[6], &mut bottom_right[2]);
            // Across
            mem::swap(&mut top_left[7], &mut bottom_right[3]);
            mem::swap(&mut top_right[5], &mut bottom_left[1]);
        }

        // The flows need to be moved around to where they are consumed.
        // We do this by splitting the grid into 2x2 chunks.
        // We exchange the diagnal movements in each chunk, and then we also
        // exchange two specific sides (but the specific sides can be chosen arbitrarily).
        // This performs four swaps per chunk, exchanging eight flows.
        // At this point we have only exchanged 1/4th of the total flows, but by sequentially
        // performing this same operation offset by (0, 0), (0, 1), (1, 0), and (1, 1)
        // we can actually exchange all flows in four simple parallel operations.
        for (y, x) in (0..2).cartesian_product(0..2) {
            par_azip!((chunk in flows.slice_mut(s![y.., x..]).exact_chunks_mut((2, 2))) {
                unsafe { exchange_chunk(chunk); }
            });
        }

        flows
    }

    fn perform_ingress(&mut self, mut flows: Array2<ManuallyDrop<UnsafeCell<[S::Flow; 8]>>>) {
        let (h, w) = self.cells.dim();
        let sim = &self.sim;
        // At the end of this line, all of the manually drops MUST have been taken or dropped.
        par_azip!((index (y, x), flow in &mut flows, cell in &mut self.cells) {
            unsafe {
                if (1..h-1).contains(&y) && (1..w-1).contains(&x) {
                    // If its not part of the padding, we run the sim here.
                    sim.ingress(cell, ManuallyDrop::take(flow).into_inner());
                } else {
                    // If this is part of the padding, we must manually drop.
                    ManuallyDrop::drop(flow);
                }
            }
        });
    }
}
