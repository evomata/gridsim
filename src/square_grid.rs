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
{
    /// Make a new grid with the given cells.
    #[allow(clippy::reversed_empty_ranges)]
    pub fn new(sim: S, mut original_cells: Array2<S::Cell>) -> Self {
        let dims = original_cells.dim();
        let mut cells =
            Array2::from_shape_simple_fn((dims.0 + 2, dims.1 + 2), || sim.cell_padding());
        azip!((dest in &mut cells.slice_mut(s![1..-1, 1..-1]), cell in &mut original_cells) {
            std::mem::swap(dest, cell);
        });
        assert!(
            dims.0 >= 1 && dims.1 >= 1,
            "grid is empty, which isnt allowed"
        );
        Self { sim, cells }
    }
}

// impl<S> SquareGrid<S>
// where
//     S: SimNeighbors,
// {
//     pub(crate) fn compute(&mut self) -> Array2<S::Diff> {
//         // Somehow need to toroidially wrap the windows adapter O.o.
//         Zip::from(self.cells.windows((3, 3))).par_apply_collect(||)
//         self.diffs = self.cells[..]
//             .par_iter()
//             .enumerate()
//             .map(|(ix, c)| self.single_step(ix, c))
//             .map(ManuallyDrop::new)
//             .collect();
//     }
// }
