use crate::Sim;
use ndarray::{par_azip, s, Array2, ArrayView2, Zip, azip};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

/// Represents the state of the simulation.
#[derive(Clone, Debug)]
pub struct SquareGrid<S>
where
    S: Sim,
{
    sim: S,
    cells: Array2<S::Cell>,
    diffs: Array2<S::Diff>,
    flows: Array2<S::Neighborhood<S::Flow>>,
}

impl<S> SquareGrid<S>
where
    S: Sim,
    S::Neighborhood<S::Cell> = Array2<S::Cell>,
{
    /// Make a new grid using the default cell as padding.
    #[allow(clippy::reversed_empty_ranges)]
    pub fn new_default_padding(sim: S, original_cells: Array2<S::Cell>) -> Self
    where
        S::Cell: Default,
    {
        Self::new_padding_with(sim, original_cells, Default::default)
    }

    /// Make a new grid using a function to generate cells as padding.
    #[allow(clippy::reversed_empty_ranges)]
    pub fn new_padding_with(
        sim: S,
        mut original_cells: Array2<S::Cell>,
        padding: impl FnMut() -> S::Cell,
    ) -> Self {
        let dims = original_cells.dim();
        let mut cells = Array2::from_shape_simple_fn((dims.0 + 2, dims.1 + 2), padding);
        azip!((dest in &mut cells.slice_mut(s![1..-1, 1..-1]), cell in &mut original_cells) {
            std::mem::swap(dest, cell);
        });
        Self::new_including_padding(sim, cells)
    }

    /// Make a new grid with the padding included.
    #[allow(clippy::reversed_empty_ranges)]
    pub fn new_including_padding(sim: S, cells: Array2<S::Cell>) -> Self {
        let dims = cells.dim();
        assert!(dims.0 >= 3, "grid is empty, which isnt allowed");
        assert!(dims.1 >= 3, "grid is empty, which isnt allowed");
        SquareGrid { sim, cells }
    }
}

impl<S> SquareGrid<S>
where
    S: Sim,
{
    pub(crate) fn compute(&mut self) -> Array2<S::Diff> {
        // Somehow need to toroidially wrap the windows adapter O.o.
        Zip::from(self.cells.windows((3, 3))).par_apply_collect(||)
        self.diffs = self.cells[..]
            .par_iter()
            .enumerate()
            .map(|(ix, c)| self.single_step(ix, c))
            .map(ManuallyDrop::new)
            .collect();
    }
}
