use crate::Sim;

use ndarray::Array2;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::mem::ManuallyDrop;

/// Represents the state of the simulation.
#[derive(Clone, Debug)]
pub struct SquareGrid<S>
where
    S: Sim,
{
    sim: S,
    cells: Array2<S::Cell>,
    diffs: Array2<Option<S::Diff>>,
    flows: Array2<Option<S::Flow>>,
}

impl<S> SquareGrid<S>
where
    S: Sim,
{
    /// Make a new grid using the Cell's Default impl.
    pub fn new(sim: S, cells: Array2<S::Cell>) -> Self
    where
        S::Cell: Default,
    {
        let dims = cells.dim();
        SquareGrid {
            sim,
            cells,
            diffs: Array2::default(dims),
            flows: Array2::default(dims),
        }
    }

    /// Offset a position to a new position, making sure to wrap it appropriately.
    #[inline]
    pub fn offset(&self, position: [usize; 2], offset: [isize; 2]) -> [usize; 2] {
        let dims = self.cells.dim();
        [
            (position[0] + ((offset[0] % dims.0 as isize) + dims.0 as isize) as usize) % dims.0,
            (position[1] + ((offset[1] % dims.1 as isize) + dims.1 as isize) as usize) % dims.1,
        ]
    }
}

// impl<S> SquareGrid<S>
// where
//     S: Sim,
//     S::Cell: Sync + Send,
//     S::Diff: Sync + Send,
//     S::Neighbors: Sync + Send,
//     S::Flow: Sync + Send,
// {
//     /// Run the Grid for one cycle and parallelize the simulation.
//     pub fn cycle(&mut self) {
//         self.step();
//         self.update();
//     }

//     pub(crate) fn step(&mut self) {
//         self.diffs = self.cells[..]
//             .par_iter()
//             .enumerate()
//             .map(|(ix, c)| self.single_step(ix, c))
//             .map(ManuallyDrop::new)
//             .collect();
//     }

//     #[inline]
//     fn single_step(&self, ix: usize, c: &C) -> (S::Diff, S::MoveNeighbors) {
//         // TODO: Convey to the compiler this is okay without unsafe.
//         let grid = unsafe { &*(self as *const Self) };
//         S::step(c, grid.get_neighbors(ix))
//     }

//     pub(crate) fn update(&mut self) {
//         self.cells[..]
//             .par_iter()
//             .enumerate()
//             .for_each(|(ix, cell)| unsafe {
//                 #[allow(clippy::cast_ref_to_mut)]
//                 S::update(
//                     &mut *(cell as *const C as *mut C),
//                     self.take_diff(ix),
//                     (self as &Self).take_move_neighbors(ix),
//                 );
//             });
//         // This wont call any `drop()` because of the `ManuallyDrop`.
//         self.diffs.clear();
//     }
// }
