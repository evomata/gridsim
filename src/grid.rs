use {GetNeighbors, Sim};

use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::mem::ManuallyDrop;

/// Represents the state of the simulation.
///
/// This is not as efficient for Rule and is optimized for Sim.
#[derive(Clone, Debug)]
pub struct SquareGrid<S: Sim> {
    cells: Vec<S::Cell>,
    diffs: Vec<ManuallyDrop<(S::Diff, S::MoveNeighbors)>>,
    width: usize,
    height: usize,
}

impl<S: Sim> GetNeighbors<'static, usize, ()> for SquareGrid<S> {
    fn get_neighbors(&self, _: usize) {}
}

impl<S: Sim> SquareGrid<S> {
    /// Make a new grid using the Cell's Default impl.
    pub fn new(width: usize, height: usize) -> SquareGrid<S>
    where
        S::Cell: Default,
    {
        SquareGrid {
            cells: (0..)
                .take(width * height)
                .map(|_| S::Cell::default())
                .collect(),
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a new grid by cloning a default Cell.
    pub fn new_default(width: usize, height: usize, default: S::Cell) -> SquareGrid<S>
    where
        S::Cell: Clone,
    {
        SquareGrid {
            cells: ::std::iter::repeat(default).take(width * height).collect(),
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a new grid directly from an initial iter.
    pub fn new_iter<I>(width: usize, height: usize, iter: I) -> SquareGrid<S>
    where
        I: IntoIterator<Item = S::Cell>,
    {
        let cells: Vec<_> = iter.into_iter().take(width * height).collect();
        // Assert that they provided enough cells. If they didn't the simulation would panic.
        assert_eq!(
            cells.len(),
            width * height,
            "gridsim::Grid::new_iter: not enough cells provided in iter"
        );
        SquareGrid {
            cells: cells,
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a grid by evaluating each centered signed coordinate to a cell with a closure.
    pub fn new_coord_map<F>(width: usize, height: usize, mut coord_map: F) -> SquareGrid<S>
    where
        F: FnMut(isize, isize) -> S::Cell,
    {
        Self::new_iter(
            width,
            height,
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(move |(x, y)| {
                    coord_map(
                        x as isize - width as isize / 2,
                        y as isize - height as isize / 2,
                    )
                }),
        )
    }

    /// Make a grid using a collection of centered signed coordinates with associated cells.
    pub fn new_coords<I>(width: usize, height: usize, coords: I) -> SquareGrid<S>
    where
        I: IntoIterator<Item = ((isize, isize), S::Cell)>,
        S::Cell: Default,
    {
        use std::collections::HashMap;
        let coords: &mut HashMap<(isize, isize), S::Cell> = &mut coords.into_iter().collect();
        Self::new_coord_map(width, height, |x, y| {
            coords.remove(&(x, y)).unwrap_or_default()
        })
    }

    /// Make a grid using a collection of centered signed coordinates that indicate true cells.
    pub fn new_true_coords<I>(width: usize, height: usize, coords: I) -> SquareGrid<S>
    where
        I: IntoIterator<Item = (isize, isize)>,
        S: Sim<Cell = bool>,
    {
        Self::new_coords(width, height, coords.into_iter().map(|c| (c, true)))
    }

    /// Offset an index to a new index.
    #[inline]
    pub fn delta_index(&self, i: usize, delta: (isize, isize)) -> usize {
        // Wrap width and height to be in the range (-dim, dim).
        let x = delta.0 % self.width as isize;
        let y = delta.1 % self.height as isize;

        ((i + self.size()) as isize + x + y * self.width as isize) as usize % self.size()
    }

    /// Get a &Cell. Panics if out of bounds.
    #[inline]
    pub fn get_cell(&self, i: usize) -> &S::Cell {
        &self.cells[i]
    }

    /// This can only be called in the trait `TakeMoveDirection` when implmenting a new `Neighborhood`.
    #[inline]
    pub unsafe fn get_move_neighbors(&self, i: usize) -> &S::MoveNeighbors {
        &self.diffs[i].1
    }

    /// Get the Grid's Cell slice.
    #[inline]
    pub fn get_cells(&self) -> &[S::Cell] {
        &self.cells[..]
    }

    /// Get the Grid's Cell slice mutably.
    #[inline]
    pub fn get_cells_mut(&mut self) -> &mut [S::Cell] {
        &mut self.cells[..]
    }

    /// Get the Grid's width.
    #[inline]
    pub fn get_width(&self) -> usize {
        self.width
    }

    /// Get the Grid's height.
    #[inline]
    pub fn get_height(&self) -> usize {
        self.height
    }

    /// Get the Grid's size.
    #[inline]
    pub fn size(&self) -> usize {
        self.width * self.height
    }
}

// impl<'a, S, C, D, M, N, MN, IN, IMN> SquareGrid<S>
// where
//     IN: Into<N>,
//     IMN: Into<MN>,
//     S: Sim<Cell = C, Diff = D, Move = M, Neighbors = N, MoveNeighbors = MN>,
//     S::Cell: Sync + Send,
//     S::Diff: Sync + Send,
//     S::Move: Sync + Send,
//     S::Neighbors: Sync + Send,
//     S::MoveNeighbors: Sync + Send,
//     Self: GetNeighbors<'a, usize, IN>,
// {
//     /// Run the Grid for one cycle and parallelize the simulation.
//     pub fn cycle(&'a mut self) {
//         self.step();
//         self.update();
//     }

//     fn step(&'a mut self) {
//         self.diffs = {
//             let cs = |i| &self.cells[i % self.size()];
//             (0..self.size())
//                 .into_par_iter()
//                 .map(|i| Sim::step(&self.cells[i % self.size()], self.get_neighbors(i).into()))
//                 .collect()
//         };
//     }

//     fn update(&'a mut self) {
//         let mut diffs = Default::default();
//         ::std::mem::swap(&mut diffs, &mut self.diffs);
//         self.cells[..]
//             .par_iter_mut()
//             .zip(diffs.into_par_iter())
//             .for_each(|(cell, diff)| {
//                 S::update(cell, diff);
//             });
//     }
// }
