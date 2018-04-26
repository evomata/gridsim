//! Gridsim is a library intended to run grid-based simulations.
//! The library in its current form is not what it is intended to be.
//! Once [const generics](https://github.com/rust-lang/rust/issues/44580) are available in nightly, this library will
//! be turned into a library which is generic across all grid simulations. This will include hex grid,
//! rhombic dodecahedral honeycomb (in its multiple tight-pack layer patterns), square grid, cube grid,
//! and even n-dimensional grids. It will also be generic over the neighbor distance including
//! moore and von-neumann neighborhood. It should also eventually be given mechanisms to easier support
//! running on clusters.
//!
//! In its current early state, it will be used for 2d square grids only. The structure will be relatively
//! similar to the final form, but include none of the above features except for the simulation part.

extern crate rayon;

use rayon::prelude::*;

/// Defines a simulation for simple things like cellular automata.
pub trait Rule {
    /// The type of cells on the grid
    type Cell: Clone;

    /// This defines a rule for how cells in a 3x3 space transform into a new Cell in the center position
    /// of the new board.
    fn rule([[Self::Cell; 3]; 3]) -> Self::Cell;
}

/// Defines a simulation for complicated things that have too much state to abandon on the next cycle.
///
/// This enforces a rule in that all new cells are only produced from old board state. This prevents the
/// update order from breaking the simulation.
pub trait Sim {
    /// The type of cells on the grid
    type Cell;
    /// Represents all information necessary to modify a cell in the previous grid to produce the version in the next.
    type Diff;

    /// Performs one step of the simulation by producing a grid of diffs that can be used to change the cells to
    /// their next state.
    fn step([[&Self::Cell; 3]; 3]) -> Self::Diff;
    /// Updates a cell with a diff.
    fn update(&mut Self::Cell, Self::Diff);
}

impl<T> Sim for T
where
    T: Rule,
{
    type Cell = T::Cell;
    type Diff = T::Cell;

    #[inline]
    fn step(old: [[&Self::Cell; 3]; 3]) -> Self::Diff {
        Self::rule([
            [old[0][0].clone(), old[0][1].clone(), old[0][2].clone()],
            [old[1][0].clone(), old[1][1].clone(), old[1][2].clone()],
            [old[2][0].clone(), old[2][1].clone(), old[2][2].clone()],
        ])
    }

    #[inline]
    fn update(cell: &mut Self::Cell, diff: Self::Diff) {
        *cell = diff;
    }
}

/// Represents the state of the simulation.
///
/// This is not as efficient for Rule and is optimized for Sim.
#[derive(Clone, Debug)]
pub struct Grid<S: Sim> {
    cells: Vec<S::Cell>,
    diffs: Vec<S::Diff>,
    width: usize,
    height: usize,
}

impl<S: Sim> Grid<S> {
    /// Make a new grid using the Cell's Default impl.
    pub fn new(width: usize, height: usize) -> Grid<S>
    where
        S::Cell: Default,
    {
        Grid {
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
    pub fn new_default(width: usize, height: usize, default: S::Cell) -> Grid<S>
    where
        S::Cell: Clone,
    {
        Grid {
            cells: ::std::iter::repeat(default).take(width * height).collect(),
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a new grid directly from an initial iter.
    pub fn new_iter<I>(width: usize, height: usize, iter: I) -> Grid<S>
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
        Grid {
            cells: cells,
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a grid by evaluating each centered signed coordinate to a cell with a closure.
    pub fn new_coord_map<F>(width: usize, height: usize, mut coord_map: F) -> Grid<S>
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
    pub fn new_coords<I>(width: usize, height: usize, coords: I) -> Grid<S>
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
    pub fn new_true_coords<I>(width: usize, height: usize, coords: I) -> Grid<S>
    where
        I: IntoIterator<Item = (isize, isize)>,
        S: Sim<Cell = bool>,
    {
        Self::new_coords(width, height, coords.into_iter().map(|c| (c, true)))
    }

    /// Run the Grid for one cycle.
    pub fn cycle(&mut self) {
        self.step();
        self.update();
    }

    /// Run the Grid for one cycle and parallelize the simulation.
    pub fn cycle_par(&mut self)
    where
        S::Cell: Sync + Send,
        S::Diff: Sync + Send,
    {
        self.step_par();
        self.update_par();
    }

    fn step(&mut self) {
        self.diffs = {
            let cs = |i| &self.cells[i % self.size()];
            (0..self.size())
                .map(|i| {
                    [
                        [
                            cs(self.size() + i - 1 - self.width),
                            cs(self.size() + i - self.width),
                            cs(self.size() + i + 1 - self.width),
                        ],
                        [
                            cs(self.size() + i - 1),
                            cs(self.size() + i),
                            cs(self.size() + i + 1),
                        ],
                        [
                            cs(self.size() + i - 1 + self.width),
                            cs(self.size() + i + self.width),
                            cs(self.size() + i + 1 + self.width),
                        ],
                    ]
                })
                .map(S::step)
                .collect()
        };
    }

    fn step_par(&mut self)
    where
        S::Cell: Sync,
        S::Diff: Sync + Send,
    {
        self.diffs = {
            let cs = |i| &self.cells[i % self.size()];
            (0..self.size())
                .into_par_iter()
                .map(|i| {
                    [
                        [
                            cs(self.size() + i - 1 - self.width),
                            cs(self.size() + i - self.width),
                            cs(self.size() + i + 1 - self.width),
                        ],
                        [
                            cs(self.size() + i - 1),
                            cs(self.size() + i),
                            cs(self.size() + i + 1),
                        ],
                        [
                            cs(self.size() + i - 1 + self.width),
                            cs(self.size() + i + self.width),
                            cs(self.size() + i + 1 + self.width),
                        ],
                    ]
                })
                .map(S::step)
                .collect()
        };
    }

    fn update(&mut self) {
        for (cell, diff) in self.cells.iter_mut().zip(self.diffs.drain(..)) {
            S::update(cell, diff);
        }
    }

    fn update_par(&mut self)
    where
        S::Cell: Sync + Send,
        S::Diff: Sync + Send,
    {
        let mut diffs = Default::default();
        std::mem::swap(&mut diffs, &mut self.diffs);
        self.cells[..]
            .par_iter_mut()
            .zip(diffs.into_par_iter())
            .for_each(|(cell, diff)| {
                S::update(cell, diff);
            });
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

/// Conway's Game of Life
#[derive(Debug)]
pub enum GOL {}

impl Rule for GOL {
    type Cell = bool;

    fn rule(cells: [[bool; 3]; 3]) -> bool {
        let n = cells
            .iter()
            .flat_map(|cs| cs.iter())
            .filter(|&&c| c)
            .count();
        if cells[1][1] {
            n >= 3 && n <= 4
        } else {
            n == 3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gol_blinker() {
        let mut grid = Grid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (0, n)));

        grid.cycle();

        assert_eq!(
            grid.get_cells(),
            Grid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (n, 0))).get_cells()
        )
    }

    #[test]
    fn gol_blinker_par() {
        let mut grid = Grid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (0, n)));

        grid.cycle_par();

        assert_eq!(
            grid.get_cells(),
            Grid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (n, 0))).get_cells()
        )
    }
}
