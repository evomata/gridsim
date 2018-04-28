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
#[macro_use]
extern crate enum_iterator_derive;

pub mod moore;
mod neighborhood;
pub mod neumann;

pub use neighborhood::*;

/// Defines a simulation for simple things like cellular automata.
pub trait Rule {
    /// The type of cells on the grid
    type Cell;
    /// The neighborhood of the rule
    type Neighbors;

    /// This defines a rule for how a cell and its neighbors transform into a new cell.
    fn rule(Self::Cell, neighbors: Self::Neighbors) -> Self::Cell;
}

/// Defines a simulation for complicated things that have too much state to abandon on the next cycle.
///
/// This enforces a rule in that all new cells are only produced from old board state. This prevents the
/// update order from breaking the simulation.
pub trait Sim {
    /// The type of cells on the grid
    type Cell;
    /// Represents all information necessary to modify a cell in the previous grid to produce the version in the next
    type Diff;
    /// Data that moves between cells
    type Move;

    /// Neighborhood of cells.
    type Neighbors;
    /// Nighborhood of moving data
    type MoveNeighbors;

    /// Performs one step of the simulation.
    fn step(&Self::Cell, neighbors: Self::Neighbors) -> (Self::Diff, Self::MoveNeighbors);

    /// Updates a cell with a diff and movements into this cell.
    fn update(&mut Self::Cell, Self::Diff, Self::MoveNeighbors);
}

/// Represents the state of the simulation.
///
/// This is not as efficient for Rule and is optimized for Sim.
#[derive(Clone, Debug)]
pub struct SquareGrid<S: Sim> {
    cells: Vec<S::Cell>,
    diffs: Vec<(S::Diff, S::MoveNeighbors)>,
    width: usize,
    height: usize,
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

    /// Get a &Cell by wrapped index.
    #[inline]
    pub fn get_cell(&self, i: usize) -> &S::Cell {
        &self.cells[i % self.size()]
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
    type Neighbors = neumann::Neighbors<bool>;

    fn rule(cell: bool, neighbors: Self::Neighbors) -> bool {
        let n = neighbors.iter().filter(|&c| c).count();
        if cell {
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
