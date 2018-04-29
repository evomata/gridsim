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

#![feature(plugin)]
#![plugin(clippy)]

extern crate rayon;
#[macro_use]
extern crate enum_iterator_derive;

mod grid;
mod neighborhood;

pub mod moore;
pub mod neumann;

pub use grid::*;
pub use neighborhood::*;

/// Defines a simulation for simple things like cellular automata.
pub trait Rule<'a> {
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
pub trait Sim<'a> {
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
    fn step(&Self::Cell, Self::Neighbors) -> (Self::Diff, Self::MoveNeighbors);

    /// Updates a cell with a diff and movements into this cell.
    fn update(&mut Self::Cell, Self::Diff, Self::MoveNeighbors);
}

pub trait TakeDiff<Idx, Diff> {
    /// This should be called exactly once for every index, making it unsafe.
    ///
    /// This is marked unsafe to ensure people read the documentation due to the above requirement.
    unsafe fn take_diff(&self, Idx) -> Diff;
}

impl<'a, R, C, N> Sim<'a> for R
where
    R: Rule<'a, Cell = C, Neighbors = N>,
    C: Clone,
{
    type Cell = C;
    type Diff = C;
    type Move = ();

    type Neighbors = N;
    type MoveNeighbors = ();

    fn step(this: &C, neighbors: N) -> (C, ()) {
        (Self::rule(this.clone(), neighbors), ())
    }

    fn update(cell: &mut C, next: C, _: ()) {
        *cell = next;
    }
}

/// Conway's Game of Life
#[derive(Debug)]
pub enum GOL {}

impl<'a> Rule<'a> for GOL {
    type Cell = bool;
    type Neighbors = neumann::Neighbors<&'a bool>;

    fn rule(cell: bool, neighbors: Self::Neighbors) -> bool {
        let n = neighbors.iter().filter(|&&c| c).count();
        if cell {
            n >= 2 && n <= 3
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
        let mut grid = SquareGrid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (0, n)));

        grid.cycle();

        assert_eq!(
            grid.get_cells(),
            SquareGrid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (n, 0))).get_cells()
        )
    }
}
