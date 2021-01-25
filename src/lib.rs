//! Gridsim is a library intended to run grid-based simulations.
//!
//! The new generics introduced in gridsim 0.2.0 make it possible to implement hex grids,
//! rhombic dodecahedral honeycombs(in its multiple tight-pack layer patterns), square grids, cube grids,
//! and even n-dimensional grids, but they are currently not yet implemented.

#![feature(test)]

extern crate rayon;
#[macro_use]
extern crate enum_iterator_derive;

mod grid;
#[cfg(feature = "multinode")]
mod multigrid;
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
    fn rule(cell: Self::Cell, neighbors: Self::Neighbors) -> Self::Cell;
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

    /// Performs one step of the simulation, creating diffs and movements that go out to neighbors.
    fn step(cell: &Self::Cell, neighbors: Self::Neighbors) -> (Self::Diff, Self::MoveNeighbors);

    /// Updates a cell with a diff and movements into this cell.
    /// Note that these movements are the ones produced in each neighboring cell.
    fn update(cell: &mut Self::Cell, diffs: Self::Diff, move_neighbors: Self::MoveNeighbors);
}

pub trait TakeDiff<Idx, Diff> {
    /// # Safety
    ///
    /// This should be called exactly once for every index, making it unsafe.
    ///
    /// This is marked unsafe to ensure people read the documentation due to the above requirement.
    unsafe fn take_diff(&self, ix: Idx) -> Diff;
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

    #[inline]
    fn step(this: &C, neighbors: N) -> (C, ()) {
        (Self::rule(this.clone(), neighbors), ())
    }

    #[inline]
    fn update(cell: &mut C, next: C, _: ()) {
        *cell = next;
    }
}

/// Conway's Game of Life
#[derive(Debug)]
pub enum GOL {}

impl<'a> Rule<'a> for GOL {
    type Cell = bool;
    type Neighbors = neumann::NeumannNeighbors<&'a bool>;

    #[inline]
    fn rule(cell: bool, neighbors: Self::Neighbors) -> bool {
        let n = neighbors.iter().filter(|&&c| c).count();
        if cell {
            (2..=3).contains(&n)
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
            grid.cells(),
            SquareGrid::<GOL>::new_true_coords(5, 5, (-1..2).map(|n| (n, 0))).cells()
        )
    }
}
