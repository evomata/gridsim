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
    /// An intermediary result used to permit asynchronous processing of the inputs.
    type Async;
    /// Represents all information necessary to modify a cell in the previous grid to produce the version in the next
    type Diff;
    /// Data that moves between cells
    type Move;

    /// Neighborhood of cells.
    type Neighbors;
    /// Nighborhood of moving data
    type MoveNeighbors;

    /// Processes the cell and its neighbors to begin an asynchronous operation.
    fn process(&Self::Cell, Self::Neighbors) -> Self::Async;

    /// Resolves the asynchronous operation, creating diffs and movements that go out to neighbors.
    fn step(Self::Async) -> (Self::Diff, Self::MoveNeighbors);

    /// Updates a cell with a diff and movements into this cell.
    /// Note that these movements are the ones produced in each neighboring cell.
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
    type Async = C;
    type Diff = C;
    type Move = ();

    type Neighbors = N;
    type MoveNeighbors = ();

    #[inline]
    fn process(this: &C, neighbors: N) -> C {
        Self::rule(this.clone(), neighbors)
    }

    #[inline]
    fn step(this: C) -> (C, ()) {
        (this, ())
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
    type Neighbors = neumann::Neighbors<&'a bool>;

    #[inline]
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
    extern crate test;
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

    #[bench]
    fn gol_r_pentomino(b: &mut test::Bencher) {
        let mut grid = SquareGrid::<GOL>::new_true_coords(
            256,
            256,
            vec![(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
        );
        b.iter(|| {
            grid.cycle();
            *grid.get_cell(0)
        });
    }
}
