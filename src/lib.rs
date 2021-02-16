//! Gridsim is a library intended to run grid-based simulations.
//!
//! The new generics introduced in gridsim 0.2.0 make it possible to implement hex grids,
//! rhombic dodecahedral honeycombs(in its multiple tight-pack layer patterns), square grids, cube grids,
//! and even n-dimensional grids, but they are currently not yet implemented.

#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]
#![allow(incomplete_features)]

mod direction;
mod grid;
mod neighborhood;

pub mod moore;
pub mod neumann;

pub use direction::*;
pub use grid::*;
pub use neighborhood::*;

/// Defines a simulation for simple things like cellular automata.
pub trait Rule {
    /// The type of cells on the grid
    type Cell;
    /// The neighborhood of the rule
    type Neighbors<'a>;

    /// This defines a rule for how a cell and its neighbors transform into a new cell.
    fn rule<'a>(&self, cell: &Self::Cell, neighbors: Self::Neighbors<'a>) -> Self::Cell;
}

/// Defines a simulation for complicated things that have too much state to abandon on the next cycle.
///
/// This enforces a rule in that all new cells are only produced from old board state. This prevents the
/// update order from breaking the simulation.
pub trait Sim {
    /// Cells on the grid
    type Cell;
    /// Result of the neighbor-observing computation
    type Diff;

    /// Neighborhood of cells
    type Neighbors<'a>;
    /// Nighborhood of moving data
    type Flow;

    /// At this stage, everything is immutable, and the diff can be computed that
    /// describes what will change between simulation states.
    fn compute<'a>(&self, cell: &Self::Cell, neighbors: Self::Neighbors<'a>) -> Self::Diff;

    /// At this stage, changes are made to the cell based on the diff and then
    /// any owned state that needs to be moved to neighbors must be returned
    /// as part of the flow.
    fn egress(&self, cell: &mut Self::Cell, diff: Self::Diff) -> Self::Flow;

    /// At this stage, the flow is received from all neighbors, allowing state
    /// to be added to this cell.
    fn ingress(&self, cell: &mut Self::Cell, flow: Self::Flow);
}

impl<R> Sim for R
where
    R: Rule,
{
    type Cell = R::Cell;
    type Diff = R::Cell;

    type Neighbors<'a> = R::Neighbors<'a>;
    type Flow = ();

    fn compute<'a>(&self, cell: &Self::Cell, neighbors: Self::Neighbors<'a>) -> Self::Diff {
        self.rule(cell, neighbors)
    }

    fn egress(&self, cell: &mut Self::Cell, diff: Self::Diff) {
        *cell = diff;
    }

    fn ingress(&self, _: &mut Self::Cell, flow: ()) {}
}

/// Conway's Game of Life
#[derive(Debug)]
pub struct Gol;

impl Rule for Gol {
    type Cell = bool;
    type Neighbors<'a> = neumann::NeumannNeighbors<&'a bool>;

    #[inline]
    fn rule<'a>(&self, &cell: &bool, neighbors: neumann::NeumannNeighbors<&'a bool>) -> bool {
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
    use ndarray::Array2;

    use super::*;

    #[test]
    fn gol_blinker() {
        let mut grid = SquareGrid::new(
            Gol,
            Array2::from_shape_fn((5, 5), |(y, x)| y == 2 && x >= 1 && x <= 3),
        );

        // grid.cycle();

        // assert_eq!(
        //     grid.cells(),
        //     SquareGrid::<Gol>::new_true_coords(5, 5, (-1..2).map(|n| (n, 0))).cells()
        // )
    }
}
