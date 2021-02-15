//! Gridsim is a library intended to run grid-based simulations.
//!
//! The new generics introduced in gridsim 0.2.0 make it possible to implement hex grids,
//! rhombic dodecahedral honeycombs(in its multiple tight-pack layer patterns), square grids, cube grids,
//! and even n-dimensional grids, but they are currently not yet implemented.

#![feature(type_alias_impl_trait)]

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
    fn rule(cell: Self::Cell, neighbors: Self::Neighbors) -> Self::Cell;
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
    type Neighbors;
    /// Nighborhood of moving data
    type Flow;

    /// At this stage, everything is immutable, and the diff can be computed that
    /// describes what will change between simulation states.
    fn compute(&self, cell: &Self::Cell, neighbors: Self::Neighbors) -> Self::Diff;

    /// At this stage, changes are made to the cell based on the diff and then
    /// any owned state that needs to be moved to neighbors must be returned
    /// as part of the flow.
    fn egress(&self, cell: &mut Self::Cell, diff: Self::Diff) -> Self::Flow;

    /// At this stage, the flow is received from all neighbors, allowing state
    /// to be added to this cell.
    fn ingress(&self, cell: &mut Self::Cell, flow: Self::Flow);
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
