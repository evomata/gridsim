//! Gridsim is a library intended to run grid-based simulations.
//!
//! The new generics introduced in gridsim 0.2.0 make it possible to implement hex grids,
//! rhombic dodecahedral honeycombs(in its multiple tight-pack layer patterns), square grids, cube grids,
//! and even n-dimensional grids, but they are currently not yet implemented.

#![feature(type_alias_impl_trait)]
#![feature(generic_associated_types)]
#![allow(incomplete_features)]

mod neumann;
mod square_grid;

pub use neumann::*;
pub use square_grid::*;

pub trait Neighborhood {
    type Neighbors<'a, T: 'a>;
    type Edges<T>;
}

/// Defines a simulation for complicated things that have too much state to abandon on the next cycle.
///
/// This enforces a rule in that all new cells are only produced from old board state. This prevents the
/// update order from breaking the simulation.
pub trait Sim<N>
where
    N: Neighborhood,
{
    /// The cells of the grid
    type Cell: 'static;
    /// Result of the neighbor-observing computation
    type Diff: 'static;
    /// The data which flows to each neighbor.
    type Flow;

    /// At this stage, everything is immutable, and the diff can be computed that
    /// describes what will change between simulation states.
    fn compute(&self, cells: N::Neighbors<'_, Self::Cell>) -> Self::Diff;

    /// At this stage, changes are made to the cell based on the diff and then
    /// any owned state that needs to be moved to neighbors must be returned
    /// as part of the flow.
    fn egress(
        &self,
        cell: &mut Self::Cell,
        diffs: N::Neighbors<'_, Self::Diff>,
    ) -> N::Edges<Self::Flow>;

    /// At this stage, the flow is received from all neighbors, allowing state
    /// to be added to this cell.
    fn ingress(&self, cell: &mut Self::Cell, flows: N::Edges<Self::Flow>);

    /// The cell used as padding.
    fn cell_padding(&self) -> Self::Cell;

    /// The diff used as padding.
    fn diff_padding(&self) -> Self::Diff;

    /// The flow used as padding.
    fn flow_padding(&self) -> Self::Flow;
}
