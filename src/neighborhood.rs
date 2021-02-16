use crate::Direction;
use std::ops::IndexMut;

/// A `Neighborhood` contains all of your neighbors, which are each in their own `Direction`.
pub trait Neighborhood<T>: IndexMut<Self::Direction> {
    type Direction: Direction;
    type Iter: Iterator<Item = T>;
    type DirIter: Iterator<Item = (Self::Direction, T)>;

    fn new<F: FnMut(Self::Direction) -> T>(dir_map: F) -> Self;

    // Swaps the thing with direction `dir` in self with the
    // thing in the opposite-facing direction in `other`.
    fn swap_adjacent(&mut self, other: &mut Self, dir: Self::Direction) {
        std::mem::swap(self[dir], other[dir.inv()]);
    }

    /// Iterate over all neighbor cells.
    fn iter(self) -> Self::Iter;
    /// Iterate over all neighbor cells with their directions.
    fn dir_iter(self) -> Self::DirIter;
}
