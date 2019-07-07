use boolinator::Boolinator;

pub trait Direction: Sized + From<usize> + Into<usize> {
    type Directions: Iterator<Item = Self>;

    /// An iterator over all directions. There must be an even number of directions and they
    /// must rotate counter-clockwise when iterating in order for all default impls to work.
    /// If your direction does not follow these rules then you must override the default impls.
    fn directions() -> Self::Directions;

    /// All directions should have an opposite direction on the opposite side of its iterator.
    fn inv(self) -> Self {
        let mut n: usize = self.into();
        n += Self::total() / 2;
        n %= Self::total();
        n.into()
    }

    /// The total number of directions.
    fn total() -> usize {
        Self::directions().count()
    }

    /// Must provide an input iterator with length of the number of directions for the `Direction` impl
    /// which contains sigmoid outputs in the range of (0.0, 1.0). This will choose a direction based on
    /// the highest sigmoid output. If none of the values are greater than 0.5, then it will choose no direction.
    #[inline]
    fn chooser(sigmoids: impl Iterator<Item = f32>) -> Option<(Self, float_ord::FloatOrd<f32>)> {
        sigmoids
            .map(float_ord::FloatOrd)
            .enumerate()
            .max_by_key(|&(_, n)| n)
            .and_then(|(ix, value)| (value.0 > 0.5).as_some((ix.into(), value)))
    }

    #[inline]
    fn turn_clockwise(self) -> Self {
        let mut n: usize = self.into();
        n += Self::total() - 1;
        n %= Self::total();
        n.into()
    }

    #[inline]
    fn turn_counterclockwise(self) -> Self {
        let mut n: usize = self.into();
        n += 1;
        n %= Self::total();
        n.into()
    }
}

/// A `Neighborhood` contains all of your neighbors, which are each in their own `Direction`.
pub trait Neighborhood<T>: std::iter::FromIterator<T> {
    type Direction: Direction;
    type Iter: Iterator<Item = T>;
    type DirIter: Iterator<Item = (Self::Direction, T)>;

    fn new<F: FnMut(Self::Direction) -> T>(dir_map: F) -> Self;

    /// Iterate over all neighbor cells.
    fn iter(self) -> Self::Iter;
    /// Iterate over all neighbor cells with their directions.
    fn dir_iter(self) -> Self::DirIter;
}

pub trait GetNeighbors<'a, Idx, Neighbors> {
    fn get_neighbors(&'a self, index: Idx) -> Neighbors;
}

pub trait TakeMoveDirection<Idx, Dir, Move> {
    /// This should be called exactly once for every index and direction.
    ///
    /// This is marked unsafe to ensure people read the documentation due to the above requirement.
    unsafe fn take_move_direction(&self, ix: Idx, dir: Dir) -> Move;
}

pub trait TakeMoveNeighbors<Idx, MoveNeighbors> {
    /// This should be called exactly once for every index, making it unsafe.
    ///
    /// This is marked unsafe to ensure people read the documentation due to the above requirement.
    unsafe fn take_move_neighbors(&self, ix: Idx) -> MoveNeighbors;
}
