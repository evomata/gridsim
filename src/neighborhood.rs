pub trait Direction: Sized {
    type Directions: Iterator<Item = Self>;

    fn inv(self) -> Self;
    fn directions() -> Self::Directions;
}

pub trait Neighborhood<T> {
    type Direction: Direction;
    type Iter: Iterator<Item = T>;
    type DirIter: Iterator<Item = (Self::Direction, T)>;

    fn new<F: FnMut(Self::Direction) -> T>(F) -> Self;

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
    unsafe fn take_move_direction(&self, Idx, Dir) -> Move;
}

pub trait TakeMoveNeighbors<Idx, MoveNeighbors> {
    /// This should be called exactly once for every index, making it unsafe.
    ///
    /// This is marked unsafe to ensure people read the documentation due to the above requirement.
    unsafe fn take_move_neighbors(&self, Idx) -> MoveNeighbors;
}
