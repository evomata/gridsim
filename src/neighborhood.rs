pub trait Direction: Sized {
    type Directions: Iterator<Item = Self>;

    fn inv(self) -> Self;
    fn directions() -> Self::Directions;
}

pub trait Neighborhood<T> {
    type Direction: Direction;
    type Iter: Iterator<Item = T>;
    type DirIter: Iterator<Item = (Self::Direction, T)>;

    /// Iterate over all neighbor cells.
    fn iter(self) -> Self::Iter;
    /// Iterate over all neighbor cells with their directions.
    fn dir_iter(self) -> Self::DirIter;
}

pub trait GetNeighbors<'a, Idx, Neighbors> {
    fn get_neighbors(&'a self, index: Idx) -> Neighbors;
}
