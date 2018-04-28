pub trait Direction {
    fn inv(self) -> Self;
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
