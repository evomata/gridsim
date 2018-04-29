use std::iter::{once, Chain, Once};
use std::ops::{Index, IndexMut};
use {Sim, SquareGrid, TakeMoveDirection, TakeMoveNeighbors};

use std::mem::transmute_copy;

use {GetNeighbors, Neighborhood};

#[derive(Copy, Clone, Debug, PartialEq, Eq, EnumIterator)]
pub enum Direction {
    Right,
    Up,
    Left,
    Down,
}

impl super::Direction for Direction {
    type Directions = DirectionEnumIterator;

    fn inv(self) -> Direction {
        use self::Direction::*;
        match self {
            Right => Left,
            Up => Down,
            Left => Right,
            Down => Up,
        }
    }

    fn directions() -> Self::Directions {
        Direction::iter_variants()
    }
}

impl Direction {
    fn delta(self) -> (isize, isize) {
        use self::Direction::*;
        match self {
            Right => (1, 0),
            Up => (0, -1),
            Left => (-1, 0),
            Down => (0, 1),
        }
    }

    pub fn left(self) -> Self {
        use self::Direction::*;
        match self {
            Right => Up,
            Up => Left,
            Left => Down,
            Down => Right,
        }
    }

    pub fn right(self) -> Self {
        use self::Direction::*;
        match self {
            Right => Down,
            Up => Right,
            Left => Up,
            Down => Left,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Neighbors<T> {
    pub right: T,
    pub up: T,
    pub left: T,
    pub down: T,
}

impl<T> Index<Direction> for Neighbors<T> {
    type Output = T;
    fn index(&self, ix: Direction) -> &T {
        use self::Direction::*;
        match ix {
            Right => &self.right,
            Up => &self.up,
            Left => &self.left,
            Down => &self.down,
        }
    }
}

impl<T> IndexMut<Direction> for Neighbors<T> {
    fn index_mut(&mut self, ix: Direction) -> &mut T {
        use self::Direction::*;
        match ix {
            Right => &mut self.right,
            Up => &mut self.up,
            Left => &mut self.left,
            Down => &mut self.down,
        }
    }
}

type NeighborhoodIter<T> = Chain<Chain<Chain<Once<T>, Once<T>>, Once<T>>, Once<T>>;

impl<T> Neighborhood<T> for Neighbors<T> {
    type Direction = Direction;
    type Iter = NeighborhoodIter<T>;
    type DirIter = NeighborhoodIter<(Direction, T)>;

    fn new<F: FnMut(Direction) -> T>(mut f: F) -> Neighbors<T> {
        use self::Direction::*;
        Neighbors {
            right: f(Right),
            up: f(Up),
            left: f(Left),
            down: f(Down),
        }
    }

    fn iter(self) -> Self::Iter {
        once(self.right)
            .chain(once(self.up))
            .chain(once(self.left))
            .chain(once(self.down))
    }

    fn dir_iter(self) -> Self::DirIter {
        use self::Direction::*;
        once((Right, self.right))
            .chain(once((Up, self.up)))
            .chain(once((Left, self.left)))
            .chain(once((Down, self.down)))
    }
}

impl<'a, T> From<Neighbors<&'a T>> for Neighbors<T>
where
    T: Clone,
{
    fn from(f: Neighbors<&'a T>) -> Self {
        Neighbors::new(|dir| f[dir].clone())
    }
}

impl<'a, C, S> GetNeighbors<'a, usize, Neighbors<&'a C>> for SquareGrid<S>
where
    S: Sim<Cell = C>,
{
    fn get_neighbors(&'a self, ix: usize) -> Neighbors<&'a C> {
        Neighbors::new(|dir| self.get_cell(self.delta_index(ix, dir.delta())))
    }
}

impl<'a, S> GetNeighbors<'static, usize, Neighbors<bool>> for SquareGrid<S>
where
    S: Sim<Cell = bool>,
{
    fn get_neighbors(&'a self, ix: usize) -> Neighbors<bool> {
        Neighbors::new(|dir| *self.get_cell(self.delta_index(ix, dir.delta())))
    }
}

impl<S, M> TakeMoveDirection<usize, Direction, M> for SquareGrid<S>
where
    S: Sim<Move = M, MoveNeighbors = Neighbors<M>>,
{
    unsafe fn take_move_direction(&self, ix: usize, dir: Direction) -> M {
        transmute_copy(&self.get_move_neighbors(ix)[dir])
    }
}

impl<S, M> TakeMoveNeighbors<usize, Neighbors<M>> for SquareGrid<S>
where
    S: Sim<Move = M, MoveNeighbors = Neighbors<M>>,
{
    unsafe fn take_move_neighbors(&self, ix: usize) -> Neighbors<M> {
        use Direction;
        Neighbors::new(|dir| self.take_move_direction(self.delta_index(ix, dir.delta()), dir.inv()))
    }
}
