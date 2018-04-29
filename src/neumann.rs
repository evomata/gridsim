use std::iter::{once, Chain, Once};
use std::ops::{Index, IndexMut};
use {Sim, SquareGrid, TakeMoveDirection, TakeMoveNeighbors};

use std::mem::transmute_copy;

use {GetNeighbors, Neighborhood};

#[derive(Copy, Clone, Debug, PartialEq, Eq, EnumIterator)]
pub enum Direction {
    Right,
    UpRight,
    Up,
    UpLeft,
    Left,
    DownLeft,
    Down,
    DownRight,
}

impl super::Direction for Direction {
    type Directions = DirectionEnumIterator;

    #[inline]
    fn inv(self) -> Direction {
        use self::Direction::*;
        match self {
            Right => Left,
            UpRight => DownLeft,
            Up => Down,
            UpLeft => DownRight,
            Left => Right,
            DownLeft => UpRight,
            Down => Up,
            DownRight => UpLeft,
        }
    }

    #[inline]
    fn directions() -> Self::Directions {
        Direction::iter_variants()
    }
}

impl Direction {
    #[inline]
    fn delta(self) -> (isize, isize) {
        use self::Direction::*;
        match self {
            Right => (1, 0),
            UpRight => (1, -1),
            Up => (0, -1),
            UpLeft => (-1, -1),
            Left => (-1, 0),
            DownLeft => (-1, 1),
            Down => (0, 1),
            DownRight => (1, 1),
        }
    }

    #[inline]
    pub fn left(self) -> Self {
        use self::Direction::*;
        match self {
            Right => UpRight,
            UpRight => Up,
            Up => UpLeft,
            UpLeft => Left,
            Left => DownLeft,
            DownLeft => Down,
            Down => DownRight,
            DownRight => Right,
        }
    }

    #[inline]
    pub fn right(self) -> Self {
        use self::Direction::*;
        match self {
            Right => DownRight,
            UpRight => Right,
            Up => UpRight,
            UpLeft => Up,
            Left => UpLeft,
            DownLeft => Left,
            Down => DownLeft,
            DownRight => Down,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Neighbors<T> {
    pub right: T,
    pub up_right: T,
    pub up: T,
    pub up_left: T,
    pub left: T,
    pub down_left: T,
    pub down: T,
    pub down_right: T,
}

impl<T> Index<Direction> for Neighbors<T> {
    type Output = T;
    #[inline]
    fn index(&self, ix: Direction) -> &T {
        use self::Direction::*;
        match ix {
            Right => &self.right,
            UpRight => &self.up_right,
            Up => &self.up,
            UpLeft => &self.up_left,
            Left => &self.left,
            DownLeft => &self.down_left,
            Down => &self.down,
            DownRight => &self.down_right,
        }
    }
}

impl<T> IndexMut<Direction> for Neighbors<T> {
    #[inline]
    fn index_mut(&mut self, ix: Direction) -> &mut T {
        use self::Direction::*;
        match ix {
            Right => &mut self.right,
            UpRight => &mut self.up_right,
            Up => &mut self.up,
            UpLeft => &mut self.up_left,
            Left => &mut self.left,
            DownLeft => &mut self.down_left,
            Down => &mut self.down,
            DownRight => &mut self.down_right,
        }
    }
}

type NeighborhoodIter<T> = Chain<
    Chain<
        Chain<Chain<Chain<Chain<Chain<Once<T>, Once<T>>, Once<T>>, Once<T>>, Once<T>>, Once<T>>,
        Once<T>,
    >,
    Once<T>,
>;

impl<T> Neighborhood<T> for Neighbors<T> {
    type Direction = Direction;
    type Iter = NeighborhoodIter<T>;
    type DirIter = NeighborhoodIter<(Direction, T)>;

    #[inline]
    fn new<F: FnMut(Direction) -> T>(mut f: F) -> Neighbors<T> {
        use self::Direction::*;
        Neighbors {
            right: f(Right),
            up_right: f(UpRight),
            up: f(Up),
            up_left: f(UpLeft),
            left: f(Left),
            down_left: f(DownLeft),
            down: f(Down),
            down_right: f(DownRight),
        }
    }

    #[inline]
    fn iter(self) -> Self::Iter {
        once(self.right)
            .chain(once(self.up_right))
            .chain(once(self.up))
            .chain(once(self.up_left))
            .chain(once(self.left))
            .chain(once(self.down_left))
            .chain(once(self.down))
            .chain(once(self.down_right))
    }

    #[inline]
    fn dir_iter(self) -> Self::DirIter {
        use self::Direction::*;
        once((Right, self.right))
            .chain(once((UpRight, self.up_right)))
            .chain(once((Up, self.up)))
            .chain(once((UpLeft, self.up_left)))
            .chain(once((Left, self.left)))
            .chain(once((DownLeft, self.down_left)))
            .chain(once((Down, self.down)))
            .chain(once((DownRight, self.down_right)))
    }
}

impl<'a, T> From<Neighbors<&'a T>> for Neighbors<T>
where
    T: Clone,
{
    #[inline]
    fn from(f: Neighbors<&'a T>) -> Self {
        Neighbors::new(|dir| f[dir].clone())
    }
}

impl<'a, C, S> GetNeighbors<'a, usize, Neighbors<&'a C>> for SquareGrid<'a, S>
where
    S: Sim<'a, Cell = C>,
{
    #[inline]
    fn get_neighbors(&'a self, ix: usize) -> Neighbors<&'a C> {
        Neighbors::new(|dir| unsafe { self.get_cell_unchecked(self.delta_index(ix, dir.delta())) })
    }
}

impl<'a, S, M> TakeMoveDirection<usize, Direction, M> for SquareGrid<'a, S>
where
    S: Sim<'a, Move = M, MoveNeighbors = Neighbors<M>>,
{
    #[inline]
    unsafe fn take_move_direction(&self, ix: usize, dir: Direction) -> M {
        transmute_copy(&self.get_move_neighbors(ix)[dir])
    }
}

impl<'a, S, M> TakeMoveNeighbors<usize, Neighbors<M>> for SquareGrid<'a, S>
where
    S: Sim<'a, Move = M, MoveNeighbors = Neighbors<M>>,
{
    #[inline]
    unsafe fn take_move_neighbors(&self, ix: usize) -> Neighbors<M> {
        use Direction;
        Neighbors::new(|dir| self.take_move_direction(self.delta_index(ix, dir.delta()), dir.inv()))
    }
}
