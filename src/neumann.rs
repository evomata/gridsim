use std::iter::{once, Chain, Once};
use std::ops::Index;
use {Rule, Sim, SquareGrid};

use super::GetNeighbors;
use rayon::prelude::*;

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

    fn directions() -> Self::Directions {
        Direction::iter_variants()
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

impl<T> super::Neighborhood<T> for Neighbors<T> {
    type Direction = Direction;
    type Iter = Chain<
        Chain<
            Chain<Chain<Chain<Chain<Chain<Once<T>, Once<T>>, Once<T>>, Once<T>>, Once<T>>, Once<T>>,
            Once<T>,
        >,
        Once<T>,
    >;
    type DirIter = Chain<
        Chain<
            Chain<
                Chain<
                    Chain<
                        Chain<
                            Chain<Once<(Direction, T)>, Once<(Direction, T)>>,
                            Once<(Direction, T)>,
                        >,
                        Once<(Direction, T)>,
                    >,
                    Once<(Direction, T)>,
                >,
                Once<(Direction, T)>,
            >,
            Once<(Direction, T)>,
        >,
        Once<(Direction, T)>,
    >;

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

impl<'a, T> Into<Neighbors<T>> for Neighbors<&'a T>
where
    T: Clone,
{
    fn into(self) -> Neighbors<T> {
        Neighbors {
            right: self.right.clone(),
            up_right: self.up_right.clone(),
            up: self.up.clone(),
            up_left: self.up_left.clone(),
            left: self.left.clone(),
            down_left: self.down_left.clone(),
            down: self.down.clone(),
            down_right: self.down_right.clone(),
        }
    }
}

impl<T, C> Sim for T
where
    T: Rule<Cell = C, Neighbors = Neighbors<C>>,
    C: Clone,
{
    type Cell = C;
    type Diff = C;
    type Move = ();

    type Neighbors = Neighbors<C>;
    type MoveNeighbors = ();

    #[inline]
    fn step(cell: &C, neighbors: Self::Neighbors) -> (C, ()) {
        (Self::rule(cell.clone(), neighbors), Default::default())
    }

    #[inline]
    fn update(cell: &mut Self::Cell, diff: Self::Diff, _: ()) {
        *cell = diff;
    }
}

impl<'a, C, S> GetNeighbors<'a, usize, Neighbors<&'a C>> for SquareGrid<S>
where
    S: Sim<Cell = C>,
{
    fn get_neighbors(&'a self, ix: usize) -> Neighbors<&'a C> {
        Neighbors {
            up_left: self.get_cell(self.size() + ix - 1 - self.width),
            up: self.get_cell(self.size() + ix - self.width),
            up_right: self.get_cell(self.size() + ix + 1 - self.width),
            left: self.get_cell(self.size() + ix - 1),
            right: self.get_cell(self.size() + ix + 1),
            down_left: self.get_cell(self.size() + ix - 1 + self.width),
            down: self.get_cell(self.size() + ix + self.width),
            down_right: self.get_cell(self.size() + ix + 1 + self.width),
        }
    }
}
