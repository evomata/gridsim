use std::iter::{once, Chain, Once};
use std::ops::Index;

#[derive(Copy, Clone, Debug, PartialEq, Eq, EnumIterator)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn inv(self) -> Direction {
        use self::Direction::*;
        match self {
            Up => Down,
            Down => Up,
            Left => Right,
            Right => Left,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Neighborhood<T> {
    pub up: T,
    pub down: T,
    pub left: T,
    pub right: T,
}

impl<'a, T> Index<Direction> for Neighborhood<T> {
    type Output = T;
    fn index(&self, ix: Direction) -> &T {
        use self::Direction::*;
        match ix {
            Up => &self.up,
            Down => &self.down,
            Left => &self.left,
            Right => &self.right,
        }
    }
}
