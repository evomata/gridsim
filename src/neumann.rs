use crate::{Direction, Neighborhood};
use enum_iterator::IntoEnumIterator;
use std::iter::{once, Chain, Once};
use std::ops::{Index, IndexMut};
use NeumannDirection::*;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, IntoEnumIterator)]
pub enum NeumannDirection {
    Right,
    UpRight,
    Up,
    UpLeft,
    Left,
    DownLeft,
    Down,
    DownRight,
}

impl Direction for NeumannDirection {
    type Directions = <NeumannDirection as IntoEnumIterator>::Iterator;

    #[inline]
    fn directions() -> Self::Directions {
        NeumannDirection::into_enum_iter()
    }
}

impl NeumannDirection {
    #[inline]
    pub fn delta(self) -> (isize, isize) {
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
}

impl From<usize> for NeumannDirection {
    fn from(n: usize) -> Self {
        match n {
            0 => Right,
            1 => UpRight,
            2 => Up,
            3 => UpLeft,
            4 => Left,
            5 => DownLeft,
            6 => Down,
            7 => DownRight,
            _ => panic!("invalid integer conversion to Direction"),
        }
    }
}

impl Into<usize> for NeumannDirection {
    fn into(self) -> usize {
        match self {
            Right => 0,
            UpRight => 1,
            Up => 2,
            UpLeft => 3,
            Left => 4,
            DownLeft => 5,
            Down => 6,
            DownRight => 7,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct NeumannNeighbors<T> {
    pub right: T,
    pub up_right: T,
    pub up: T,
    pub up_left: T,
    pub left: T,
    pub down_left: T,
    pub down: T,
    pub down_right: T,
}

impl<T> NeumannNeighbors<T> {
    pub fn as_ref(&self) -> NeumannNeighbors<&T> {
        NeumannNeighbors {
            right: &self.right,
            up_right: &self.up_right,
            up: &self.up,
            up_left: &self.up_left,
            left: &self.left,
            down_left: &self.down_left,
            down: &self.down,
            down_right: &self.down_right,
        }
    }
}

impl NeumannNeighbors<bool> {
    /// Must provide an input iterator with length of the number of directions for the `Direction` impl
    /// which contains sigmoid outputs in the range of (0.0, 1.0). For each direction this will instantiate
    /// a bool that indicates whether that direction was above 0.5.
    #[inline]
    pub fn chooser(sigmoids: impl Iterator<Item = f32>) -> Self {
        sigmoids.map(|value| value > 0.5).collect()
    }

    #[inline]
    pub fn chooser_slice(sigmoids: &[f32]) -> Self {
        Self::chooser(sigmoids.iter().cloned())
    }
}

impl<T> std::iter::FromIterator<T> for NeumannNeighbors<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        Self {
            right: iter.next().unwrap(),
            up_right: iter.next().unwrap(),
            up: iter.next().unwrap(),
            up_left: iter.next().unwrap(),
            left: iter.next().unwrap(),
            down_left: iter.next().unwrap(),
            down: iter.next().unwrap(),
            down_right: iter.next().unwrap(),
        }
    }
}

impl<T> Index<NeumannDirection> for NeumannNeighbors<T> {
    type Output = T;
    #[inline]
    fn index(&self, ix: NeumannDirection) -> &T {
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

impl<T> IndexMut<NeumannDirection> for NeumannNeighbors<T> {
    #[inline]
    fn index_mut(&mut self, ix: NeumannDirection) -> &mut T {
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

impl<T> Neighborhood<T> for NeumannNeighbors<T> {
    type Direction = NeumannDirection;
    type Iter = NeighborhoodIter<T>;
    type DirIter = NeighborhoodIter<(NeumannDirection, T)>;

    #[inline]
    fn new<F: FnMut(NeumannDirection) -> T>(mut f: F) -> NeumannNeighbors<T> {
        NeumannNeighbors {
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

impl<'a, T> From<NeumannNeighbors<&'a T>> for NeumannNeighbors<T>
where
    T: Clone,
{
    #[inline]
    fn from(f: NeumannNeighbors<&'a T>) -> Self {
        NeumannNeighbors::new(|dir| f[dir].clone())
    }
}
