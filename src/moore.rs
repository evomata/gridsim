use crate::Direction;
use enum_iterator::IntoEnumIterator;
use std::iter::{once, Chain, Once};
use std::ops::{Index, IndexMut};
use MooreDirection::*;

use crate::Neighborhood;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, IntoEnumIterator)]
pub enum MooreDirection {
    Right,
    Up,
    Left,
    Down,
}

impl Direction for MooreDirection {
    type Directions = <MooreDirection as IntoEnumIterator>::Iterator;

    #[inline]
    fn directions() -> Self::Directions {
        MooreDirection::into_enum_iter()
    }
}

impl MooreDirection {
    #[inline]
    pub fn delta(self) -> (isize, isize) {
        match self {
            Right => (1, 0),
            Up => (0, -1),
            Left => (-1, 0),
            Down => (0, 1),
        }
    }
}

impl From<usize> for MooreDirection {
    fn from(n: usize) -> Self {
        match n {
            0 => Right,
            1 => Up,
            2 => Left,
            3 => Down,
            _ => panic!("invalid integer conversion to Direction"),
        }
    }
}

impl Into<usize> for MooreDirection {
    fn into(self) -> usize {
        use MooreDirection::*;
        match self {
            Right => 0,
            Up => 1,
            Left => 2,
            Down => 3,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct MooreNeighbors<T> {
    pub right: T,
    pub up: T,
    pub left: T,
    pub down: T,
}

impl<T> MooreNeighbors<T> {
    pub fn as_ref(&self) -> MooreNeighbors<&T> {
        MooreNeighbors {
            right: &self.right,
            up: &self.up,
            left: &self.left,
            down: &self.down,
        }
    }
}

impl MooreNeighbors<bool> {
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

impl<T> std::iter::FromIterator<T> for MooreNeighbors<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        Self {
            right: iter.next().unwrap(),
            up: iter.next().unwrap(),
            left: iter.next().unwrap(),
            down: iter.next().unwrap(),
        }
    }
}

impl<T> Index<MooreDirection> for MooreNeighbors<T> {
    type Output = T;
    #[inline]
    fn index(&self, ix: MooreDirection) -> &T {
        match ix {
            Right => &self.right,
            Up => &self.up,
            Left => &self.left,
            Down => &self.down,
        }
    }
}

impl<T> IndexMut<MooreDirection> for MooreNeighbors<T> {
    #[inline]
    fn index_mut(&mut self, ix: MooreDirection) -> &mut T {
        match ix {
            Right => &mut self.right,
            Up => &mut self.up,
            Left => &mut self.left,
            Down => &mut self.down,
        }
    }
}

type NeighborhoodIter<T> = Chain<Chain<Chain<Once<T>, Once<T>>, Once<T>>, Once<T>>;

impl<T> Neighborhood<T> for MooreNeighbors<T> {
    type Direction = MooreDirection;
    type Iter = impl Iterator<Item = T>;
    type DirIter = impl Iterator<Item = (MooreDirection, T)>;

    #[inline]
    fn new<F: FnMut(MooreDirection) -> T>(mut f: F) -> MooreNeighbors<T> {
        Self {
            right: f(Right),
            up: f(Up),
            left: f(Left),
            down: f(Down),
        }
    }

    #[inline]
    fn iter(self) -> Self::Iter {
        once(self.right)
            .chain(once(self.up))
            .chain(once(self.left))
            .chain(once(self.down))
    }

    #[inline]
    fn dir_iter(self) -> Self::DirIter {
        once((Right, self.right))
            .chain(once((Up, self.up)))
            .chain(once((Left, self.left)))
            .chain(once((Down, self.down)))
    }
}

impl<'a, T> From<MooreNeighbors<&'a T>> for MooreNeighbors<T>
where
    T: Clone,
{
    #[inline]
    fn from(f: MooreNeighbors<&'a T>) -> Self {
        Self::new(|dir| f[dir].clone())
    }
}
