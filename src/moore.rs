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
pub struct Neighborhood<'a, T: 'a>([&'a T; 4]);

impl<'a, T> Index<Direction> for Neighborhood<'a, T> {
    type Output = T;
    fn index(&self, ix: Direction) -> &T {
        use self::Direction::*;
        self.0[match ix {
                   Up => 0,
                   Down => 1,
                   Left => 2,
                   Right => 3,
               }]
    }
}

impl<'a, T> Neighborhood<'a, T> {
    pub fn new(neighborhood: [&'a T; 4]) -> Self {
        Neighborhood(neighborhood)
    }
}
