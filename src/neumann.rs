use std::ops::Index;

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

impl Direction {
    pub fn inv(self) -> Direction {
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
}

#[derive(Copy, Clone, Debug)]
pub struct Neighborhood<'a, T: 'a>([&'a T; 8]);

impl<'a, T> Index<Direction> for Neighborhood<'a, T> {
    type Output = T;
    fn index(&self, ix: Direction) -> &T {
        use self::Direction::*;
        self.0[match ix {
                   Right => 0,
                   UpRight => 1,
                   Up => 2,
                   UpLeft => 3,
                   Left => 4,
                   DownLeft => 5,
                   Down => 6,
                   DownRight => 7,
               }]
    }
}

impl<'a, T> Neighborhood<'a, T> {
    pub fn new(neighborhood: [&'a T; 8]) -> Self {
        Neighborhood(neighborhood)
    }
}
