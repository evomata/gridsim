pub trait Direction: Copy + From<usize> + Into<usize> {
    type Directions: Iterator<Item = Self>;

    /// An iterator over all directions. There must be an even number of directions and they
    /// must rotate counter-clockwise when iterating in order for all default impls to work.
    /// If your direction does not follow these rules then you must override the default impls.
    fn directions() -> Self::Directions;

    /// All directions should have an opposite direction on the opposite side of its iterator.
    #[inline]
    fn inv(self) -> Self {
        let mut n: usize = self.into();
        n += Self::total() / 2;
        n %= Self::total();
        n.into()
    }

    /// The total number of directions.
    #[inline]
    fn total() -> usize {
        Self::directions().count()
    }

    #[inline]
    fn turn_clockwise(self) -> Self {
        let mut n: usize = self.into();
        n += Self::total() - 1;
        n %= Self::total();
        n.into()
    }

    #[inline]
    fn turn_counterclockwise(self) -> Self {
        let mut n: usize = self.into();
        n += 1;
        n %= Self::total();
        n.into()
    }
}
