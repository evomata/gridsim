pub trait Direction: Sized + From<usize> + Into<usize> {
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

    /// Must provide an input iterator with length of the number of directions for the `Direction` impl
    /// which contains sigmoid outputs in the range of (0.0, 1.0). This will choose a direction based on
    /// the highest sigmoid output. If none of the values are greater than 0.5, then it will choose no direction.
    #[inline]
    fn chooser(sigmoids: impl Iterator<Item = f32>) -> Option<(Self, f32)> {
        sigmoids
            .map(float_ord::FloatOrd)
            .enumerate()
            .max_by_key(|&(_, n)| n)
            .and_then(|(ix, value)| (value.0 > 0.5).then(|| (ix.into(), value.0)))
    }

    /// Must provide an input slice with length of the number of directions for the `Direction` impl
    /// which contains sigmoid outputs in the range of (0.0, 1.0). This will choose a direction based on
    /// the highest sigmoid output. If none of the values are greater than 0.5, then it will choose no direction.
    #[inline]
    fn chooser_slice(sigmoids: &[f32]) -> Option<(Self, f32)> {
        Self::chooser(sigmoids.iter().copied())
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
