use crate::Neighborhood;
use ndarray::ArrayView2;

pub enum Neumann {}

impl Neighborhood for Neumann {
    type Neighbors<'a, T: 'a> = ArrayView2<'a, T>;
    type Edges<T> = [T; 8];
}
