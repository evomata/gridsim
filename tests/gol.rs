/// Conway's Game of Life
#[derive(Debug)]
pub struct Gol;

impl Sim for Gol {
    type Cell = bool;
    type Diff = bool;
    type Flow = ();
    type Neighborhood<T> = neumann::NeumannNeighbors<T>;

    fn compute(&self, cells: Self::Neighborhood<&'_ Self::Cell>) -> Self::Diff {
        let n = cells.iter().filter(|&&c| c).count();
        if cells[(1, 1)] {
            (3..=4).contains(&n)
        } else {
            n == 3
        }
    }

    fn egress(
        &self,
        cell: &mut Self::Cell,
        diffs: Self::Neighborhood<&'_ Self::Diff>,
    ) -> Self::Neighborhood<Self::Flow> {
        neumann::NeumannNeighbors
    }

    fn ingress(&self, cell: &mut Self::Cell, flows: Self::Neighborhood<Self::Flow>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn gol_blinker() {
        let mut grid = SquareGrid::new(
            Gol,
            Array2::from_shape_fn((5, 5), |(y, x)| y == 2 && x >= 1 && x <= 3),
        );

        // grid.cycle();

        // assert_eq!(
        //     grid.cells(),
        //     SquareGrid::<Gol>::new_true_coords(5, 5, (-1..2).map(|n| (n, 0))).cells()
        // )
    }
}
