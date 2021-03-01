use gridsim::{Neumann, Sim, SquareGrid};
use ndarray::ArrayView2;

/// Conway's Game of Life
#[derive(Debug)]
pub struct Gol;

impl Sim<Neumann> for Gol {
    type Cell = bool;
    type Diff = bool;
    type Flow = ();

    fn compute(&self, cells: ArrayView2<'_, bool>) -> bool {
        let n = cells.iter().filter(|&&c| c).count();
        if cells[(1, 1)] {
            (3..=4).contains(&n)
        } else {
            n == 3
        }
    }

    fn egress(&self, cell: &mut Self::Cell, diffs: ArrayView2<'_, bool>) -> [(); 8] {
        *cell = diffs[(1, 1)];
        [(); 8]
    }

    fn ingress(&self, _: &mut Self::Cell, _: [(); 8]) {}

    fn cell_padding(&self) -> Self::Cell {
        false
    }

    fn diff_padding(&self) -> Self::Diff {
        false
    }

    fn flow_padding(&self) -> Self::Flow {}
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
        grid.step_parallel();
        assert_eq!(
            grid.cells(),
            Array2::from_shape_fn((5, 5), |(y, x)| x == 2 && y >= 1 && y <= 3)
        );
        grid.step_parallel();
        assert_eq!(
            grid.cells(),
            Array2::from_shape_fn((5, 5), |(y, x)| y == 2 && x >= 1 && x <= 3)
        );
    }
}
