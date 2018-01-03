//! Gridsim is a library intended to run grid-based simulations.
//! The library in its current form is not what it is intended to be.
//! Once [const generics](https://github.com/rust-lang/rust/issues/44580) are available in nightly, this library will
//! be turned into a library which is generic across all grid simulations. This will include hex grid,
//! rhombic dodecahedral honeycomb (in its multiple tight-pack layer patterns), square grid, cube grid,
//! and even n-dimensional grids. It will also be generic over the neighbor distance including
//! moore and von-neumann neighborhood. It should also eventually be given mechanisms to easier support
//! running on clusters.
//!
//! In its current early state, it will be used for 2d square grids only. The structure will be relatively
//! similar to the final form, but include none of the above features except for the simulation part.

/// Defines a simulation for simple things like cellular automata.
pub trait Rule {
    /// The type of cells on the grid
    type Cell;

    /// This defines a rule for how cells in a 3x3 space transform into a new Cell in the center position
    /// of the new board.
    fn rule([[Self::Cell; 3]; 3]) -> Self::Cell;
}

/// Defines a simulation for complicated things that have too much state to abandon on the next cycle.
///
/// This enforces a rule in that all new cells are only produced from old board state. This prevents the
/// update order from breaking the simulation.
pub trait Sim {
    /// The type of cells on the grid
    type Cell;
    /// Represents all information necessary to modify a cell in the previous grid to produce the version in the next.
    type Diff;

    /// Performs one step of the simulation by producing a grid of diffs that can be used to change the cells to
    /// their next state.
    fn step([[&Self::Cell; 3]; 3]) -> Self::Diff;
    /// Updates a cell with a diff.
    fn update(&mut Self::Cell, Self::Diff);
}

impl<T> Sim for T
where
    T: Rule,
    T::Cell: Clone,
{
    type Cell = T::Cell;
    type Diff = T::Cell;

    #[inline]
    fn step(old: [[&Self::Cell; 3]; 3]) -> Self::Diff {
        Self::rule([
            [old[0][0].clone(), old[0][1].clone(), old[0][2].clone()],
            [old[1][0].clone(), old[1][1].clone(), old[1][2].clone()],
            [old[2][0].clone(), old[2][1].clone(), old[2][2].clone()],
        ])
    }

    #[inline]
    fn update(cell: &mut Self::Cell, diff: Self::Diff) {
        *cell = diff;
    }
}

/// Represents the state of the simulation.
///
/// This is not as efficient for Rule and is optimized for Sim.
pub struct Grid<S: Sim> {
    cells: Vec<S::Cell>,
    diffs: Vec<S::Diff>,
    width: usize,
    height: usize,
}

impl<S: Sim> Grid<S> {
    /// Make a new grid using the Cell's Default impl.
    pub fn new(width: usize, height: usize) -> Grid<S>
    where
        S::Cell: Default,
    {
        Grid {
            cells: (0..)
                .take(width * height)
                .map(|_| S::Cell::default())
                .collect(),
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a new grid by cloning a default Cell.
    pub fn new_default(width: usize, height: usize, default: S::Cell) -> Grid<S>
    where
        S::Cell: Clone,
    {
        Grid {
            cells: ::std::iter::repeat(default).take(width * height).collect(),
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Make a new grid directly from an initial iter.
    pub fn new_iter<I>(width: usize, height: usize, iter: I) -> Grid<S>
    where
        I: IntoIterator<Item = S::Cell>,
    {
        let cells: Vec<_> = iter.into_iter().take(width * height).collect();
        // Assert that they provided enough cells. If they didn't the simulation would panic.
        assert_eq!(
            cells.len(),
            width * height,
            "gridsim::Grid::new_iter: not enough cells provided in iter"
        );
        Grid {
            cells: cells,
            diffs: Vec::new(),
            width: width,
            height: height,
        }
    }

    /// Run the Grid for one cycle.
    pub fn cycle(&mut self) {
        self.step();
        self.update();
    }

    fn step(&mut self) {
        self.diffs = {
            let cs = |i| &self.cells[i % self.size()];
            (0..self.size())
                .map(|i| {
                    [
                        [
                            cs(self.size() + i - 1 - self.width),
                            cs(self.size() + i - self.width),
                            cs(self.size() + i + 1 - self.width),
                        ],
                        [
                            cs(self.size() + i - 1),
                            cs(self.size() + i),
                            cs(self.size() + i + 1),
                        ],
                        [
                            cs(self.size() + i - 1 + self.width),
                            cs(self.size() + i + self.width),
                            cs(self.size() + i + 1 + self.width),
                        ],
                    ]
                })
                .map(S::step)
                .collect()
        };
    }

    fn update(&mut self) {
        for (cell, diff) in self.cells.iter_mut().zip(self.diffs.drain(..)) {
            S::update(cell, diff);
        }
    }

    /// Get the Grid's Cell slice.
    #[inline]
    pub fn get_cells(&self) -> &[S::Cell] {
        &self.cells[..]
    }

    /// Get the Grid's Cell slice mutably.
    #[inline]
    pub fn get_cells_mut(&mut self) -> &mut [S::Cell] {
        &mut self.cells[..]
    }

    /// Get the Grid's size.
    #[inline]
    pub fn size(&self) -> usize {
        self.width * self.height
    }
}

/// Conway's Game of Life
pub enum GOL {}

impl Rule for GOL {
    type Cell = bool;

    fn rule(cells: [[bool; 3]; 3]) -> bool {
        let n = cells
            .iter()
            .flat_map(|cs| cs.iter())
            .filter(|&&c| c)
            .count();
        if cells[1][1] {
            n >= 3 && n <= 4
        } else {
            n == 3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gol_blinker() {
        let mut grid = Grid::<GOL>::new_iter(
            5,
            5,
            vec![
                false, false, false, false, false, false, false, true, false, false, false, false,
                true, false, false, false, false, true, false, false, false, false, false, false,
                false,
            ],
        );

        grid.cycle();

        assert_eq!(
            grid.get_cells(),
            &vec![
                false, false, false, false, false, false, false, false, false, false, false, true,
                true, true, false, false, false, false, false, false, false, false, false, false,
                false,
            ][..]
        )
    }
}
