extern crate bincode;
extern crate serde;

use {GetNeighbors, Sim, TakeDiff, TakeMoveNeighbors};

use self::bincode::{deserialize_from, serialize_into};
use self::serde::{Deserialize, Serialize};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::io::{Read, Write};
use std::mem::transmute_copy;
use std::mem::ManuallyDrop;

use neumann::*;
use Neighborhood;

/// Represents the state of the simulation.
#[derive(Clone, Debug)]
pub struct VonNeumannMultiSquareGrid<'a, S: Sim<'a>> {
    cells: Vec<S::Cell>,
    neighbor_cells: Neighbors<Vec<S::Cell>>,
    diffs: Vec<ManuallyDrop<(S::Diff, S::MoveNeighbors)>>,
    width: usize,
    height: usize,
}

impl<'a, S: Sim<'a>> TakeMoveNeighbors<usize, ()> for VonNeumannMultiSquareGrid<'a, S> {
    #[inline]
    unsafe fn take_move_neighbors(&self, _: usize) {}
}

impl<'a, S, D> TakeDiff<usize, D> for VonNeumannMultiSquareGrid<'a, S>
where
    S: Sim<'a, Diff = D>,
{
    #[inline]
    unsafe fn take_diff(&self, ix: usize) -> D {
        transmute_copy(self.get_diff(ix))
    }
}

impl<'a, S: Sim<'a>> VonNeumannMultiSquareGrid<'a, S> {
    /// Make a new grid using the Cell's Default impl.
    pub fn new(width: usize, height: usize) -> Self
    where
        S::Cell: Default,
    {
        VonNeumannMultiSquareGrid {
            cells: (0..)
                .take(width * height)
                .map(|_| S::Cell::default())
                .collect(),
            neighbor_cells: Neighbors::new(|_| Vec::new()),
            diffs: Vec::new(),
            width,
            height,
        }
    }

    /// Make a new grid by cloning a default Cell.
    pub fn new_default(width: usize, height: usize, default: S::Cell) -> Self
    where
        S::Cell: Clone,
    {
        VonNeumannMultiSquareGrid {
            cells: ::std::iter::repeat(default).take(width * height).collect(),
            neighbor_cells: Neighbors::new(|_| Vec::new()),
            diffs: Vec::new(),
            width,
            height,
        }
    }

    /// Make a new grid directly from an initial iter.
    pub fn new_iter<I>(width: usize, height: usize, iter: I) -> Self
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
        VonNeumannMultiSquareGrid {
            cells,
            neighbor_cells: Neighbors::new(|_| Vec::new()),
            diffs: Vec::new(),
            width,
            height,
        }
    }

    /// Make a grid by evaluating each centered signed coordinate to a cell with a closure.
    pub fn new_coord_map<F>(width: usize, height: usize, mut coord_map: F) -> Self
    where
        F: FnMut(isize, isize) -> S::Cell,
    {
        Self::new_iter(
            width,
            height,
            (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .map(move |(x, y)| {
                    coord_map(
                        x as isize - width as isize / 2,
                        y as isize - height as isize / 2,
                    )
                }),
        )
    }

    /// Make a grid using a collection of centered signed coordinates with associated cells.
    pub fn new_coords<I>(width: usize, height: usize, coords: I) -> Self
    where
        I: IntoIterator<Item = ((isize, isize), S::Cell)>,
        S::Cell: Default,
    {
        use std::collections::HashMap;
        let coords: &mut HashMap<(isize, isize), S::Cell> = &mut coords.into_iter().collect();
        Self::new_coord_map(width, height, |x, y| {
            coords.remove(&(x, y)).unwrap_or_default()
        })
    }

    /// Make a grid using a collection of centered signed coordinates that indicate true cells.
    pub fn new_true_coords<I>(width: usize, height: usize, coords: I) -> Self
    where
        I: IntoIterator<Item = (isize, isize)>,
        S: Sim<'a, Cell = bool>,
    {
        Self::new_coords(width, height, coords.into_iter().map(|c| (c, true)))
    }

    /// Offset an index to a new index.
    #[inline]
    pub fn delta_index(&self, i: usize, delta: (isize, isize)) -> usize {
        // Wrap width and height to be in the range (-dim, dim).
        // NOTE: This is technically wrong because if the y or x is negative enough index wont go positive.
        // Technically this will result in a panic due to usize conversion in debug mode if used incorrectly.
        // In release mode if it goes too negative it will continue working only if the size of the grid is
        // a power of two due to the automatic correct modding behavior of negative wraparound.
        let x = delta.0;
        let y = delta.1;

        ((i + self.size()) as isize + x + y * self.width as isize) as usize % self.size()
    }

    /// Get a &Cell. Panics if out of bounds.
    #[inline]
    pub fn get_cell(&self, i: usize) -> &S::Cell {
        &self.cells[i]
    }

    /// Get a &Cell. Panics if out of bounds.
    #[inline]
    pub unsafe fn get_cell_unchecked(&self, i: usize) -> &S::Cell {
        self.cells.get_unchecked(i)
    }

    /// This can only be called in the trait `TakeMoveDirection` when implmenting a new `Neighborhood`.
    #[inline]
    pub unsafe fn get_move_neighbors(&self, i: usize) -> &S::MoveNeighbors {
        &self.diffs.get_unchecked(i).1
    }

    /// This can only be called in the trait `TakeMoveDirection` when implmenting a new `Neighborhood`.
    #[inline]
    pub unsafe fn get_diff(&self, i: usize) -> &S::Diff {
        &self.diffs.get_unchecked(i).0
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

    /// Get the Grid's width.
    #[inline]
    pub fn get_width(&self) -> usize {
        self.width
    }

    /// Get the Grid's height.
    #[inline]
    pub fn get_height(&self) -> usize {
        self.height
    }

    /// Get the Grid's size.
    #[inline]
    pub fn size(&self) -> usize {
        self.width * self.height
    }
}

impl<'a, S, C, D, M, N, MN> VonNeumannMultiSquareGrid<'a, S>
where
    S: Sim<'a, Cell = C, Diff = D, Move = M, Neighbors = N, MoveNeighbors = MN> + 'a,
    S::Cell: Sync + Send + Serialize + Deserialize<'a>,
    S::Diff: Sync + Send,
    S::Move: Sync + Send + Serialize + Deserialize<'a>,
    S::Neighbors: Sync + Send,
    S::MoveNeighbors: Sync + Send,
    Self: GetNeighbors<'a, usize, N>,
    Self: TakeMoveNeighbors<usize, MN>,
{
    /// Run the Grid for one cycle and parallelize the simulation.
    ///
    /// Make sure the reads and writes are only connected to other `VonNeumannMultiSquareGrid::cycle` running
    /// on any machine using THE EXACT SAME simulation or else there may be undefined behavior.
    pub unsafe fn cycle<
        I0: Read,
        I1: Read,
        I2: Read,
        I3: Read,
        I4: Read,
        I5: Read,
        I6: Read,
        I7: Read,
        O0: Write,
        O1: Write,
        O2: Write,
        O3: Write,
        O4: Write,
        O5: Write,
        O6: Write,
        O7: Write,
    >(
        &mut self,
        mut in_right: I0,
        mut in_up_right: I1,
        mut in_up: I2,
        mut in_up_left: I3,
        mut in_left: I4,
        mut in_down_left: I5,
        mut in_down: I6,
        mut in_down_right: I7,
        mut out_right: O0,
        mut out_up_right: O1,
        mut out_up: O2,
        mut out_up_left: O3,
        mut out_left: O4,
        mut out_down_left: O5,
        mut out_down: O6,
        mut out_down_right: O7,
    ) -> bincode::Result<()> {
        self.sync_cells(
            &mut in_right,
            &mut in_up_right,
            &mut in_up,
            &mut in_up_left,
            &mut in_left,
            &mut in_down_left,
            &mut in_down,
            &mut in_down_right,
            &mut out_right,
            &mut out_up_right,
            &mut out_up,
            &mut out_up_left,
            &mut out_left,
            &mut out_down_left,
            &mut out_down,
            &mut out_down_right,
        )?;
        self.step();
        self.sync_moves(
            in_right,
            in_up_right,
            in_up,
            in_up_left,
            in_left,
            in_down_left,
            in_down,
            in_down_right,
            out_right,
            out_up_right,
            out_up,
            out_up_left,
            out_left,
            out_down_left,
            out_down,
            out_down_right,
        )?;
        self.update();
        Ok(())
    }

    /// Synchronize cells with other grids.
    fn sync_cells<
        I0: Read,
        I1: Read,
        I2: Read,
        I3: Read,
        I4: Read,
        I5: Read,
        I6: Read,
        I7: Read,
        O0: Write,
        O1: Write,
        O2: Write,
        O3: Write,
        O4: Write,
        O5: Write,
        O6: Write,
        O7: Write,
    >(
        &mut self,
        in_right: I0,
        in_up_right: I1,
        in_up: I2,
        in_up_left: I3,
        in_left: I4,
        in_down_left: I5,
        in_down: I6,
        in_down_right: I7,
        mut out_right: O0,
        out_up_right: O1,
        mut out_up: O2,
        out_up_left: O3,
        mut out_left: O4,
        out_down_left: O5,
        mut out_down: O6,
        out_down_right: O7,
    ) -> bincode::Result<()> {
        // Send data first (so others can receive).

        // Start with the corners.

        serialize_into(out_up_right, &self.cells[self.width - 1])?;
        serialize_into(out_up_left, &self.cells[0])?;
        serialize_into(out_down_left, &self.cells[(self.height - 1) * self.width])?;
        serialize_into(out_down_right, &self.cells[self.size() - 1])?;

        // Do the sides.

        // Right
        for c in (0..self.height).map(|i| &self.cells[(i + 1) * self.width - 1]) {
            serialize_into(&mut out_right, c)?;
        }

        // Up
        for c in &self.cells[0..self.width] {
            serialize_into(&mut out_up, c)?;
        }

        // Left
        for c in (0..self.height).map(|i| &self.cells[i * self.width]) {
            serialize_into(&mut out_left, c)?;
        }

        // Down
        for c in &self.cells[(self.size() - self.width)..self.size()] {
            serialize_into(&mut out_down, c)?;
        }

        // TODO: Add receive code.
        unimplemented!();

        Ok(())
    }

    /// Synchronize cells with other grids.
    fn sync_moves<
        I0: Read,
        I1: Read,
        I2: Read,
        I3: Read,
        I4: Read,
        I5: Read,
        I6: Read,
        I7: Read,
        O0: Write,
        O1: Write,
        O2: Write,
        O3: Write,
        O4: Write,
        O5: Write,
        O6: Write,
        O7: Write,
    >(
        &mut self,
        in_right: I0,
        in_up_right: I1,
        in_up: I2,
        in_up_left: I3,
        in_left: I4,
        in_down_left: I5,
        in_down: I6,
        in_down_right: I7,
        out_right: O0,
        out_up_right: O1,
        out_up: O2,
        out_up_left: O3,
        out_left: O4,
        out_down_left: O5,
        out_down: O6,
        out_down_right: O7,
    ) -> bincode::Result<()> {
        unimplemented!()
    }

    fn step(&mut self) {
        self.diffs = self.cells[..]
            .par_iter()
            .enumerate()
            .map(|(ix, c)| self.single_step(ix, c))
            .map(ManuallyDrop::new)
            .collect();
    }

    #[inline]
    fn single_step(&self, ix: usize, c: &C) -> (S::Diff, S::MoveNeighbors) {
        // TODO: Convey to the compiler this is okay without unsafe.
        let grid = unsafe { &*(self as *const Self) };
        S::step(c, grid.get_neighbors(ix))
    }

    fn update(&mut self) {
        self.cells[..]
            .par_iter()
            .enumerate()
            .for_each(|(ix, cell)| unsafe {
                S::update(
                    &mut *(cell as *const C as *mut C),
                    self.take_diff(ix),
                    (self as &Self).take_move_neighbors(ix),
                );
            });
        // This wont call any `drop()` because of the `ManuallyDrop`.
        self.diffs.clear();
    }
}
