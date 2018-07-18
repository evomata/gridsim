use {GetNeighbors, Sim, TakeDiff, TakeMoveNeighbors};

use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use std::mem::{swap, transmute_copy, ManuallyDrop};

/// Represents the state of the simulation.
#[derive(Clone, Debug)]
pub struct SquareGrid<'a, S: Sim<'a>> {
    cells: Vec<S::Cell>,
    asyncs: Vec<S::Async>,
    diffs: Vec<ManuallyDrop<(S::Diff, S::MoveNeighbors)>>,
    width: usize,
    height: usize,
}

impl<'a, S: Sim<'a>> TakeMoveNeighbors<usize, ()> for SquareGrid<'a, S> {
    #[inline]
    unsafe fn take_move_neighbors(&self, _: usize) {}
}

impl<'a, S, D> TakeDiff<usize, D> for SquareGrid<'a, S>
where
    S: Sim<'a, Diff = D>,
{
    #[inline]
    unsafe fn take_diff(&self, ix: usize) -> D {
        transmute_copy(self.get_diff(ix))
    }
}

impl<'a, S: Sim<'a>> SquareGrid<'a, S> {
    /// Make a new grid using the Cell's Default impl.
    pub fn new(width: usize, height: usize) -> Self
    where
        S::Cell: Default,
    {
        SquareGrid {
            cells: (0..)
                .take(width * height)
                .map(|_| S::Cell::default())
                .collect(),
            asyncs: Vec::new(),
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
        SquareGrid {
            cells: ::std::iter::repeat(default).take(width * height).collect(),
            asyncs: Vec::new(),
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
        SquareGrid {
            cells,
            asyncs: Vec::new(),
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

impl<'a, S, C, D, M, N, MN> SquareGrid<'a, S>
where
    S: Sim<'a, Cell = C, Diff = D, Move = M, Neighbors = N, MoveNeighbors = MN> + 'a,
    S::Cell: Sync + Send,
    S::Async: Sync + Send,
    S::Diff: Sync + Send,
    S::Move: Sync + Send,
    S::Neighbors: Sync + Send,
    S::MoveNeighbors: Sync + Send,
    Self: GetNeighbors<'a, usize, N>,
    Self: TakeMoveNeighbors<usize, MN>,
{
    /// Run the Grid for one cycle and parallelize the simulation.
    pub fn cycle(&mut self) {
        self.process();
        self.step();
        self.update();
    }

    fn process(&mut self) {
        self.asyncs = self.cells[..]
            .par_iter()
            .enumerate()
            .map(|(ix, c)| self.single_process(ix, c))
            .collect();
    }

    fn step(&mut self) {
        let mut asyncs = Vec::new();
        swap(&mut asyncs, &mut self.asyncs);
        self.diffs = asyncs
            .into_par_iter()
            .map(S::step)
            .map(ManuallyDrop::new)
            .collect();
    }

    #[inline]
    fn single_process(&self, ix: usize, c: &C) -> S::Async {
        // TODO: Convey to the compiler this is okay without unsafe.
        let grid = unsafe { &*(self as *const Self) };
        S::process(c, grid.get_neighbors(ix))
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
