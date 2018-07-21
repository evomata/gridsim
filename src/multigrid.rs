extern crate bincode;
extern crate itertools;
extern crate serde;

use {GetNeighbors, Sim, SquareGrid, TakeMoveNeighbors};

use self::bincode::{deserialize_from, serialize_into};
use self::itertools::Itertools;
use self::serde::{Deserialize, Serialize};
use std::io::{Read, Write};

impl<'a, S, C, D, M, N, MN> SquareGrid<'a, S>
where
    S: Sim<'a, Cell = C, Diff = D, Move = M, Neighbors = N, MoveNeighbors = MN> + 'a,
    for<'dc> S::Cell: Sync + Send + Serialize + Deserialize<'dc> + 'a,
    S::Diff: Sync + Send,
    S::Move: Sync + Send,
    S::Neighbors: Sync + Send,
    S::MoveNeighbors: Sync + Send,
    Self: GetNeighbors<'a, usize, N>,
    Self: TakeMoveNeighbors<usize, MN>,
{
    /// Run the Grid for one cycle and parallelize the simulation.
    ///
    /// Make sure the reads and writes are only connected to other `SquareGrid::cycle` running
    /// on any machine using THE EXACT SAME simulation or else there may be undefined behavior.
    pub unsafe fn cycle_multi<
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
        out_right: O0,
        out_up_right: O1,
        out_up: O2,
        out_up_left: O3,
        out_left: O4,
        out_down_left: O5,
        out_down: O6,
        out_down_right: O7,
    ) -> bincode::Result<()> {
        let right = self.right(2);
        let top_right = self.top_right(2);
        let top = self.top(2);
        let top_left = self.top_left(2);
        let left = self.left(2);
        let bottom_left = self.bottom_left(2);
        let bottom = self.bottom(2);
        let bottom_right = self.bottom_right(2);

        self.serialize(out_right, right.clone())?;
        self.serialize(out_up_right, top_right.clone())?;
        self.serialize(out_up, top.clone())?;
        self.serialize(out_up_left, top_left.clone())?;
        self.serialize(out_left, left.clone())?;
        self.serialize(out_down_left, bottom_left.clone())?;
        self.serialize(out_down, bottom.clone())?;
        self.serialize(out_down_right, bottom_right.clone())?;

        self.deserialize(in_right, right)?;
        self.deserialize(in_up_right, top_right)?;
        self.deserialize(in_up, top)?;
        self.deserialize(in_up_left, top_left)?;
        self.deserialize(in_left, left)?;
        self.deserialize(in_down_left, bottom_left)?;
        self.deserialize(in_down, bottom)?;
        self.deserialize(in_down_right, bottom_right)?;

        Ok(())
    }

    fn serialize(
        &self,
        mut out: impl Write,
        it: impl Iterator<Item = usize>,
    ) -> bincode::Result<()> {
        for i in it {
            serialize_into(&mut out, &self.cells[i])?;
        }
        Ok(())
    }

    fn deserialize(
        &mut self,
        mut input: impl Read,
        it: impl Iterator<Item = usize>,
    ) -> bincode::Result<()> {
        for i in it {
            self.cells[i] = deserialize_from(&mut input)?;
        }
        Ok(())
    }

    fn right(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (width - thickness..width)
            .cartesian_product(thickness..height - thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn top_right(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (width - thickness..width)
            .cartesian_product(0..thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn top(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (thickness..width - thickness)
            .cartesian_product(0..thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn top_left(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let height = self.get_height();
        (0..thickness)
            .cartesian_product(0..thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn left(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let height = self.get_height();
        (0..thickness)
            .cartesian_product(thickness..height - thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn bottom_left(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let height = self.get_height();
        (0..thickness)
            .cartesian_product(height - thickness..height)
            .map(move |(x, y)| y * height + x)
    }

    fn bottom(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (thickness..width - thickness)
            .cartesian_product(height - thickness..height)
            .map(move |(x, y)| y * height + x)
    }

    fn bottom_right(&self, thickness: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (width - thickness..width)
            .cartesian_product(height - thickness..height)
            .map(move |(x, y)| y * height + x)
    }
}
