extern crate bincode;
extern crate itertools;
extern crate serde;

use crate::{GetNeighbors, Sim, SquareGrid, TakeMoveNeighbors};

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
        let outside_right = self.right(2, 0);
        let outside_top_right = self.top_right(2, 0);
        let outside_top = self.top(2, 0);
        let outside_top_left = self.top_left(2, 0);
        let outside_left = self.left(2, 0);
        let outside_bottom_left = self.bottom_left(2, 0);
        let outside_bottom = self.bottom(2, 0);
        let outside_bottom_right = self.bottom_right(2, 0);

        let inside_right = self.right(2, 2);
        let inside_top_right = self.top_right(2, 2);
        let inside_top = self.top(2, 2);
        let inside_top_left = self.top_left(2, 2);
        let inside_left = self.left(2, 2);
        let inside_bottom_left = self.bottom_left(2, 2);
        let inside_bottom = self.bottom(2, 2);
        let inside_bottom_right = self.bottom_right(2, 2);

        self.serialize(out_right, inside_right.clone())?;
        self.serialize(out_up_right, inside_top_right.clone())?;
        self.serialize(out_up, inside_top.clone())?;
        self.serialize(out_up_left, inside_top_left.clone())?;
        self.serialize(out_left, inside_left.clone())?;
        self.serialize(out_down_left, inside_bottom_left.clone())?;
        self.serialize(out_down, inside_bottom.clone())?;
        self.serialize(out_down_right, inside_bottom_right.clone())?;

        self.deserialize(in_right, outside_right)?;
        self.deserialize(in_up_right, outside_top_right)?;
        self.deserialize(in_up, outside_top)?;
        self.deserialize(in_up_left, outside_top_left)?;
        self.deserialize(in_left, outside_left)?;
        self.deserialize(in_down_left, outside_bottom_left)?;
        self.deserialize(in_down, outside_bottom)?;
        self.deserialize(in_down_right, outside_bottom_right)?;

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
        out.flush()?;
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

    fn right(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (width - thickness - skip..width - skip)
            .cartesian_product(thickness..height - thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn top_right(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (width - thickness - skip..width - skip)
            .cartesian_product(skip..thickness + skip)
            .map(move |(x, y)| y * height + x)
    }

    fn top(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (thickness..width - thickness)
            .cartesian_product(skip..skip + thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn top_left(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let height = self.get_height();
        (skip..skip + thickness)
            .cartesian_product(skip..skip + thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn left(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let height = self.get_height();
        (skip..skip + thickness)
            .cartesian_product(thickness..height - thickness)
            .map(move |(x, y)| y * height + x)
    }

    fn bottom_left(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let height = self.get_height();
        (skip..skip + thickness)
            .cartesian_product(height - thickness - skip..height - skip)
            .map(move |(x, y)| y * height + x)
    }

    fn bottom(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (thickness..width - thickness)
            .cartesian_product(height - thickness - skip..height - skip)
            .map(move |(x, y)| y * height + x)
    }

    fn bottom_right(&self, thickness: usize, skip: usize) -> impl Iterator<Item = usize> + Clone {
        let width = self.get_width();
        let height = self.get_height();
        (width - thickness - skip..width - skip)
            .cartesian_product(height - thickness - skip..height - skip)
            .map(move |(x, y)| y * height + x)
    }
}
