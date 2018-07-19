extern crate bincode;
extern crate serde;

use {GetNeighbors, Sim, SquareGrid, TakeMoveNeighbors};

use self::bincode::{deserialize_from, serialize_into};
use self::serde::{Deserialize, Serialize};
use std::io::{Read, Write};

impl<'a, S, C, D, M, N, MN> SquareGrid<'a, S>
where
    S: Sim<'a, Cell = C, Diff = D, Move = M, Neighbors = N, MoveNeighbors = MN> + 'a,
    for<'dc> S::Cell: Sync + Send + Serialize + Deserialize<'dc>,
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
        mut in_right: I0,
        in_up_right: I1,
        mut in_up: I2,
        in_up_left: I3,
        mut in_left: I4,
        in_down_left: I5,
        mut in_down: I6,
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
        let size = self.size();
        let width = self.get_width();
        let height = self.get_height();
        // Send data first (so others can receive).

        // Start with the corners.

        serialize_into(out_up_right, &self.cells[width - 1])?;
        serialize_into(out_up_left, &self.cells[0])?;
        serialize_into(out_down_left, &self.cells[(height - 1) * width])?;
        serialize_into(out_down_right, &self.cells[size - 1])?;

        // Do the sides.

        // Right
        for c in (0..height).map(|i| &self.cells[(i + 1) * width - 1]) {
            serialize_into(&mut out_right, c)?;
        }

        // Up
        for c in &self.cells[0..width] {
            serialize_into(&mut out_up, c)?;
        }

        // Left
        for c in (0..height).map(|i| &self.cells[i * width]) {
            serialize_into(&mut out_left, c)?;
        }

        // Down
        for c in &self.cells[(self.size() - width)..self.size()] {
            serialize_into(&mut out_down, c)?;
        }

        // Now receive data.

        self.cells[width - 1] = deserialize_from(in_up_right)?;
        self.cells[0] = deserialize_from(in_up_left)?;
        self.cells[(height - 1) * width] = deserialize_from(in_down_left)?;
        self.cells[size - 1] = deserialize_from(in_down_right)?;

        // Right
        for i in 0..height {
            self.cells[(i + 1) * width - 1] = deserialize_from(&mut in_right)?;
        }

        // Up
        for c in &mut self.cells[0..width] {
            *c = deserialize_from(&mut in_up)?;
        }

        // Left
        for i in 0..height {
            self.cells[i * width] = deserialize_from(&mut in_left)?;
        }

        // Down
        for c in &mut self.cells[(size - width)..size] {
            *c = deserialize_from(&mut in_down)?;
        }

        Ok(())
    }
}
