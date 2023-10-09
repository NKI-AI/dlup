# coding=utf-8
# Copyright (c) dlup contributors
import numpy as np
import pytest

from dlup.tiling import Grid, TilingMode, indexed_ndmesh, tiles_grid_coordinates


class TestTiling:
    @pytest.mark.parametrize("mode", [TilingMode.skip])
    def test_all_zero(self, mode):
        """If all the arguments are zero, an exception should be raised."""
        with pytest.raises(ValueError):
            (basis,) = tiles_grid_coordinates(0, 0, 0, mode=mode)

    @pytest.mark.parametrize("mode", list(TilingMode))
    @pytest.mark.parametrize("tile_overlap", [0, 1, 2])
    def test_tile_bigger_than_size(self, mode, tile_overlap):
        """Check different modes if tile_size is bigger than the size."""
        size = 2
        tile_size = 10
        (basis,) = tiles_grid_coordinates(size, tile_size, tile_overlap=tile_overlap, mode=mode)

        expected_lengths = {TilingMode.skip: 0, TilingMode.overflow: 1}

        assert (basis >= 0).all()
        assert len(basis) == expected_lengths[mode]

    @pytest.mark.parametrize(
        "size, tile_size, tile_overlap",
        [(10, 3, 0), (3, 1, 2), (17, 3.2, 2), (53.2, 12.2, 15), (1, 2, 3)],
    )
    @pytest.mark.parametrize("mode", list(TilingMode))
    def test_spanned_basis(self, size, tile_size, tile_overlap, mode):
        """Check the spanned basis behaves as configured for tiles."""
        (basis,) = tiles_grid_coordinates(size, tile_size, tile_overlap=tile_overlap, mode=mode)

        # Is sorted
        assert np.all(np.diff(basis) >= 0)

        if len(basis) == 0:
            return

        # First coordinate is always zero.
        assert basis[0] == 0

        # TODO: These are not used yet
        # tile_overlap = np.remainder(tile_overlap, np.minimum(tile_size, size), casting="safe")
        # right = basis + tile_size
        # overlap = right - basis
        stride = np.diff(basis)
        tiled_size = basis[-1] + tile_size

        # Grid is uniform
        if len(stride):
            assert np.isclose(stride, stride[0]).all()

        if np.isclose(tiled_size, size):
            return

        if mode == TilingMode.skip:
            assert tiled_size < size

        if mode == TilingMode.overflow:
            assert tiled_size > size

    def test_spanned_basis_multiple_dims(self):
        """Check that multiple dims is the same as a single dim."""
        (basis,) = tiles_grid_coordinates(10, 3, 1.2)
        dbasis, _ = tiles_grid_coordinates((10, 5), (3, 2), (1.2, 1))
        assert (basis == dbasis).all()

    def test_indexed_ndmesh(self):
        """Check ndmesh example access."""
        mesh = indexed_ndmesh(((1, 2, 3), (4, 5, 6)))
        assert (mesh[0, 0] == (1, 4)).all()
        assert (mesh[0, 1] == (1, 5)).all()
        assert (mesh[1, 0] == (2, 4)).all()

        mesh = indexed_ndmesh(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        assert (mesh[0, 0, 0] == (1, 4, 7)).all()
        assert (mesh[0, 1, 0] == (1, 5, 7)).all()
        assert (mesh[2, 1, 1] == (3, 5, 8)).all()

    def test_grid(self):
        """Test Grid basic api."""
        grid = Grid([np.array((0, 1)), np.array((2, 3, 4))], order="F")

        assert grid.size == (2, 3)
        assert len(grid) == 6

        # First row, first column
        assert (grid[0] == (0, 2)).all()
        # First row, second column
        assert (grid[1] == (0, 3)).all()
