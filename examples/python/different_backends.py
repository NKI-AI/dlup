"""Benchmark comparing dlup slide backends across scalings and region counts.

Measures read_region and to_numpy latency for FASTSLIDE and FIMAGE backends
at multiple scaling levels with randomized region locations.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import Any

import dlup


BACKENDS = ("FASTSLIDE", "FIMAGE", "OPENSLIDE")
SCALINGS = (1.0, 0.5, 0.25)
REGION_SIZE = (512, 512)
NUM_REGIONS = 100
SEED = 42


def _random_locations(
    view_size: tuple[int, int],
    region_size: tuple[int, int],
    n: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Generate *n* random (x, y) locations that fit inside *view_size*."""
    max_x = max(0, view_size[0] - region_size[0])
    max_y = max(0, view_size[1] - region_size[1])
    return [(rng.randint(0, max_x), rng.randint(0, max_y)) for _ in range(n)]


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def benchmark_backend(
    path: Path,
    backend: str,
    scalings: tuple[float, ...],
    region_size: tuple[int, int],
    num_regions: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Run the benchmark for a single backend.

    Returns:
        Mapping of ``"scaling_{s}"`` to a dict with ``read_region_mean``,
        ``read_region_std``, ``to_numpy_mean``, ``to_numpy_std`` (all in
        seconds), plus ``view_size``.
    """
    slide = dlup.SlideImage.from_file_path(path, backend=backend)
    results: dict[str, dict[str, Any]] = {}

    for scaling in scalings:
        view = slide.get_view_at_scaling(scaling)
        view_size = view.size

        rng = random.Random(seed)
        locations = _random_locations(view_size, region_size, num_regions, rng)

        read_times: list[float] = []
        numpy_times: list[float] = []

        # Warm-up: one read to prime any internal caches / codec init.
        _ = view.read_region(locations[0], region_size).to_numpy()

        for loc in locations:
            t0 = time.perf_counter()
            region = view.read_region(loc, region_size)
            t1 = time.perf_counter()
            _ = region.to_numpy()
            t2 = time.perf_counter()

            read_times.append(t1 - t0)
            numpy_times.append(t2 - t1)

        results[f"{scaling}"] = {
            "view_size": view_size,
            "read_region_mean": statistics.mean(read_times),
            "read_region_std": statistics.stdev(read_times) if len(read_times) > 1 else 0.0,
            "to_numpy_mean": statistics.mean(numpy_times),
            "to_numpy_std": statistics.stdev(numpy_times) if len(numpy_times) > 1 else 0.0,
            "total_mean": statistics.mean(r + n for r, n in zip(read_times, numpy_times)),
            "total_std": statistics.stdev(r + n for r, n in zip(read_times, numpy_times))
            if len(read_times) > 1
            else 0.0,
        }
    slide.close()
    return results


def _print_table(all_results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Pretty-print results as a comparison table."""
    header = f"{'Backend':<12} {'Scaling':>8} {'View size':>18} {'read_region':>16} {'to_numpy':>16} {'total':>16}"
    print()
    print(header)
    print("-" * len(header))

    for backend, scaling_results in all_results.items():
        for scaling, m in scaling_results.items():
            vw, vh = m["view_size"]
            read_str = f"{_fmt_ms(m['read_region_mean'])} ± {_fmt_ms(m['read_region_std'])}"
            numpy_str = f"{_fmt_ms(m['to_numpy_mean'])} ± {_fmt_ms(m['to_numpy_std'])}"
            total_str = f"{_fmt_ms(m['total_mean'])} ± {_fmt_ms(m['total_std'])}"
            print(f"{backend:<12} {scaling:>8} {vw:>8}x{vh:<8} {read_str:>16} {numpy_str:>16} {total_str:>16}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dlup slide backends.")
    parser.add_argument("slide", type=Path, help="Path to a whole-slide image.")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKENDS),
        help=f"Backends to benchmark (default: {' '.join(BACKENDS)}).",
    )
    parser.add_argument(
        "--scalings",
        nargs="+",
        type=float,
        default=list(SCALINGS),
        help=f"Scaling factors (default: {' '.join(str(s) for s in SCALINGS)}).",
    )
    parser.add_argument(
        "--region-size",
        nargs=2,
        type=int,
        default=list(REGION_SIZE),
        metavar=("W", "H"),
        help=f"Region size in pixels (default: {REGION_SIZE[0]} {REGION_SIZE[1]}).",
    )
    parser.add_argument(
        "--num-regions",
        type=int,
        default=NUM_REGIONS,
        help=f"Number of random regions per (backend, scaling) pair (default: {NUM_REGIONS}).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"RNG seed (default: {SEED}).")
    args = parser.parse_args()

    region_size = (args.region_size[0], args.region_size[1])
    scalings = tuple(args.scalings)

    print(f"Slide:       {args.slide}")
    print(f"Backends:    {args.backends}")
    print(f"Scalings:    {scalings}")
    print(f"Region size: {region_size}")
    print(f"Regions:     {args.num_regions}")
    print(f"Seed:        {args.seed}")

    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    for backend in args.backends:
        print(f"\nBenchmarking {backend} ...")
        all_results[backend] = benchmark_backend(
            args.slide,
            backend,
            scalings,
            region_size,
            args.num_regions,
            args.seed,
        )

    _print_table(all_results)


if __name__ == "__main__":
    main()
