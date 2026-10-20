"""Reproducible fitting where a library's parallelism is not.

scikit-learn's KMeans reduces per-thread partial sums across its OpenMP threads in an
order that varies from run to run, so two fits with the same ``random_state`` on the
same data can return centroids a float rounding step apart. On data with many
duplicate rows (which share one distance) that is enough to move a whole tied block
across a fixed threshold: a backtest on KDD Cup 1999 flagged up to 90 rows more or
fewer between identical runs. One OpenMP thread for the fit makes it reproducible, at
about 18% more fit time at KMeans's 200,000-row training cap.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def single_openmp_thread() -> Iterator[None]:
    """Limit OpenMP to one thread for the enclosed fit, so it gives the same result every run."""
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    with threadpool_limits(limits=1, user_api="openmp"):
        yield
