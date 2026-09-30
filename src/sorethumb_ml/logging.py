"""Logger setup helpers.

Every module obtains its logger as ``logging.getLogger(__name__)``, giving the
whole library one ``sorethumb_ml.*`` namespace a caller can configure in one
line::

    logging.getLogger("sorethumb_ml").setLevel(logging.DEBUG)
"""

import logging


def configure(level: str = "INFO") -> None:
    """Configure the root sorethumb_ml logger with a standard formatter."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
