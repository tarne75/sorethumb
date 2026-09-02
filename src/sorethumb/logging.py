"""Logger setup helpers.

Every module obtains its logger as ``logging.getLogger(__name__)``, giving the
whole library one ``sorethumb.*`` namespace a caller can configure in one line::

    logging.getLogger("sorethumb").setLevel(logging.DEBUG)
"""

import logging


def configure(level: str = "INFO") -> None:
    """Configure the root sorethumb logger with a standard formatter."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
