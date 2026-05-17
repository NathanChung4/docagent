"""Sample domain pack — public, generic vocabulary, committed to the repo.

Exports a `get_domain()` factory consumed by the core's get_domain() resolver.
"""

from domains.sample.config import SampleDomain


def get_domain() -> SampleDomain:
    return SampleDomain()


__all__ = ["SampleDomain", "get_domain"]
