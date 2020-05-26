"""
Multiple infillers

Multiple infillers provide easy-to-use infiller options for the most common use-cases.
"""

from .decompose_collection_with_time_dep_ratio import (  # noqa: F401
    DecomposeCollectionTimeDepRatio,
)
from .infill_all_required_emissions_for_openscm import (  # noqa: F401
    infill_all_required_variables,
)
from .infill_composite_values import infill_composite_values  # noqa: F401
from .split_collection_with_remainder_emissions import (  # noqa: F401
    SplitCollectionWithRemainderEmissions,
)
