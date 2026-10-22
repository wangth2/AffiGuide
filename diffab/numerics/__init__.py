"""Opt-in, versioned numerical policies for AffiGuide inference."""
from .backend import (BACKENDS, NumericalContext, configure_generator,
                      configure_ddg, configure_gearbind, sample_so3_categories,
                      apply_execution_flags, execution_flags)
