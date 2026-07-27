"""Shared constants for hydromend."""
import re

BAD_QC_FLAGS = ("N", "M", "T")
LAG_FEATURE_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>-?\d+(?:\.\d+)?)$")
