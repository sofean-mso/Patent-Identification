# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

import os

# Useful directories
PROJECT_ROOT = os.path.dirname(os.path.dirname( os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "/data")
SOURCE_DATA_ROOT = os.path.join(DATA_ROOT, "source")
NO_REFS_ARXIV_CS_DATA_ROOT = os.path.join(DATA_ROOT, "arxiv_no_refs")
