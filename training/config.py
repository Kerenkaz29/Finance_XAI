"""
Paths and config for the training pipeline.
Set XAI_DATA_DIR to the folder containing loan_data_set.csv, american_bankruptcy.csv, etc.
"""
import os

# Default: datasets in project sibling folder. Override with XAI_DATA_DIR env var.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "datasets")
if os.name == "nt" and not os.path.isdir(_DATA_DIR):
    _DATA_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "datasets")
DATA_DIR = os.environ.get("XAI_DATA_DIR", _DATA_DIR)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Dataset filenames (relative to DATA_DIR or absolute)
LOAN_APPROVAL_CSV = "loan_data_set.csv"
BANKRUPTCY_CSV = "american_bankruptcy.csv"
# Give Me Some Credit: extract GiveMeSomeCredit.zip; common filenames:
CREDIT_RISK_CSV = "cs-training.csv"  # or "GiveMeSomeCredit.csv" depending on zip contents

# Training defaults
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
