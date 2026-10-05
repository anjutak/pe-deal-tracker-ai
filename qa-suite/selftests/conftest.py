import sys
from pathlib import Path

# Let the self-tests import the suite's modules (test_deal_tracker, judge, compare_judge)
sys.path.insert(0, str(Path(__file__).parent.parent))
