import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.metrics import InvocationLog, MetricsTracker


def test_metrics_tracker_summary():
    tracker = MetricsTracker()
    tracker.record(InvocationLog("toolA", True, True, 0, 2.0, 100.0))
    tracker.record(InvocationLog("toolB", False, False, 1, 8.0, 200.0))
    summary = tracker.summarise()
    assert summary["calls"] == 2
    assert 0 <= summary["pass_at_1"] <= 1
    assert summary["avg_retries"] == 0.5
    assert "pass_at_1_low" in summary
