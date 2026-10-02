"""Print a JSON report of the machine (device, driver, versions). Embedded in every benchmark result."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygpt.utils import env_report, resolve_device  # noqa: E402

if __name__ == "__main__":
    dev = resolve_device(sys.argv[1] if len(sys.argv) > 1 else "auto")
    print(json.dumps(env_report(dev, measure_bandwidth=True), indent=2))
