"""Report the selected SageMaker training-job quota without creating resources."""

from __future__ import annotations

import argparse
import json
import os

from lava.readers.sagemaker import find_training_quota


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", default="ml.g6e.2xlarge")
    parser.add_argument("--spot", action="store_true")
    args = parser.parse_args()
    result = find_training_quota(
        region=os.environ.get("AWS_REGION", "us-west-2"),
        instance_type=args.instance_type,
        spot=args.spot,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SAGEMAKER_TRAINING_QUOTA_CHECKED")


if __name__ == "__main__":
    main()
