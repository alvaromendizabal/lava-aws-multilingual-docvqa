import hashlib
import json
from pathlib import Path

import yaml

from lava.readers.sagemaker import build_job_plan, submit_or_preview_job


def _model_lock(path: Path) -> None:
    body = {
        "schema_version": 2,
        "generated_at_utc": "2026-09-02T00:00:00+00:00",
        "config_sha256": "f" * 64,
        "candidate_count": 1,
        "unique_model_repository_count": 1,
        "resolved_models": [
            {
                "model_key": "qwen",
                "model_id": "Qwen/Qwen3.5-4B",
                "expected_license": "apache-2.0",
                "expected_pipeline_tag": "image-text-to-text",
                "parameters_billion": 4.0,
                "instance_type": "ml.g5.2xlarge",
                "input_mode": "fused",
                "dtype": "bfloat16",
                "attention_implementation": "sdpa",
                "use_kernels": False,
                "processor_min_pixels": 200704,
                "processor_max_pixels": 1605632,
                "generation": {
                    "mode": "direct",
                    "max_new_tokens": 64,
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                    "min_p": None,
                    "repetition_penalty": 1.0,
                    "seed": 1,
                },
                "revision": "a" * 40,
                "observed_license": "apache-2.0",
                "observed_pipeline_tag": "image-text-to-text",
                "resolved_at_utc": "2026-09-02T00:00:00+00:00",
                "last_modified": None,
                "gated": False,
                "private": False,
            }
        ],
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path.write_text(json.dumps({**body, "registry_sha256": hashlib.sha256(canonical).hexdigest()}))


def test_job_plan_is_bounded_endpoint_free_and_preview_is_pure(tmp_path: Path) -> None:
    config = {
        "protocol_lock_id": "b" * 64,
        "benchmark": {
            "private_manifest_s3_key": "processed/manifest.jsonl",
            "results_prefix": "experiments/oracle",
        },
        "training_runtime": {
            "sdk_version": "3.21.0",
            "image_uri": "image",
            "image_digest": "sha256:" + "d" * 64,
            "instance_count": 1,
            "volume_size_gb": 100,
            "max_runtime_seconds": 3600,
            "max_wait_seconds": 3600,
            "managed_spot_for_smoke": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    lock_path = tmp_path / "lock.json"
    _model_lock(lock_path)

    import lava.readers.sagemaker as module

    original = module._git_sha
    module._git_sha = lambda _: "c" * 40
    try:
        plan = build_job_plan(
            repo_root=tmp_path,
            config_path=config_path,
            model_lock_path=lock_path,
            model_key="qwen",
            bucket="bucket",
            limit=1,
        )
    finally:
        module._git_sha = original
    result = submit_or_preview_job(
        plan=plan,
        repo_root=tmp_path,
        region="us-west-2",
        submit=False,
        wait=False,
        acknowledgement="NO",
    )
    assert plan.instance_count == 1
    assert plan.max_runtime_seconds == 3600
    assert plan.creates_endpoint is False
    assert result["submitted"] is False
    assert result["charged_resources_created"] is False


def test_quota_permission_error_is_reported_without_creating_resources(monkeypatch) -> None:
    from botocore.exceptions import ClientError

    import lava.readers.sagemaker as module

    class DeniedClient:
        def list_service_quotas(self, **_: object) -> dict[str, object]:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "denied",
                    }
                },
                "ListServiceQuotas",
            )

    monkeypatch.setattr(module.boto3, "client", lambda *args, **kwargs: DeniedClient())
    result = module.find_training_quota(
        region="us-west-2",
        instance_type="ml.g6e.2xlarge",
        spot=False,
    )
    assert result["found"] is False
    assert result["status"] == "permission_or_api_error"
    assert result["error_code"] == "AccessDeniedException"
