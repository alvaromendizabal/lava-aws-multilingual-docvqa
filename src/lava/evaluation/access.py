"""Read-only Hugging Face readiness checks and actionable, sanitized access errors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalTokenNotFoundError

from lava.readers.runtime_logging import RuntimeEventLogger

LOGIN = "uv run --frozen --group judge hf auth login"
MODEL_PAGE = "https://huggingface.co/google/gemma-3-1b-it"
MINIMUM_AVAILABLE_MEMORY_BYTES = 8 * 1024**3


class EvaluationAccessError(RuntimeError):
    """An operator-actionable access problem, safe to show without a raw HTTP traceback."""


def available_memory_bytes(
    meminfo: Path = Path("/proc/meminfo"), cgroup: Path = Path("/sys/fs/cgroup")
) -> int:
    """Measure Linux available RAM, bounded by the container's remaining memory allowance."""
    try:
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].split()
            for line in meminfo.read_text().splitlines()
        }
        available = int(fields["MemAvailable"][0]) * 1024
        for limit_path, usage_path in (
            (cgroup / "memory.max", cgroup / "memory.current"),
            (cgroup / "memory/memory.limit_in_bytes", cgroup / "memory/memory.usage_in_bytes"),
        ):
            if limit_path.exists():
                limit = limit_path.read_text().strip()
                if limit != "max":
                    available = min(available, max(0, int(limit) - int(usage_path.read_text())))
        if available < 0:
            raise ValueError("Negative available memory")
    except (OSError, KeyError, IndexError, ValueError):
        raise EvaluationAccessError(
            "Cannot verify available memory on this host. Run CPU judging in the supported "
            "Linux SageMaker workspace; no model weights have been loaded."
        ) from None
    return available


def check_cpu_memory(
    logger: RuntimeEventLogger, *, available: Callable[[], int] = available_memory_bytes
) -> dict[str, float]:
    """Require conservative headroom before loading the float32 Gemma judge."""
    measured = available()
    details = {"available_memory_gib": measured / 1024**3, "minimum_available_memory_gib": 8.0}
    logger.emit(
        "judge.memory.checked",
        available_memory_gib=details["available_memory_gib"],
        minimum_available_memory_gib=8.0,
    )
    if measured < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise EvaluationAccessError(
            f"Only {details['available_memory_gib']:.2f} GiB of memory is available; CPU judging "
            "requires at least 8 GiB available before loading weights. In SageMaker Studio, "
            "save work, stop the lava-dev JupyterLab space, select ml.m7i.2xlarge (32 GiB), "
            "and run the space again. This changes billed CPU capacity. Then run "
            "`make evaluation-check`. Keep the space and its storage; do not delete them."
        )
    return details


@contextmanager
def hub_access_errors() -> Iterator[None]:
    """Translate expected Hub failures without exposing request headers or token details."""
    try:
        yield
    except LocalTokenNotFoundError:
        raise EvaluationAccessError(
            f"This terminal is not signed in to Hugging Face. Run `{LOGIN}`, choose "
            "'Log in with your browser', and finish authorization. No Claude account is needed. "
            "Then run `make evaluation-check`."
        ) from None
    except GatedRepoError:
        raise EvaluationAccessError(
            f"Gemma access was denied. Open {MODEL_PAGE} using the same Hugging Face account, "
            "accept Google's terms or request access, and confirm access is granted. "
            "If using a restricted token, it must allow this gated model. "
            "Then run `make evaluation-check`. No reader answers need to be regenerated."
        ) from None
    except HfHubHTTPError as error:
        status = error.response.status_code
        if status == 401:
            message = (
                f"Hugging Face rejected this terminal's login. Run `{LOGIN} --force`, "
                "finish browser authorization, then run `make evaluation-check`."
            )
        elif status == 403:
            message = (
                "Hugging Face authenticated the request but denied permission. Confirm the "
                f"account has access at {MODEL_PAGE} and the token permits that model. "
                "Then run `make evaluation-check`."
            )
        elif status == 404:
            message = (
                "The pinned Hugging Face resource could not be resolved. Check model access "
                "and the committed judge configuration; do not replace its revision with 'main'."
            )
        else:
            message = (
                f"Hugging Face returned HTTP {status}. Access could not be verified. "
                "Retry `make evaluation-check`; completed reader work remains saved."
            )
        raise EvaluationAccessError(message) from None
    except httpx.RequestError:
        raise EvaluationAccessError(
            "Could not reach Hugging Face. Check this machine's network access and retry "
            "`make evaluation-check`. This does not establish that your login is invalid."
        ) from None
    except OSError as error:
        # Transformers wraps gated Hub failures in OSError. Unwrap only a typed
        # Hub/network cause; unrelated filesystem failures must still propagate.
        cause = error.__cause__
        if isinstance(cause, (LocalTokenNotFoundError, HfHubHTTPError, httpx.RequestError)):
            with hub_access_errors():
                raise cause
        raise


def check_judge_access(
    config: Mapping[str, Any],
    logger: RuntimeEventLogger,
    *,
    api: Any = None,
    metadata: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify terminal identity and pinned-file access without downloading weights or using AWS."""
    with logger.stage("judge.access", heartbeat_seconds=15), hub_access_errors():
        account = (api if api is not None else HfApi()).whoami(token=True)
        username = account.get("name")
        if not isinstance(username, str) or not username:
            raise EvaluationAccessError("Hugging Face returned no verifiable account name.")
        lookup = metadata if metadata is not None else get_hf_file_metadata
        header = lookup(
            hf_hub_url(config["model_id"], "config.json", revision=config["model_revision"]),
            token=True,
            timeout=10,
            retry_on_errors=False,
        )
        if header.commit_hash != config["model_revision"]:
            raise EvaluationAccessError(
                "Hugging Face metadata differs from the pinned judge revision."
            )
        result = {
            "username": username,
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "downloaded_weights": False,
            "new_cloud_compute": False,
        }
        logger.emit("judge.access.verified", **result)
        return result
