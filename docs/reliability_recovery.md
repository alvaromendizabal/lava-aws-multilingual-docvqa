# Recoverable question answering

A job reporting success is not evidence that every answer is valid. The retained pipeline checks each question's source, model, input and output identity before accepting a saved result. It re-parses generations and distinguishes a completed checkpoint from an interrupted write.

[Checkpoint implementation](../src/lava/readers/checkpoints.py) · [Private-artifact safety](../src/lava/readers/private_artifacts.py) · [Artifact safety tests](../tests/unit/test_private_artifacts.py)

The historical closeout recorded **1,081 unchanged checkpoints**, no new model calls and two unresolved answers. The [public verifier](../examples/verify_submission_closeout.py) recomputes aggregate consistency; it cannot independently authenticate private checkpoint comparisons or make the unresolved answers correct.

The original project-specific AWS spaces and buckets were deleted after closeout. Reproduction now starts with committed fixtures and aggregate evidence, not live cloud backups. The reusable storage/identity contracts remain inspectable; operational retry, monitoring and cleanup scripts for the retired account resources were removed.
