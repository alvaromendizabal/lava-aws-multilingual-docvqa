"""Synthetic integration contracts, not model-quality evidence or benchmark results."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.validate import validate_parameters

from lava.evaluation.judges import NormalizedExactJudge
from lava.evaluation.schemas import ReferenceRecord
from lava.evaluation.semantic import digest, encode
from lava.evaluation.system import load_summary, score_predictions
from lava.readers.parsing import ReaderOutputError, parse_reader_response
from lava.readers.qwen35 import Qwen35Reader
from lava.readers.runtime_logging import RuntimeEventLogger
from lava.readers.schemas import OracleExample, OraclePageAsset, ReaderInput, ReaderPrediction, ReaderTelemetry
from lava.readers.system import infer_requests, put_blob, select_pages, validate_manifest, validate_prediction
from lava.readers.system_execution import bootstrap, ensure_job, job_name, training_request, validate_remote

ROOT = Path(__file__).resolve().parents[2]


class MemoryStore:
    def __init__(self):
        self.values = {}

    def read(self, key):
        return deepcopy(self.values.get(key))

    def write(self, key, value):
        if key in self.values and self.values[key] != value:
            raise ValueError("Conflicting completion")
        self.values[key] = deepcopy(value)


def asset(number=1, document=1):
    return OraclePageAsset(
        asset_version="test", document_id=f"private-pdf-{document:02d}",
        document_alias=f"doc-{document:02d}", page_number=number,
        source_pdf_s3_uri="s3://test/source.pdf", source_pdf_sha256="a"*64,
        image_s3_uri="s3://test/image.png", image_sha256="b"*64,
        text_s3_uri="s3://test/text.txt", text_sha256="c"*64,
        layout_s3_uri="s3://test/layout.json", layout_sha256="d"*64,
        width_pixels=100, height_pixels=100, dpi=180, native_text_characters=1,
        word_count=1, text_block_count=1, embedded_image_count=0,
    )


def request(index=1, pages=(1, 3, 5, 7, 9)):
    document = (index-1) % 5 + 1
    return ReaderInput(
        question_id=f"private-question-{index:02d}", document_id=f"private-pdf-{document:02d}",
        document_alias=f"doc-{document:02d}", question="Private question text", answer_format="string",
        language="ja", pages=tuple(asset(number, document) for number in pages),
    )


def telemetry(images=5):
    return ReaderTelemetry(
        model_load_seconds=0, preprocessing_seconds=1, generation_seconds=2, total_seconds=3,
        prompt_tokens=10, generated_tokens=2, image_count=images, total_image_pixels=images*10000,
        raw_response_characters=100, peak_cuda_memory_allocated_mib=1,
        peak_cuda_memory_reserved_mib=2, gpu_name="synthetic", cuda_compute_capability="synthetic",
        torch_version="test", transformers_version="test", dtype="bfloat16",
        attention_implementation="sdpa", deterministic_algorithms_enabled=True, template_switch_supported=True,
    )


def fixture():
    inputs = [request(index).model_dump(mode="json") for index in range(1, 17)]
    contract = {"contract_id": "1"*64, "config": {
        "question_count":16, "document_count":5, "page_budget":5,
        "max_runtime_seconds":1800, "max_pending_seconds":86400,
    }}
    manifest = {"schema_version":1, "contract":contract, "inputs":inputs, "inputs_sha256":digest(encode(inputs))}
    raws = {row["question_id"]: '{"answer":"x","evidence_pages":[1],"confidence":0.5,"abstain":false}' for row in inputs}

    def predict(row):
        return parse_reader_response(question_id=row.question_id, answer_format=row.answer_format,
                                     raw_response=raws[row.question_id], allowed_pages=row.available_pages), telemetry()

    return manifest, contract, raws, predict


def test_label_free_input_rejects_gold_fields_and_five_pages_are_accepted():
    value = request().model_dump()
    assert not {"answer", "evidence_pages", "gold_evidence_pages"}.intersection(value)
    assert len(ReaderInput.model_validate(value).pages) == 5
    for field in ("answer", "evidence_pages", "gold_evidence_pages"):
        with pytest.raises(ValueError):
            ReaderInput.model_validate({**value, field:"untrusted label"})


def test_existing_oracle_contract_remains_gold_aligned_and_four_page_bounded():
    row = request(pages=(1,))
    original = OracleExample(protocol_lock_id="a"*64, answer="private label", evidence_pages=(1,), **row.model_dump())
    changed = original.model_copy(update={"answer":"different label"})
    assert ReaderInput.from_oracle(original) == ReaderInput.from_oracle(changed)
    with pytest.raises(ValueError):
        OracleExample(protocol_lock_id="a"*64, answer="label", evidence_pages=(2,), **row.model_dump())
    with pytest.raises(ValueError):
        OracleExample(protocol_lock_id="a"*64, answer="label", evidence_pages=(1,3,5,7,9), **request().model_dump())


@pytest.mark.parametrize("pages", [(3,1), (1,1)])
def test_reader_inputs_reject_unsorted_or_duplicate_physical_pages(pages):
    with pytest.raises(ValueError):
        request(pages=pages)


def test_reader_inputs_reject_cross_document_assets():
    value = request().model_dump()
    value["pages"] = (asset(1,2),)
    with pytest.raises(ValueError):
        ReaderInput.model_validate(value)


@pytest.mark.parametrize("order,budget", [([1,1],1),([0,1],1),([1,3],1),([True,2],1),([1,2],True),([1,2],0)])
def test_invalid_rankings_are_rejected(order,budget):
    with pytest.raises(ValueError):
        select_pages({"bm25":order},2,budget)


def test_rank_selection_preserves_physical_page_numbers_and_document_order():
    assert select_pages({"bm25":[7,3,1,4,2,6,5]},7,5) == (1,2,3,4,7)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "hash", "contract", "label"])
def test_input_manifest_rejects_partial_corrupt_and_label_bearing_inputs(mutation):
    manifest, contract, _, _ = fixture()
    if mutation == "duplicate": manifest["inputs"][1] = manifest["inputs"][0]
    if mutation == "missing": manifest["inputs"].pop()
    if mutation == "hash": manifest["inputs_sha256"] = "0"*64
    if mutation == "contract": contract = {**contract, "contract_id":"2"*64}
    if mutation == "label": manifest["inputs"][0]["answer"] = "leak"
    with pytest.raises(ValueError): validate_manifest(manifest,contract)


def test_interrupted_inference_resumes_only_unfinished_questions_and_all_cached_skips_model(capsys):
    manifest, contract, raw, predict = fixture()
    store = MemoryStore()
    calls = []
    def interrupted(row):
        calls.append(row.question_id)
        if len(calls) == 4: raise KeyboardInterrupt()
        return predict(row)
    logger = RuntimeEventLogger("system.test")
    with pytest.raises(KeyboardInterrupt):
        infer_requests(manifest,contract,store,interrupted,logger,read_raw=raw.__getitem__)
    assert len(store.values) == 3
    resumed = Mock(side_effect=predict)
    result = infer_requests(manifest,contract,store,resumed,logger,read_raw=raw.__getitem__)
    assert resumed.call_count == 13
    forbidden = Mock(side_effect=AssertionError("Cached inference must not load or invoke a model"))
    assert infer_requests(manifest,contract,store,forbidden,logger,read_raw=forbidden) == result
    forbidden.assert_not_called()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(e["timestamp_utc"].endswith("+00:00") and e["elapsed_seconds"] >= 0 for e in events)
    assert events[-1]["new_count"] == 0 and events[-1]["reused_count"] == 16
    assert events[-1]["total_elapsed_seconds"] >= 0


def test_invalid_model_output_is_a_completed_counted_failure_not_silently_retried():
    row = request()
    raw = "not json"
    with pytest.raises(ReaderOutputError) as error:
        parse_reader_response(question_id=row.question_id,answer_format=row.answer_format,raw_response=raw,allowed_pages=row.available_pages)
    prediction = ReaderPrediction(question_id=row.question_id,answer_format=row.answer_format,answer="",
                                  evidence_pages=(),confidence=0,abstain=True,schema_valid=False,
                                  parser_error_code=error.value.code,raw_response_sha256=digest(raw.encode()))
    saved = {"input_sha256":digest(encode(row.model_dump(mode="json"))),"raw_response":raw,
             "prediction":prediction.model_dump(),"telemetry":telemetry().model_dump()}
    assert not validate_prediction(row,saved)[0].schema_valid
    saved["prediction"]["answer"] = "invented success"
    with pytest.raises(ValueError): validate_prediction(row,saved)


def test_scoring_uses_predicted_not_gold_pages_and_preserves_all_denominators():
    manifest, contract, raw, predict = fixture()
    result = infer_requests(manifest,contract,MemoryStore(),predict,RuntimeEventLogger("test"),read_raw=raw.__getitem__)
    refs = tuple(ReferenceRecord(question_id=r.question_id,document_id=r.document_id,question=r.question,
                                answer_format=r.answer_format,language=r.language,answer="x",evidence_pages=(2,))
                 for r in validate_manifest(manifest,contract))
    scored = score_predictions(manifest,result,refs,NormalizedExactJudge())
    assert scored["metrics"]["question_count"] == 16
    assert scored["metrics"]["document_count"] == 5
    assert scored["metrics"]["question_micro"] == {"answer":1.0,"grounding":0.0,"overall":0.5}
    assert scored["diagnostics"]["all_evidence_retrieved"] == 0
    assert scored["diagnostics"]["failure_counts"] == {"missing_evidence":16}
    public = json.dumps(scored)
    assert "private-question" not in public and "Private question" not in public and "private-pdf" not in public
    with pytest.raises(ValueError): score_predictions(manifest,{**result,"records":result["records"][:-1]},refs,NormalizedExactJudge())


@pytest.mark.parametrize("bad_sha,bad_version", [(False,False),(True,False),(False,True)])
def test_reader_verifies_asset_bytes_versions_and_closes_stream(bad_sha,bad_version):
    reader = object.__new__(Qwen35Reader)
    stream = io.BytesIO(b"image")
    reader.s3 = Mock()
    reader.s3.get_object.return_value = {"Body":stream,"VersionId":"wrong" if bad_version else "v1"}
    if bad_sha or bad_version:
        with pytest.raises(ValueError):
            reader._get_bytes("s3://test/image.png",expected_sha256="a"*64 if bad_sha else digest(b"image"),version_id="v1")
    else:
        assert reader._get_bytes("s3://test/image.png",expected_sha256=digest(b"image"),version_id="v1") == b"image"
    assert stream.closed


def test_new_immutable_assets_do_not_require_ungranted_get_object_version():
    reader = object.__new__(Qwen35Reader)
    reader.s3 = Mock()
    reader.s3.get_object.return_value = {"Body":io.BytesIO(b"image")}
    reader._get_bytes("s3://test/image.png",expected_sha256=digest(b"image"))
    assert "VersionId" not in reader.s3.get_object.call_args.kwargs


def test_immutable_blob_rejects_conflicting_completion():
    s3 = Mock()
    s3.put_object.side_effect = ClientError({"Error":{"Code":"PreconditionFailed"}},"PutObject")
    s3.get_object.return_value = {"Body":io.BytesIO(b"wrong"),"Metadata":{"sha256":digest(b"wrong")}}
    with pytest.raises(ValueError,match="read-back"):
        put_blob(s3,"bucket","key",b"expected","application/json")


def test_absent_end_to_end_report_means_pending_not_zero(tmp_path):
    assert load_summary(tmp_path) is None


def test_job_request_passes_botocore_contract_and_enforces_cloud_limits():
    manifest, _, _, _ = fixture()
    request_body = training_request(ROOT,manifest,"bucket","us-west-2","arn:aws:iam::123456789012:role/test",1,"source.tar.gz","a"*64,"b"*40)
    service = boto3.Session()._session.get_service_model("sagemaker")
    validate_parameters(request_body,service.operation_model("CreateTrainingJob").input_shape)
    assert request_body["ResourceConfig"]["InstanceCount"] == 1
    assert request_body["StoppingCondition"] == {"MaxRuntimeInSeconds":1800,"MaxPendingTimeInSeconds":86400}
    assert all(len(arg) <= 256 for arg in request_body["AlgorithmSpecification"]["ContainerArguments"])
    assert "@sha256:" in request_body["AlgorithmSpecification"]["TrainingImage"]
    assert "RetryStrategy" not in request_body
    validate_remote({**request_body,"TrainingJobStatus":"Completed"},request_body)
    with pytest.raises(ValueError):
        validate_remote({**request_body,"Environment":{}},request_body)


@pytest.mark.parametrize("identity,attempt", [("x",1),("a"*64,0),("a"*64,True),("a"*64,100)])
def test_job_name_rejects_invalid_identity_and_attempt(identity,attempt):
    with pytest.raises(ValueError): job_name(identity,attempt)


def test_source_bootstrap_has_checksum_verification_before_extraction():
    text = bootstrap("a"*64,"bucket","us-west-2","b"*64)
    assert text.index("sha256sum") < text.index("tar -xzf")


@pytest.mark.parametrize("retry,attempt", [(False,1),(False,2)])
def test_paid_job_cannot_be_created_without_explicit_acknowledgement(retry,attempt):
    client = Mock()
    client.describe_training_job.side_effect = ClientError({"Error":{"Code":"ResourceNotFound"}},"DescribeTrainingJob")
    with pytest.raises(ValueError):
        ensure_job(client,{"TrainingJobName":"test"},acknowledge_charges="NO",allow_retry=retry,attempt=attempt,logger=RuntimeEventLogger("test"))
    client.create_training_job.assert_not_called()


def test_ambiguous_create_timeout_reattaches_instead_of_creating_another_job():
    manifest, _, _, _ = fixture()
    body = training_request(ROOT,manifest,"bucket","us-west-2","arn:aws:iam::123456789012:role/test",1,"source","a"*64,"b"*40)
    client = Mock()
    missing = ClientError({"Error":{"Code":"ResourceNotFound"}},"DescribeTrainingJob")
    client.describe_training_job.side_effect = [missing,{**body,"TrainingJobStatus":"InProgress"}]
    client.get_paginator.return_value.paginate.return_value = [{"TrainingJobSummaries":[]}]
    client.create_training_job.side_effect = EndpointConnectionError(endpoint_url="https://sagemaker.test")
    result = ensure_job(client,body,acknowledge_charges="YES",allow_retry=False,attempt=1,logger=RuntimeEventLogger("test"))
    assert result["TrainingJobStatus"] == "InProgress"
    assert client.create_training_job.call_count == 1


def test_remote_progress_does_not_expose_arbitrary_log_fields(capsys):
    from lava.readers.system_execution import stream_progress
    client = Mock()
    record = {"event":"system.question.completed","completed":3,"total":16,
              "raw_response":"PRIVATE_ANSWER","question":"PRIVATE_QUESTION","elapsed_seconds":2}
    client.get_paginator.return_value.paginate.return_value = [{"events":[
        {"eventId":"one","message":json.dumps(record)},
        {"eventId":"two","message":"arbitrary PRIVATE_MESSAGE"},
    ]}]
    seen = set()
    stream_progress(client,"job",RuntimeEventLogger("test"),seen)
    stream_progress(client,"job",RuntimeEventLogger("test"),seen)
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert output.count('"system.remote.progress"') == 1


def test_preparation_reuses_real_pdf_extraction_ranking_and_rendered_assets(tmp_path, monkeypatch):
    import csv
    import shutil

    import pymupdf

    from lava.readers import system
    from lava.retrieval import evaluation

    class S3:
        def __init__(self): self.values = {}
        def put_object(self, *, Key, Body, Metadata, **kwargs):
            existing = self.values.get(Key)
            if kwargs.get("IfNoneMatch") == "*" and existing is not None:
                raise ClientError({"Error":{"Code":"PreconditionFailed"}},"PutObject")
            if "IfMatch" in kwargs and (existing is None or kwargs["IfMatch"] != digest(existing[0])):
                raise ClientError({"Error":{"Code":"PreconditionFailed"}},"PutObject")
            self.values[Key] = (Body,Metadata)
            return {"VersionId":"v1"}
        def get_object(self, *, Key, **kwargs):
            data,metadata = self.values[Key]
            return {"Body":io.BytesIO(data),"Metadata":metadata,"ETag":digest(data),"VersionId":"v1"}

    s3 = S3()
    sources = {}
    def install(name, data):
        sources[name] = {"key":name,"sha256":digest(data),"version_id":"v1"}
        s3.values[name] = (data,{"sha256":digest(data)})
    for index in range(1,6):
        with pymupdf.open() as document:
            for page in range(7):
                document.new_page().insert_text((72,72),f"target on page {page+1}")
            install(f"pdf-{index}.pdf",document.tobytes())
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["id","file_id","question","answer_format","answer","evidence_page_number","language"])
    for index in range(16):
        writer.writerow([f"question-{index:02d}",f"pdf-{index%5+1}","target","string","label","[7]","ja"])
    install("train.csv",text.getvalue().encode())
    contract = system.system_contract(ROOT)
    monkeypatch.setattr(system,"system_contract",lambda _:contract)
    monkeypatch.setattr(evaluation,"pilot_sources",lambda *_:sources)
    (tmp_path/"configs").mkdir()
    shutil.copy(ROOT/"configs/oracle_reader_benchmark.yaml",tmp_path/"configs")
    logger = RuntimeEventLogger("synthetic.preparation")
    manifest = system.prepare_inputs(tmp_path,s3,"bucket",logger)
    inputs = validate_manifest(manifest,contract)
    assert all(row.available_pages == (1,2,3,4,5) for row in inputs)
    assert all(not hasattr(row,"answer") for row in inputs)
    assert all(asset.image_version_id is None for row in inputs for asset in row.pages)
    monkeypatch.setattr(system,"_render_page",Mock(side_effect=AssertionError("No duplicate rendering")))
    monkeypatch.setattr(system,"extract_pages",Mock(side_effect=AssertionError("No duplicate extraction")))
    monkeypatch.setattr(system,"rank_query",Mock(side_effect=AssertionError("No duplicate ranking")))
    assert system.prepare_inputs(tmp_path,s3,"bucket",logger) == manifest


def test_synthetic_scoring_rejects_a_changed_reference_question():
    manifest, contract, raw, predict = fixture()
    result = infer_requests(manifest,contract,MemoryStore(),predict,RuntimeEventLogger("test"),read_raw=raw.__getitem__)
    refs = tuple(ReferenceRecord(question_id=r.question_id,document_id=r.document_id,question="different",
                                answer_format=r.answer_format,language=r.language,answer="x",evidence_pages=(1,))
                 for r in validate_manifest(manifest,contract))
    with pytest.raises(ValueError,match="frozen reference"):
        score_predictions(manifest,result,refs,NormalizedExactJudge())
