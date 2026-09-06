import pytest

from lava.evaluation.schemas import AnswerFormat
from lava.readers.parsing import ReaderOutputError, parse_reader_response, strip_hidden_reasoning


def test_strip_thinking_and_parse_fenced_json() -> None:
    raw = (
        "<think>private work</think>\n```json\n"
        '{"answer":"42","evidence_pages":[2],"confidence":0.9,"abstain":false}'
        "\n```"
    )
    prediction = parse_reader_response(
        question_id="q1",
        answer_format=AnswerFormat.NUMBER,
        raw_response=raw,
        allowed_pages=(2, 4),
    )
    assert prediction.answer == "42"
    assert prediction.evidence_pages == (2,)
    assert "private work" not in strip_hidden_reasoning(raw)


def test_list_answer_is_canonical_json() -> None:
    prediction = parse_reader_response(
        question_id="q1",
        answer_format=AnswerFormat.UNORDERED_LIST,
        raw_response=('{"answer":["A","B"],"evidence_pages":[1],"confidence":0.5,"abstain":false}'),
        allowed_pages=(1,),
    )
    assert prediction.answer == '["A", "B"]'


def test_out_of_scope_page_is_rejected_with_code() -> None:
    with pytest.raises(ReaderOutputError) as caught:
        parse_reader_response(
            question_id="q1",
            answer_format=AnswerFormat.STRING,
            raw_response=('{"answer":"x","evidence_pages":[9],"confidence":0.5,"abstain":false}'),
            allowed_pages=(1,),
        )
    assert caught.value.code == "out_of_scope_page"


def test_extra_output_key_is_rejected() -> None:
    with pytest.raises(ReaderOutputError) as caught:
        parse_reader_response(
            question_id="q1",
            answer_format=AnswerFormat.STRING,
            raw_response=(
                '{"answer":"x","evidence_pages":[1],"confidence":0.5,'
                '"abstain":false,"reasoning":"hidden"}'
            ),
            allowed_pages=(1,),
        )
    assert caught.value.code == "schema_validation_failed"


def test_non_finite_confidence_is_rejected() -> None:
    with pytest.raises(ReaderOutputError, match="Confidence"):
        parse_reader_response(
            question_id="q1",
            answer_format=AnswerFormat.STRING,
            raw_response=('{"answer":"x","evidence_pages":[1],"confidence":NaN,"abstain":false}'),
            allowed_pages=(1,),
        )


def test_string_confidence_is_rejected_in_strict_mode() -> None:
    with pytest.raises(ReaderOutputError) as caught:
        parse_reader_response(
            question_id="q1",
            answer_format=AnswerFormat.STRING,
            raw_response=('{"answer":"x","evidence_pages":[1],"confidence":"0.5","abstain":false}'),
            allowed_pages=(1,),
        )
    assert caught.value.code == "schema_validation_failed"


def test_integer_like_string_page_is_rejected() -> None:
    with pytest.raises(ReaderOutputError) as caught:
        parse_reader_response(
            question_id="q1",
            answer_format=AnswerFormat.STRING,
            raw_response=('{"answer":"x","evidence_pages":["1"],"confidence":0.5,"abstain":false}'),
            allowed_pages=(1,),
        )
    assert caught.value.code == "schema_validation_failed"
