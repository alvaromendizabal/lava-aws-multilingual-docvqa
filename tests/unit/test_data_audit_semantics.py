"""Regression tests for answer-column semantics."""

from lava.data.audit import infer_semantic_columns, profile_csv


def test_answer_format_is_not_selected_as_answer() -> None:
    inferred = infer_semantic_columns(
        [
            "id",
            "file_id",
            "question",
            "answer_format",
            "answer",
            "evidence_page_number",
        ]
    )
    assert inferred["answer"] == ["answer"]


def test_train_profile_uses_actual_answer() -> None:
    payload = (
        "id,file_id,question,answer_format,answer,"
        "evidence_page_number,language\n"
        'q1,j_1,"これは何ですか",number,100,"[1]",ja\n'
    ).encode()

    profile = profile_csv(
        "raw/kaggle/train.csv",
        payload,
        {"j_1"},
    )

    assert profile["selected_columns"]["answer"] == "answer"
    assert profile["answer_type_counts"] == {"number": 1}


def test_test_profile_has_no_answer_label() -> None:
    payload = (
        'id,file_id,question,answer_format,language\nq1,j_1,"これは何ですか",string,ja\n'
    ).encode()

    profile = profile_csv(
        "raw/kaggle/test.csv",
        payload,
        {"j_1"},
    )

    assert profile["selected_columns"]["answer"] is None
    assert profile["answer_type_counts"] == {}
