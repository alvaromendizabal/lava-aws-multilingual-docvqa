from lava.evaluation.schemas import AnswerFormat
from lava.readers.prompts import PROMPT_VERSION, SYSTEM_INSTRUCTION, build_reader_instruction


def test_prompt_is_answer_blind_and_has_schema() -> None:
    prompt = build_reader_instruction(
        question="質問は何ですか?",
        language="ja",
        answer_format=AnswerFormat.NUMBER,
        available_pages=(2, 4),
    )
    assert PROMPT_VERSION == "oracle-reader-json-v3"
    assert "質問は何ですか?" in prompt
    assert '"evidence_pages"' in prompt
    assert "[2, 4]" in prompt
    assert "outside knowledge" in SYSTEM_INSTRUCTION
