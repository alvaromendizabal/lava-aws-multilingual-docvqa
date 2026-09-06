from lava.readers.qwen35 import stable_question_seed


def test_stable_question_seed_is_repeatable_and_question_specific() -> None:
    assert stable_question_seed(42, "q1") == stable_question_seed(42, "q1")
    assert stable_question_seed(42, "q1") != stable_question_seed(42, "q2")
    assert 0 <= stable_question_seed(42, "q1") < 2**31 - 1
