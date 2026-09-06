from __future__ import annotations

from lava.data.kaggle_sync import parse_page, write_csv


def test_parse_page_with_next_token() -> None:
    text = (
        "Next Page Token = next-token\nname,size,creationDate\ntrain.csv,123,2026-04-16 12:00:00\n"
    )
    rows, token = parse_page(text)
    assert token == "next-token"
    assert rows == [
        {
            "name": "train.csv",
            "size": "123",
            "creationDate": "2026-04-16 12:00:00",
        }
    ]


def test_parse_page_without_next_token() -> None:
    text = "name,size,creationDate\ntest.csv,456,2026-04-16 12:00:01\n"
    rows, token = parse_page(text)
    assert token is None
    assert rows[0]["name"] == "test.csv"


def test_write_csv_accepts_string_valued_mappings(tmp_path) -> None:
    path = tmp_path / "inventory.csv"
    rows = [{"name": "train.csv", "size": "123"}]

    write_csv(path, rows, ["name", "size"])

    assert path.read_text(encoding="utf-8").splitlines() == [
        "name,size",
        "train.csv,123",
    ]
