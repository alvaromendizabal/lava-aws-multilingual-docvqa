import hashlib

from lava.readers.oracle_assets import document_aliases, parse_training_csv


def test_parse_training_csv_and_aliases() -> None:
    payload = (
        b"id,file_id,question,answer_format,answer,evidence_page_number,language\n"
        b'q1,j_1,"question",string,"answer","[1,2]",ja\n'
        b'q2,v_1,"question",number,"10","[3]",vi\n'
    )
    records = parse_training_csv(payload)
    assert len(records) == 2
    assert records[0].evidence_pages == (1, 2)
    assert document_aliases(records) == {"j_1": "doc-01", "v_1": "doc-02"}
    assert hashlib.sha256(payload).hexdigest()
