"""Reading and writing the GeoGuessr map format."""

from __future__ import annotations

import json

import pytest

from cr_labeler.geoguessr_io import GeoGuessrMap

SAMPLE = {
    "name": "*test",
    "customCoordinates": [
        {
            "lat": 52.5,
            "lng": 11.1,
            "heading": 206.9,
            "pitch": 41.0,
            "zoom": 0,
            "panoId": "abc123",
            "countryCode": None,
            "stateCode": None,
            "extra": {"tags": ["GT_2024"], "panoDate": "2022-04"},
        },
        {"lat": 1.0, "lng": 2.0, "panoId": "", "extra": {"tags": []}},
    ],
}


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return path


def test_reads_locations(sample):
    document = GeoGuessrMap.load(sample)
    assert len(document.locations) == 2
    assert document.locations[0].pano_id == "abc123"
    assert document.locations[0].ground_truth() == "2024"
    assert document.locations[0].lat == 52.5


def test_empty_pano_id_reads_as_missing(sample):
    assert GeoGuessrMap.load(sample).locations[1].pano_id is None


def test_tagging_preserves_every_other_field(sample, tmp_path):
    document = GeoGuessrMap.load(sample)
    document.locations[0].apply_tag("2024")
    out = tmp_path / "out.json"
    document.save(out)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["name"] == "*test"
    entry = written["customCoordinates"][0]
    assert entry["extra"]["tags"] == ["GT_2024", "CR_2024"]
    assert entry["extra"]["panoDate"] == "2022-04"
    for key in ("lat", "lng", "heading", "pitch", "zoom", "panoId", "countryCode"):
        assert entry[key] == SAMPLE["customCoordinates"][0][key]


def test_retagging_replaces_the_previous_cr_tag(sample, tmp_path):
    document = GeoGuessrMap.load(sample)
    document.locations[0].apply_tag("2024")
    document.locations[0].apply_tag("2025")
    assert document.locations[0].tags == ["GT_2024", "CR_2025"]


def test_tagging_is_idempotent(sample, tmp_path):
    first = tmp_path / "a.json"
    document = GeoGuessrMap.load(sample)
    for location in document.locations:
        location.apply_tag("2024")
    document.save(first)

    second = tmp_path / "b.json"
    again = GeoGuessrMap.load(first)
    for location in again.locations:
        location.apply_tag("2024")
    again.save(second)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_accepts_a_bare_array(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(SAMPLE["customCoordinates"]), encoding="utf-8")
    assert len(GeoGuessrMap.load(path).locations) == 2


def test_rejects_json_that_is_not_a_map(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"nope": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="customCoordinates"):
        GeoGuessrMap.load(path)


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        GeoGuessrMap.load(path)


def test_missing_extra_is_created_on_tagging():
    from cr_labeler.geoguessr_io import Location

    location = Location(index=0, raw={"lat": 1.0, "lng": 2.0})
    location.apply_tag("None")
    assert location.raw["extra"]["tags"] == ["CR_None"]


def test_describe_falls_back_to_coordinates():
    from cr_labeler.geoguessr_io import Location

    assert "1.500000" in Location(index=3, raw={"lat": 1.5, "lng": 2.5}).describe()
    assert "xyz" in Location(index=3, raw={"panoId": "xyz"}).describe()
