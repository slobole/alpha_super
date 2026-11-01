"""The picture step: any source becomes a small band, a bad one writes nothing."""

from PIL import Image
import pytest

from scripts import prepare_report_art as art


def _picture_path(tmp_path, name_str, size_tuple=(1672, 941)):
    path = tmp_path / name_str
    Image.new("RGB", size_tuple, "#336699").save(path)
    return path


@pytest.mark.parametrize("name_str", ["2026-11.png", "2026-11.jpg", "2026-11.webp"])
def test_a_picture_becomes_the_reports_band(tmp_path, name_str):
    output_path = art.prepare_picture_path(_picture_path(tmp_path, name_str), "2026-11", output_directory_path=tmp_path / "out")
    assert output_path.name == "2026-11.jpg"
    with Image.open(output_path) as image:
        assert image.size == (art.BAND_WIDTH_INT, art.BAND_HEIGHT_INT) and image.format == "JPEG"


@pytest.mark.parametrize("size_tuple,month_str,message_str", [((800, 450), "2026-11", "pixels wide"), ((1600, 200), "2026-11", "too wide"), ((1672, 941), "2026-13", "month")])
def test_a_bad_source_writes_nothing(tmp_path, size_tuple, month_str, message_str):
    with pytest.raises(art.ReportArtError, match=message_str):
        art.prepare_picture_path(_picture_path(tmp_path, "x.png", size_tuple), month_str, output_directory_path=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_an_unreadable_file_is_refused(tmp_path):
    path = tmp_path / "2026-11.png"
    path.write_bytes(b"not a picture")
    with pytest.raises(art.ReportArtError, match="not a readable picture"):
        art.prepare_picture_path(path, "2026-11", output_directory_path=tmp_path / "out")


def test_a_folder_gives_its_month_named_pictures_only(tmp_path):
    for name_str in ("2026-12.png", "2026-11.jpg", "cover.png", "2026-13.png", "2026-10.txt"):
        (tmp_path / name_str).write_bytes(b"x")
    assert [(path.name, month_str) for path, month_str in art.source_month_list([str(tmp_path)], None)] == [("2026-11.jpg", "2026-11"), ("2026-12.png", "2026-12")]
    with pytest.raises(art.ReportArtError, match="exactly one"):
        art.source_month_list([str(tmp_path)], "2026-11")


def test_the_command_reports_a_failure_without_a_traceback(tmp_path, capsys):
    assert art.main([str(tmp_path)]) == 1
    assert "No picture named by month" in capsys.readouterr().err
