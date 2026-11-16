import io

from PIL import Image

from src.data.vnmmkg_commons import (local_file_name, normalize_license, pick_candidates, strip_html,
                                     to_thumbnail_jpeg)

CFG = {"allowed_mime": ["image/jpeg", "image/png"], "min_image_width": 400, "max_image_candidates": 3}


def lic(code):
    return {"License": {"value": code}}


def test_normalize_license():
    assert normalize_license(lic("cc-by-sa-4.0")) == "CC BY-SA 4.0"
    assert normalize_license(lic("cc-by-2.0")) == "CC BY 2.0"
    assert normalize_license(lic("cc-by-sa-3.0-migrated")) == "CC BY-SA 3.0"
    assert normalize_license(lic("cc0")) == "CC0"
    assert normalize_license(lic("pd")) == "Public Domain"
    assert normalize_license(lic("cc-by-nc-sa-2.0")) is None
    assert normalize_license(lic("cc-by-nd-4.0")) is None
    assert normalize_license(lic("gfdl")) is None
    assert normalize_license({}) is None


def test_strip_html():
    assert strip_html('<a href="//x">Nguyễn &amp; Co</a>\n  ') == "Nguyễn & Co"


def test_local_file_name_is_stable_and_nfc():
    import unicodedata
    title = "File:Chùa Một Cột.jpg"
    assert local_file_name(title) == local_file_name(unicodedata.normalize("NFD", title))
    assert local_file_name(title).endswith(".jpg") and len(local_file_name(title)) == 20


def info(width=800, mime="image/jpeg", code="cc-by-sa-4.0", title="x"):
    return {"width": width, "height": 600, "mime": mime, "descriptionurl": f"https://commons/{title}",
            "thumburl": f"https://upload/{title}", "extmetadata": {
                "License": {"value": code}, "LicenseUrl": {"value": "https://creativecommons.org/x"},
                "Artist": {"value": "<b>Tác giả</b>"}}}


def test_pick_candidates_order_filters_and_cap():
    files = ["File:D.jpg", "File:A.jpg", "File:C.png", "File:B.jpg", "File:E.jpg"]
    infos = {"File:A.jpg": info(title="A"), "File:B.jpg": info(width=300, title="B"),
             "File:C.png": info(mime="image/png", title="C"), "File:D.jpg": info(title="D"),
             "File:E.jpg": info(code="cc-by-nc-2.0", title="E"), "File:P.jpg": info(title="P")}
    picked, drops = pick_candidates("File:P.jpg", files, infos, CFG)
    assert [p["file_title"] for p in picked] == ["File:P.jpg", "File:A.jpg", "File:C.png"]
    assert drops == {"too_small": 1}                         # cap reached before D and E are looked at
    assert picked[0]["artist"] == "Tác giả" and picked[0]["license"] == "CC BY-SA 4.0"
    picked, drops = pick_candidates(None, files, infos, {**CFG, "max_image_candidates": 12})
    assert [p["file_title"] for p in picked] == ["File:A.jpg", "File:C.png", "File:D.jpg"]
    assert drops == {"license": 1, "too_small": 1}


def png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_to_thumbnail_jpeg_downscales_never_upscales_and_is_deterministic():
    data, w, h = to_thumbnail_jpeg(png_bytes(1000, 500), 640)
    assert (w, h) == (640, 320) and data[:2] == b"\xff\xd8"
    assert to_thumbnail_jpeg(png_bytes(1000, 500), 640)[0] == data
    assert to_thumbnail_jpeg(png_bytes(500, 400), 640)[1:] == (500, 400)
