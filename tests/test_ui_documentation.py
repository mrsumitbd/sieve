"""
tests/test_ui_documentation.py

Smoke test for the demo video on the Documentation page, using Streamlit's AppTest.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parents[1] / "src" / "sieve" / "ui" / "pages" / "1_Documentation.py")

VIDEO_URL = "https://youtu.be/1Imgn7kpkNg"
VIDEO_ID = "1Imgn7kpkNg"


def test_documentation_page_embeds_the_demo_video_once():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    videos = at.get("video")
    assert len(videos) == 1
    # Streamlit rewrites youtu.be links to the embed form, so match on the video id.
    assert VIDEO_ID in videos[0].proto.url


def test_documentation_page_has_the_plain_link_fallback():
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(f"]({VIDEO_URL})" in m.value for m in at.markdown)
