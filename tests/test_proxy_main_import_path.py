from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "src" / "proxy_app" / "main.py"


def test_proxy_main_prefers_local_src_before_site_packages():
    source = MAIN_PATH.read_text(encoding="utf-8")

    assert "sys.path.append" not in source
    assert "sys.path.remove(_LOCAL_SRC_PATH)" in source
    assert "sys.path.insert(0, _LOCAL_SRC_PATH)" in source
    assert source.index("sys.path.remove(_LOCAL_SRC_PATH)") < source.index(
        "sys.path.insert(0, _LOCAL_SRC_PATH)"
    )
    assert source.index("sys.path.insert(0, _LOCAL_SRC_PATH)") < source.index(
        "from rotator_library.credential_tool import run_credential_tool"
    )
