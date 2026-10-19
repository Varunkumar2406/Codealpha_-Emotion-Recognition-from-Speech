import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ser.data import parse_ravdess_name, parse_tess_name, parse_emodb_name, build_index

def test_parsers():
    assert parse_ravdess_name("03-01-06-01-02-01-12.wav") == ("fear", "rav_12")
    assert parse_ravdess_name("03-01-02-01-02-01-05.wav") == ("neutral", "rav_05")   # calm->neutral
    assert parse_ravdess_name("03-02-06-01-02-01-12.wav") is None                    # song dropped
    assert parse_tess_name("OAF_back_ps.wav") == ("surprise", "tess_OAF")
    assert parse_tess_name("YAF_bar_angry.wav") == ("angry", "tess_YAF")
    assert parse_emodb_name("03a01Fa.wav") == ("happy", "emodb_03")
    assert parse_emodb_name("16b10Wb.wav") == ("angry", "emodb_16")
    assert parse_emodb_name("08a02La.wav") is None                                   # boredom dropped

def test_build_index():
    with tempfile.TemporaryDirectory() as d:
        for f in ["OAF_back_sad.wav", "YAF_bar_fear.wav", "junk.txt"]:
            open(os.path.join(d, f), "w").close()
        df = build_index(tess=d)
        assert len(df) == 2 and set(df.emotion) == {"sad", "fear"}

if __name__ == "__main__":
    test_parsers(); test_build_index(); print("all data tests passed")
