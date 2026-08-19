from pathlib import Path


def test_q8_video_bridge_uses_one_sequential_ffmpeg_reader() -> None:
    source = Path("cpp/sam3_bridge.cpp").read_text()

    assert "class sequential_video_reader" in source
    assert "-f rawvideo -pix_fmt rgb24 pipe:1" in source
    assert "sam3_decode_video_frame" not in source


def test_q8_build_script_does_not_advertise_unsupported_cuda_runtime() -> None:
    source = Path("scripts/build_sam3_cpp.sh").read_text()

    assert 'if [[ "${BACKEND}" == "cuda" ]]' in source
    assert "initializes only CPU and Apple Metal backends" in source
    assert "-DGGML_CUDA=ON" not in source
