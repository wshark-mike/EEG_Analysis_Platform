"""Tests for utils/data_loader.py."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import mne

from utils.data_loader import (
    get_supported_formats,
    load_eeg_file,
    _load_csv_as_raw,
    _rewrite_brainvision_header,
    load_brainvision_files,
    get_raw_info,
)


def _create_sample_raw(n_channels=4, sfreq=256.0, duration=2.0):
    """Create a synthetic MNE Raw object for testing."""
    n_times = int(sfreq * duration)
    data = np.random.RandomState(42).randn(n_channels, n_times) * 1e-6
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


class TestGetSupportedFormats:
    def test_returns_dict(self):
        formats = get_supported_formats()
        assert isinstance(formats, dict)

    def test_contains_edf(self):
        formats = get_supported_formats()
        assert ".edf" in formats

    def test_contains_csv(self):
        formats = get_supported_formats()
        assert ".csv" in formats

    def test_contains_vhdr(self):
        formats = get_supported_formats()
        assert ".vhdr" in formats

    def test_returns_copy(self):
        f1 = get_supported_formats()
        f2 = get_supported_formats()
        f1[".xyz"] = "test"
        assert ".xyz" not in f2


class TestLoadCSV:
    def test_load_csv_basic(self):
        """Test loading a simple CSV file."""
        n_channels = 3
        n_samples = 512
        data = np.random.RandomState(42).randn(n_samples, n_channels) * 50
        df = pd.DataFrame(data, columns=["Fp1", "Fp2", "Cz"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            raw = _load_csv_as_raw(tmp_path, sfreq=256.0)
            assert isinstance(raw, mne.io.RawArray)
            assert len(raw.ch_names) == n_channels
            assert raw.info["sfreq"] == 256.0
            assert raw.ch_names == ["Fp1", "Fp2", "Cz"]
        finally:
            os.unlink(tmp_path)

    def test_load_csv_with_time_column(self):
        """Test CSV loading drops time column."""
        n_samples = 100
        df = pd.DataFrame({
            "time": np.arange(n_samples) / 256.0,
            "Ch1": np.random.randn(n_samples),
            "Ch2": np.random.randn(n_samples),
        })

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            raw = _load_csv_as_raw(tmp_path)
            assert "time" not in raw.ch_names
            assert len(raw.ch_names) == 2
        finally:
            os.unlink(tmp_path)


class TestLoadEEGFile:
    def test_load_csv_via_load_eeg_file(self):
        """Test loading CSV through the main interface."""
        df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        })

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            raw = load_eeg_file(tmp_path)
            assert isinstance(raw, mne.io.BaseRaw)
        finally:
            os.unlink(tmp_path)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_eeg_file("test.xyz", file_type=".xyz")

    def test_load_fif(self):
        """Test loading a FIF file."""
        raw = _create_sample_raw()
        with tempfile.NamedTemporaryFile(suffix="_raw.fif", delete=False) as f:
            raw.save(f.name, overwrite=True, verbose=False)
            tmp_path = f.name

        try:
            loaded = load_eeg_file(tmp_path, file_type=".fif")
            assert isinstance(loaded, mne.io.BaseRaw)
            assert len(loaded.ch_names) == 4
        finally:
            os.unlink(tmp_path)


class TestGetRawInfo:
    def test_basic_info(self):
        raw = _create_sample_raw(n_channels=3, sfreq=128.0, duration=1.0)
        info = get_raw_info(raw)

        assert info["n_channels"] == 3
        assert info["sfreq"] == 128.0
        assert info["n_samples"] == 128
        assert abs(info["duration_sec"] - 1.0) < 0.01
        assert len(info["ch_names"]) == 3
        assert len(info["ch_types"]) == 3


def _create_brainvision_files(tmp_dir, n_channels=3, sfreq=256.0, n_samples=512):
    """Create minimal synthetic BrainVision (.vhdr, .eeg, .vmrk) files."""
    ch_names = [f"Ch{i+1}" for i in range(n_channels)]
    basename = "test"

    # Generate binary data (IEEE float 32)
    rng = np.random.RandomState(42)
    data = rng.randn(n_samples, n_channels).astype(np.float32)

    # .eeg — binary data (samples interleaved across channels)
    eeg_path = os.path.join(tmp_dir, f"{basename}.eeg")
    data.tofile(eeg_path)

    # .vmrk — marker file
    vmrk_path = os.path.join(tmp_dir, f"{basename}.vmrk")
    vmrk_content = (
        "Brain Vision Data Exchange Marker File Version 1.0\n"
        "\n"
        "[Common Infos]\n"
        f"Codepage=UTF-8\n"
        f"DataFile={basename}.eeg\n"
        "\n"
        "[Marker Infos]\n"
        "; Each entry: Mk<Marker number>=<Type>,<Description>,"
        "<Position in data points>,<Size in data points>,<Channel number>\n"
        f"Mk1=New Segment,,1,1,0,00000000000000000000\n"
    )
    with open(vmrk_path, "w") as f:
        f.write(vmrk_content)

    # .vhdr — header file
    ch_section = "\n".join(
        f"Ch{i+1}={name},,1" for i, name in enumerate(ch_names)
    )
    vhdr_content = (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "\n"
        "[Common Infos]\n"
        f"Codepage=UTF-8\n"
        f"DataFile={basename}.eeg\n"
        f"MarkerFile={basename}.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={n_channels}\n"
        f"SamplingInterval={1e6 / sfreq}\n"
        "\n"
        "[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n"
        "\n"
        "[Channel Infos]\n"
        f"{ch_section}\n"
    )
    vhdr_path = os.path.join(tmp_dir, f"{basename}.vhdr")
    with open(vhdr_path, "w") as f:
        f.write(vhdr_content)

    return vhdr_path, eeg_path, vmrk_path


class TestRewriteBrainvisionHeader:
    def test_rewrites_datafile(self):
        header = "DataFile=old.eeg\nMarkerFile=old.vmrk\n"
        result = _rewrite_brainvision_header(header, "new.eeg", "new.vmrk")
        assert "DataFile=new.eeg" in result
        assert "MarkerFile=new.vmrk" in result

    def test_preserves_other_content(self):
        header = "[Common Infos]\nDataFile=x.eeg\nMarkerFile=x.vmrk\nDataFormat=BINARY\n"
        result = _rewrite_brainvision_header(header, "y.eeg", "y.vmrk")
        assert "DataFormat=BINARY" in result


class TestLoadBrainVision:
    def test_load_vhdr_via_load_eeg_file(self):
        """Test loading BrainVision .vhdr through the main interface."""
        tmp_dir = tempfile.mkdtemp()
        try:
            vhdr_path, _, _ = _create_brainvision_files(tmp_dir, n_channels=3)
            raw = load_eeg_file(vhdr_path, file_type=".vhdr")
            assert isinstance(raw, mne.io.BaseRaw)
            assert len(raw.ch_names) == 3
        finally:
            for f in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)

    def test_load_brainvision_files_from_buffers(self):
        """Test load_brainvision_files with in-memory buffers."""
        tmp_dir = tempfile.mkdtemp()
        try:
            vhdr_path, eeg_path, vmrk_path = _create_brainvision_files(
                tmp_dir, n_channels=4, sfreq=500.0, n_samples=1000,
            )
            with open(vhdr_path, "rb") as f:
                vhdr_buf = f.read()
            with open(eeg_path, "rb") as f:
                eeg_buf = f.read()
            with open(vmrk_path, "rb") as f:
                vmrk_buf = f.read()

            raw = load_brainvision_files(vhdr_buf, eeg_buf, vmrk_buf)
            assert isinstance(raw, mne.io.BaseRaw)
            assert len(raw.ch_names) == 4
            assert raw.info["sfreq"] == 500.0
        finally:
            for f in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)

    def test_load_brainvision_files_temp_cleanup(self):
        """Verify temp files are cleaned up after loading."""
        tmp_dir = tempfile.mkdtemp()
        try:
            vhdr_path, eeg_path, vmrk_path = _create_brainvision_files(tmp_dir)
            with open(vhdr_path, "rb") as f:
                vhdr_buf = f.read()
            with open(eeg_path, "rb") as f:
                eeg_buf = f.read()
            with open(vmrk_path, "rb") as f:
                vmrk_buf = f.read()

            # Load succeeds and returns valid data
            raw = load_brainvision_files(vhdr_buf, eeg_buf, vmrk_buf)
            assert raw.n_times > 0
        finally:
            for f in os.listdir(tmp_dir):
                os.unlink(os.path.join(tmp_dir, f))
            os.rmdir(tmp_dir)
