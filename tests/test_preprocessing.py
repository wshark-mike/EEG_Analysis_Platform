"""Tests for utils/preprocessing.py."""

import numpy as np
import mne
import pytest

from utils.preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_rereferencing,
    run_ica,
    apply_ica_exclusion,
    detect_bad_channels,
)


def _create_sample_raw(n_channels=8, sfreq=256.0, duration=4.0):
    """Create a synthetic MNE Raw object for testing."""
    n_times = int(sfreq * duration)
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_times) * 1e-6
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


class TestBandpassFilter:
    def test_returns_raw(self):
        raw = _create_sample_raw()
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=30.0)
        assert isinstance(filtered, mne.io.BaseRaw)

    def test_does_not_modify_original(self):
        raw = _create_sample_raw()
        original_data = raw.get_data().copy()
        apply_bandpass_filter(raw, l_freq=1.0, h_freq=30.0)
        np.testing.assert_array_equal(raw.get_data(), original_data)

    def test_preserves_shape(self):
        raw = _create_sample_raw()
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=30.0)
        assert filtered.get_data().shape == raw.get_data().shape


class TestNotchFilter:
    def test_returns_raw(self):
        raw = _create_sample_raw()
        notched = apply_notch_filter(raw, freqs=50.0)
        assert isinstance(notched, mne.io.BaseRaw)

    def test_accepts_list_of_freqs(self):
        raw = _create_sample_raw()
        notched = apply_notch_filter(raw, freqs=[50.0, 100.0])
        assert isinstance(notched, mne.io.BaseRaw)

    def test_does_not_modify_original(self):
        raw = _create_sample_raw()
        original_data = raw.get_data().copy()
        apply_notch_filter(raw, freqs=50.0)
        np.testing.assert_array_equal(raw.get_data(), original_data)


class TestRereferencing:
    def test_average_reference(self):
        raw = _create_sample_raw()
        reref = apply_rereferencing(raw, ref_type="average")
        assert isinstance(reref, mne.io.BaseRaw)

    def test_channel_reference(self):
        raw = _create_sample_raw()
        reref = apply_rereferencing(raw, ref_type="EEG1")
        assert isinstance(reref, mne.io.BaseRaw)

    def test_does_not_modify_original(self):
        raw = _create_sample_raw()
        original_data = raw.get_data().copy()
        apply_rereferencing(raw, ref_type="average")
        np.testing.assert_array_equal(raw.get_data(), original_data)


class TestICA:
    def test_run_ica_returns_ica(self):
        raw = _create_sample_raw(duration=5.0)
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        ica = run_ica(filtered, n_components=4)
        assert isinstance(ica, mne.preprocessing.ICA)

    def test_ica_exclusion(self):
        raw = _create_sample_raw(duration=5.0)
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        ica = run_ica(filtered, n_components=4)
        cleaned = apply_ica_exclusion(filtered, ica, exclude_idx=[0])
        assert isinstance(cleaned, mne.io.BaseRaw)
        assert cleaned.get_data().shape == filtered.get_data().shape


class TestBadChannelDetection:
    def test_no_bad_channels(self):
        raw = _create_sample_raw()
        bad = detect_bad_channels(raw, threshold=5.0)
        assert isinstance(bad, list)

    def test_detects_bad_channel(self):
        """Create data where one channel is clearly an outlier."""
        raw = _create_sample_raw(n_channels=8)
        data = raw.get_data()
        # Make channel 0 have much higher variance
        data[0] *= 100
        raw_modified = mne.io.RawArray(data, raw.info)
        bad = detect_bad_channels(raw_modified, threshold=2.0)
        assert "EEG1" in bad

    def test_unsupported_method(self):
        raw = _create_sample_raw()
        with pytest.raises(ValueError, match="Unsupported method"):
            detect_bad_channels(raw, method="unknown")