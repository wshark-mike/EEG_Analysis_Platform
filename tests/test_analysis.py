"""Tests for utils/analysis.py."""

import numpy as np
import mne
import pytest

from utils.analysis import (
    get_freq_bands,
    compute_psd,
    compute_band_power,
    compute_connectivity,
)


def _create_sample_raw(n_channels=4, sfreq=256.0, duration=4.0):
    """Create a synthetic MNE Raw object for testing."""
    n_times = int(sfreq * duration)
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_times) * 1e-6
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


class TestFreqBands:
    def test_returns_dict(self):
        bands = get_freq_bands()
        assert isinstance(bands, dict)

    def test_standard_bands_present(self):
        bands = get_freq_bands()
        for name in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            assert name in bands

    def test_band_values_are_tuples(self):
        bands = get_freq_bands()
        for name, val in bands.items():
            assert isinstance(val, tuple)
            assert len(val) == 2
            assert val[0] < val[1]


class TestComputePSD:
    def test_returns_correct_shapes(self):
        raw = _create_sample_raw()
        psd_data, freqs = compute_psd(raw, fmin=1.0, fmax=40.0)

        assert psd_data.ndim == 2
        assert psd_data.shape[0] == 4  # n_channels
        assert len(freqs) == psd_data.shape[1]

    def test_freqs_in_range(self):
        raw = _create_sample_raw()
        _, freqs = compute_psd(raw, fmin=5.0, fmax=30.0)
        assert freqs[0] >= 5.0
        assert freqs[-1] <= 30.0

    def test_psd_positive(self):
        raw = _create_sample_raw()
        psd_data, _ = compute_psd(raw)
        assert np.all(psd_data >= 0)


class TestComputeBandPower:
    def test_returns_all_bands(self):
        raw = _create_sample_raw()
        band_powers = compute_band_power(raw)

        bands = get_freq_bands()
        for band_name in bands:
            assert band_name in band_powers

    def test_power_per_channel(self):
        raw = _create_sample_raw(n_channels=4)
        band_powers = compute_band_power(raw)

        for powers in band_powers.values():
            assert len(powers) == 4

    def test_custom_bands(self):
        raw = _create_sample_raw()
        custom_bands = {"MyBand": (8.0, 12.0)}
        band_powers = compute_band_power(raw, bands=custom_bands)
        assert "MyBand" in band_powers


class TestComputeConnectivity:
    def test_correlation_shape(self):
        raw = _create_sample_raw(n_channels=4)
        conn, ch_names = compute_connectivity(raw, method="correlation")

        assert conn.shape == (4, 4)
        assert len(ch_names) == 4

    def test_correlation_diagonal_is_one(self):
        raw = _create_sample_raw()
        conn, _ = compute_connectivity(raw, method="correlation")
        np.testing.assert_array_almost_equal(np.diag(conn), 1.0)

    def test_correlation_symmetric(self):
        raw = _create_sample_raw()
        conn, _ = compute_connectivity(raw, method="correlation")
        np.testing.assert_array_almost_equal(conn, conn.T)

    def test_unsupported_method(self):
        raw = _create_sample_raw()
        with pytest.raises(ValueError, match="Unsupported method"):
            compute_connectivity(raw, method="unknown")
