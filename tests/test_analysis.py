"""Tests for utils/analysis.py."""

import numpy as np
import mne
import pytest

from utils.analysis import (
    get_freq_bands,
    compute_psd,
    compute_band_power,
    compute_connectivity,
    compute_erp,
    compute_tfr,
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


class TestComputeERP:
    @staticmethod
    def _make_raw_with_events(n_channels=4, sfreq=256.0, duration=10.0):
        """Create raw data and synthetic events for ERP testing."""
        n_times = int(sfreq * duration)
        rng = np.random.RandomState(42)
        data = rng.randn(n_channels, n_times) * 1e-6
        ch_names = [f"EEG{i+1}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
        # Events at sample 512, 1024, 1536 with event id 1
        events = np.array([[512, 0, 1], [1024, 0, 1], [1536, 0, 1]])
        event_id = {"target": 1}
        return raw, events, event_id

    def test_returns_dict_of_evoked(self):
        raw, events, event_id = self._make_raw_with_events()
        evoked_dict = compute_erp(raw, events, event_id, tmin=-0.1, tmax=0.5)
        assert isinstance(evoked_dict, dict)
        assert "target" in evoked_dict
        assert isinstance(evoked_dict["target"], mne.Evoked)

    def test_evoked_has_correct_channels(self):
        raw, events, event_id = self._make_raw_with_events(n_channels=3)
        evoked_dict = compute_erp(raw, events, event_id)
        assert len(evoked_dict["target"].ch_names) == 3

    def test_evoked_time_range(self):
        raw, events, event_id = self._make_raw_with_events()
        evoked_dict = compute_erp(raw, events, event_id, tmin=-0.2, tmax=0.8)
        evoked = evoked_dict["target"]
        assert evoked.times[0] < 0  # includes pre-stimulus
        assert evoked.times[-1] > 0  # includes post-stimulus


class TestComputeTFR:
    def test_returns_correct_shapes(self):
        raw = _create_sample_raw(n_channels=2, duration=2.0)
        freqs = np.arange(5, 20, 2.0)
        power, times, freqs_out = compute_tfr(raw, freqs=freqs)
        assert power.shape[0] == 2  # n_channels
        assert power.shape[1] == len(freqs)  # n_freqs
        assert len(times) == raw.n_times
        np.testing.assert_array_equal(freqs_out, freqs)

    def test_power_non_negative(self):
        raw = _create_sample_raw(n_channels=2, duration=2.0)
        freqs = np.arange(5, 15, 2.0)
        power, _, _ = compute_tfr(raw, freqs=freqs)
        assert np.all(power >= 0)

    def test_default_freqs(self):
        raw = _create_sample_raw(duration=2.0)
        power, times, freqs = compute_tfr(raw)
        assert len(freqs) == 40  # default 1-40 Hz
        assert freqs[0] == 1.0
        assert freqs[-1] == 40.0
