"""Tests for utils/visualization.py."""

import numpy as np
import mne
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils.visualization import (
    plot_raw_signals,
    plot_raw_signals_plotly,
    plot_psd,
    plot_band_power,
    plot_connectivity_matrix,
    plot_ica_components,
)
from utils.preprocessing import run_ica, apply_bandpass_filter


def _create_sample_raw(n_channels=4, sfreq=256.0, duration=2.0):
    """Create a synthetic MNE Raw object for testing."""
    n_times = int(sfreq * duration)
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_times) * 1e-6
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


class TestPlotRawSignals:
    def test_returns_matplotlib_figure(self):
        raw = _create_sample_raw()
        fig = plot_raw_signals(raw, duration=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plots_all_channels(self):
        raw = _create_sample_raw(n_channels=4)
        fig = plot_raw_signals(raw, duration=1.0)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_limits_channels(self):
        raw = _create_sample_raw(n_channels=4)
        fig = plot_raw_signals(raw, duration=1.0, n_channels=2)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_single_channel(self):
        raw = _create_sample_raw(n_channels=1)
        fig = plot_raw_signals(raw, duration=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_start_offset(self):
        raw = _create_sample_raw(duration=4.0)
        fig = plot_raw_signals(raw, duration=1.0, start=2.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_duration_exceeds_data_length(self):
        raw = _create_sample_raw(duration=1.0)
        fig = plot_raw_signals(raw, duration=10.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRawSignalsPlotly:
    def test_returns_plotly_figure(self):
        raw = _create_sample_raw()
        fig = plot_raw_signals_plotly(raw, duration=1.0)
        assert isinstance(fig, go.Figure)

    def test_plots_all_channels(self):
        raw = _create_sample_raw(n_channels=3)
        fig = plot_raw_signals_plotly(raw, duration=1.0)
        assert len(fig.data) == 3

    def test_limits_channels(self):
        raw = _create_sample_raw(n_channels=4)
        fig = plot_raw_signals_plotly(raw, duration=1.0, n_channels=2)
        assert len(fig.data) == 2

    def test_single_channel(self):
        raw = _create_sample_raw(n_channels=1)
        fig = plot_raw_signals_plotly(raw, duration=1.0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_start_offset(self):
        raw = _create_sample_raw(duration=4.0)
        fig = plot_raw_signals_plotly(raw, duration=1.0, start=2.0)
        assert isinstance(fig, go.Figure)


class TestPlotPSD:
    def test_returns_plotly_figure(self):
        freqs = np.linspace(1, 50, 100)
        psd_data = np.random.rand(4, 100)
        ch_names = ["EEG1", "EEG2", "EEG3", "EEG4"]
        fig = plot_psd(psd_data, freqs, ch_names)
        assert isinstance(fig, go.Figure)

    def test_trace_count_matches_channels(self):
        freqs = np.linspace(1, 50, 100)
        psd_data = np.random.rand(3, 100)
        ch_names = ["A", "B", "C"]
        fig = plot_psd(psd_data, freqs, ch_names)
        assert len(fig.data) == 3

    def test_log_scale_mode(self):
        freqs = np.linspace(1, 50, 50)
        psd_data = np.abs(np.random.rand(2, 50)) + 1e-10
        ch_names = ["Ch1", "Ch2"]
        fig = plot_psd(psd_data, freqs, ch_names, log_scale=True)
        assert isinstance(fig, go.Figure)

    def test_linear_scale_mode(self):
        freqs = np.linspace(1, 50, 50)
        psd_data = np.random.rand(2, 50)
        ch_names = ["Ch1", "Ch2"]
        fig = plot_psd(psd_data, freqs, ch_names, log_scale=False)
        assert isinstance(fig, go.Figure)

    def test_single_channel(self):
        freqs = np.linspace(1, 50, 50)
        psd_data = np.random.rand(1, 50)
        ch_names = ["Solo"]
        fig = plot_psd(psd_data, freqs, ch_names)
        assert len(fig.data) == 1


class TestPlotBandPower:
    def test_returns_plotly_figure(self):
        band_powers = {
            "Delta": np.array([1.0, 2.0]),
            "Alpha": np.array([3.0, 4.0]),
        }
        ch_names = ["Ch1", "Ch2"]
        fig = plot_band_power(band_powers, ch_names)
        assert isinstance(fig, go.Figure)

    def test_trace_count_matches_bands(self):
        band_powers = {
            "Delta": np.array([1.0]),
            "Theta": np.array([2.0]),
            "Alpha": np.array([3.0]),
        }
        ch_names = ["Ch1"]
        fig = plot_band_power(band_powers, ch_names)
        assert len(fig.data) == 3

    def test_empty_band_powers(self):
        fig = plot_band_power({}, ["Ch1"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestPlotConnectivityMatrix:
    def test_returns_plotly_figure(self):
        conn = np.eye(3)
        ch_names = ["A", "B", "C"]
        fig = plot_connectivity_matrix(conn, ch_names)
        assert isinstance(fig, go.Figure)

    def test_heatmap_data_shape(self):
        conn = np.corrcoef(np.random.rand(4, 100))
        ch_names = ["A", "B", "C", "D"]
        fig = plot_connectivity_matrix(conn, ch_names)
        assert fig.data[0].z.shape == (4, 4)

    def test_colorscale_limits(self):
        conn = np.eye(2)
        ch_names = ["X", "Y"]
        fig = plot_connectivity_matrix(conn, ch_names)
        assert fig.data[0].zmin == -1
        assert fig.data[0].zmax == 1


class TestPlotICAComponents:
    def test_returns_matplotlib_figure(self):
        raw = _create_sample_raw(n_channels=8, duration=5.0)
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        ica = run_ica(filtered, n_components=4)
        fig = plot_ica_components(ica, filtered)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_limits_components(self):
        raw = _create_sample_raw(n_channels=8, duration=5.0)
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        ica = run_ica(filtered, n_components=4)
        fig = plot_ica_components(ica, filtered, n_components=2)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_single_component(self):
        raw = _create_sample_raw(n_channels=8, duration=5.0)
        filtered = apply_bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        ica = run_ica(filtered, n_components=4)
        fig = plot_ica_components(ica, filtered, n_components=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
