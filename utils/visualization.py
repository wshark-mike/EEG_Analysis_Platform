"""EEG visualization utilities.

Provides helper functions to create various EEG-related plots
using Matplotlib and Plotly for use in Streamlit.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_raw_signals(raw, duration=10.0, n_channels=None, start=0.0):
    """Plot raw EEG signals using Matplotlib.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    duration : float
        Duration of data to plot in seconds. Must be positive.
    n_channels : int or None
        Number of channels to plot. If None, plots all.
    start : float
        Start time in seconds. Must be non-negative.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If duration is not positive or start is negative.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")

    sfreq = raw.info["sfreq"]
    start_sample = int(start * sfreq)
    end_sample = int((start + duration) * sfreq)
    end_sample = min(end_sample, raw.n_times)

    data = raw.get_data(start=start_sample, stop=end_sample)
    times = np.arange(start_sample, end_sample) / sfreq

    ch_names = raw.ch_names
    if n_channels is not None:
        data = data[:n_channels]
        ch_names = ch_names[:n_channels]

    n_ch = len(ch_names)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, max(2 * n_ch, 4)), sharex=True)

    if n_ch == 1:
        axes = [axes]

    for i, (ax, ch_name) in enumerate(zip(axes, ch_names)):
        ax.plot(times, data[i] * 1e6, linewidth=0.5, color="steelblue")
        ax.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=40, va="center")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw EEG Signals", fontsize=14)
    fig.tight_layout()
    return fig


def plot_raw_signals_plotly(raw, duration=10.0, n_channels=None, start=0.0):
    """Plot raw EEG signals using Plotly (interactive).

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    duration : float
        Duration of data to plot in seconds. Must be positive.
    n_channels : int or None
        Number of channels to plot. If None, plots all.
    start : float
        Start time in seconds. Must be non-negative.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated interactive figure.

    Raises
    ------
    ValueError
        If duration is not positive or start is negative.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")

    sfreq = raw.info["sfreq"]
    start_sample = int(start * sfreq)
    end_sample = int((start + duration) * sfreq)
    end_sample = min(end_sample, raw.n_times)

    data = raw.get_data(start=start_sample, stop=end_sample)
    times = np.arange(start_sample, end_sample) / sfreq

    ch_names = raw.ch_names
    if n_channels is not None:
        data = data[:n_channels]
        ch_names = ch_names[:n_channels]

    n_ch = len(ch_names)
    fig = make_subplots(
        rows=n_ch, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=ch_names,
    )

    for i, ch_name in enumerate(ch_names):
        fig.add_trace(
            go.Scatter(
                x=times, y=data[i] * 1e6,
                mode="lines",
                name=ch_name,
                line=dict(width=0.8),
            ),
            row=i + 1, col=1,
        )
        fig.update_yaxes(title_text="µV", row=i + 1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)
    fig.update_layout(
        height=max(200 * n_ch, 400),
        title_text="Raw EEG Signals",
        showlegend=False,
    )
    return fig


def plot_psd(psd_data, freqs, ch_names, log_scale=True):
    """Plot Power Spectral Density.

    Parameters
    ----------
    psd_data : np.ndarray
        PSD values, shape (n_channels, n_freqs).
    freqs : np.ndarray
        Frequency values in Hz.
    ch_names : list
        Channel names.
    log_scale : bool
        Whether to use log scale for y-axis.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If psd_data and ch_names have mismatched dimensions.
    """
    if len(psd_data) != len(ch_names):
        raise ValueError(
            f"psd_data has {len(psd_data)} channels but ch_names has {len(ch_names)}"
        )
    fig = go.Figure()

    for i, ch_name in enumerate(ch_names):
        y_values = 10 * np.log10(psd_data[i]) if log_scale else psd_data[i]
        fig.add_trace(
            go.Scatter(
                x=freqs, y=y_values,
                mode="lines",
                name=ch_name,
            )
        )

    y_label = "Power (dB)" if log_scale else "Power (V²/Hz)"
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_label,
        height=500,
    )
    return fig


def plot_band_power(band_powers, ch_names):
    """Plot band power as a grouped bar chart.

    Parameters
    ----------
    band_powers : dict
        Dictionary mapping band names to power arrays per channel.
    ch_names : list
        Channel names.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure.
    """
    fig = go.Figure()

    for band_name, powers in band_powers.items():
        fig.add_trace(
            go.Bar(
                x=ch_names,
                y=powers,
                name=band_name,
            )
        )

    fig.update_layout(
        title="Band Power by Channel",
        xaxis_title="Channel",
        yaxis_title="Power",
        barmode="group",
        height=500,
    )
    return fig


def plot_connectivity_matrix(conn_matrix, ch_names):
    """Plot connectivity matrix as a heatmap.

    Parameters
    ----------
    conn_matrix : np.ndarray
        Connectivity matrix, shape (n_channels, n_channels).
    ch_names : list
        Channel names.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The generated figure.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=conn_matrix,
            x=ch_names,
            y=ch_names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
    )

    fig.update_layout(
        title="Channel Connectivity Matrix",
        height=600,
        width=600,
    )
    return fig


def plot_ica_components(ica, raw, n_components=None):
    """Plot ICA component time courses.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        The raw EEG data.
    n_components : int or None
        Number of components to plot. If None, plots all.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    sources = ica.get_sources(raw)
    data = sources.get_data()
    times = sources.times

    if n_components is not None:
        data = data[:n_components]

    n_comp = data.shape[0]
    fig, axes = plt.subplots(n_comp, 1, figsize=(12, max(1.5 * n_comp, 4)), sharex=True)

    if n_comp == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(times, data[i], linewidth=0.5, color="darkblue")
        ax.set_ylabel(f"IC{i}", fontsize=8, rotation=0, labelpad=30, va="center")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("ICA Components", fontsize=14)
    fig.tight_layout()
    return fig
