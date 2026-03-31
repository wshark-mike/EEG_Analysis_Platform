"""EEG visualization utilities.

Provides helper functions to create various EEG-related plots
using Matplotlib and Plotly for use in Streamlit.
"""

from typing import Optional, Dict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mne

from config import (
    PLOT_MAX_POINTS_PER_TRACE,
    PLOT_SIGNAL_LINEWIDTH,
    PLOT_FIGURE_HEIGHT_PER_CHANNEL,
    PLOT_MIN_FIGURE_HEIGHT,
)
from utils.logger import get_logger
from utils.mne_utils import is_mne_raw, is_mne_ica, get_raw_type_name

logger = get_logger("visualization")


def plot_raw_signals(
    raw: mne.io.BaseRaw,
    duration: float = 10.0,
    n_channels: Optional[int] = None,
    start: float = 0.0,
) -> plt.Figure:
    """Plot raw EEG signals using Matplotlib.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any MNE Raw format).
    duration : float
        Duration of data to plot in seconds. Default is 10.0.
    n_channels : int or None
        Number of channels to plot. If None, plots all.
    start : float
        Start time in seconds. Default is 0.0.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If parameters are invalid.

    Examples
    --------
    >>> fig = plot_raw_signals(raw, duration=5.0, n_channels=10)
    """
    # Type validation - now handles all MNE Raw types
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array"
        )

    # Parameter validation
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")

    try:
        sfreq = raw.info["sfreq"]
        start_sample = int(start * sfreq)
        end_sample = int((start + duration) * sfreq)
        end_sample = min(end_sample, raw.n_times)

        # Validate time range
        if start_sample >= raw.n_times:
            raise ValueError(f"start time exceeds recording duration")

        data = raw.get_data(start=start_sample, stop=end_sample)
        times = np.arange(start_sample, end_sample) / sfreq

        ch_names = raw.ch_names
        if n_channels is not None:
            if not isinstance(n_channels, int) or n_channels <= 0:
                raise ValueError(f"n_channels must be positive int, got {n_channels}")
            data = data[:n_channels]
            ch_names = ch_names[:n_channels]

        n_ch = len(ch_names)
        fig, axes = plt.subplots(n_ch, 1, figsize=(12, max(2 * n_ch, 4)), sharex=True)

        if n_ch == 1:
            axes = [axes]

        for i, (ax, ch_name) in enumerate(zip(axes, ch_names)):
            ax.plot(
                times, data[i] * 1e6, linewidth=PLOT_SIGNAL_LINEWIDTH, color="steelblue"
            )
            ax.set_ylabel(ch_name, fontsize=8, rotation=0, labelpad=40, va="center")
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Raw EEG Signals ({get_raw_type_name(raw)})", fontsize=14)
        fig.tight_layout()

        logger.debug(
            f"Created matplotlib figure for {n_ch} channels ({get_raw_type_name(raw)})"
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to create plot: {str(e)}")
        raise


def plot_raw_signals_plotly(
    raw: mne.io.BaseRaw,
    duration: float = 10.0,
    n_channels: Optional[int] = None,
    start: float = 0.0,
    max_points_per_trace: int = PLOT_MAX_POINTS_PER_TRACE,
) -> go.Figure:
    """Plot raw EEG signals using Plotly (interactive).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any MNE Raw format).
    duration : float
        Duration of data to plot in seconds. Default is 10.0.
    n_channels : int or None
        Number of channels to plot. If None, plots all.
    start : float
        Start time in seconds. Default is 0.0.
    max_points_per_trace : int
        Maximum number of points to display per trace. Default is 2000.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated interactive figure.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If parameters are invalid.

    Examples
    --------
    >>> fig = plot_raw_signals_plotly(raw, duration=5.0, n_channels=10)
    """
    # Type validation - now handles all MNE Raw types
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array"
        )

    # Parameter validation
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if max_points_per_trace <= 0:
        raise ValueError(
            f"max_points_per_trace must be > 0, got {max_points_per_trace}"
        )

    try:
        sfreq = raw.info["sfreq"]
        start_sample = int(start * sfreq)
        end_sample = int((start + duration) * sfreq)
        end_sample = min(end_sample, raw.n_times)

        data = raw.get_data(start=start_sample, stop=end_sample)
        times = np.arange(start_sample, end_sample) / sfreq

        # Down-sample for performance
        n_samples = data.shape[1]
        step = max(1, n_samples // max_points_per_trace)

        data_plot = data[:, ::step]
        times_plot = times[::step]

        ch_names = raw.ch_names
        if n_channels is not None:
            if not isinstance(n_channels, int) or n_channels <= 0:
                raise ValueError(f"n_channels must be positive int, got {n_channels}")
            data_plot = data_plot[:n_channels]
            ch_names = ch_names[:n_channels]

        n_ch = len(ch_names)
        fig = make_subplots(
            rows=n_ch,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=ch_names,
        )

        for i, ch_name in enumerate(ch_names):
            fig.add_trace(
                go.Scatter(
                    x=times_plot,
                    y=data_plot[i] * 1e6,
                    mode="lines",
                    name=ch_name,
                    line=dict(width=0.8),
                ),
                row=i + 1,
                col=1,
            )
            fig.update_yaxes(title_text="µV", row=i + 1, col=1)

        fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)

        title_suffix = f" (downsampled to {len(times_plot)} points)" if step > 1 else ""

        fig.update_layout(
            height=max(PLOT_FIGURE_HEIGHT_PER_CHANNEL * n_ch, PLOT_MIN_FIGURE_HEIGHT),
            title_text=f"Raw EEG Signals - {get_raw_type_name(raw)}{title_suffix}",
            showlegend=False,
        )

        logger.debug(
            f"Created Plotly figure for {n_ch} channels ({get_raw_type_name(raw)})"
        )
        return fig

    except Exception as e:
        logger.error(f"Failed to create Plotly figure: {str(e)}")
        raise


def plot_psd(
    psd_data: np.ndarray, freqs: np.ndarray, ch_names: list, log_scale: bool = True
) -> go.Figure:
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
        Whether to use log scale for y-axis. Default is True.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated figure.

    Raises
    ------
    TypeError
        If parameters are not the correct types.
    ValueError
        If data shapes are invalid.

    Examples
    --------
    >>> fig = plot_psd(psd_data, freqs, ch_names)
    """
    try:
        if not isinstance(psd_data, np.ndarray):
            raise TypeError(f"psd_data must be ndarray, got {type(psd_data).__name__}")

        if psd_data.shape[1] != len(freqs):
            raise ValueError(
                f"psd_data has {psd_data.shape[1]} frequencies, "
                f"but freqs has {len(freqs)}"
            )

        if psd_data.shape[0] != len(ch_names):
            raise ValueError(
                f"psd_data has {psd_data.shape[0]} channels, "
                f"but ch_names has {len(ch_names)}"
            )

        fig = go.Figure()

        for i, ch_name in enumerate(ch_names):
            y_values = 10 * np.log10(psd_data[i]) if log_scale else psd_data[i]
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=y_values,
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

        logger.debug(f"Created PSD plot for {len(ch_names)} channels")
        return fig

    except Exception as e:
        logger.error(f"Failed to create PSD plot: {str(e)}")
        raise


def plot_band_power(band_powers: Dict[str, np.ndarray], ch_names: list) -> go.Figure:
    """Plot band power as a grouped bar chart.

    Parameters
    ----------
    band_powers : dict
        Dictionary mapping band names to power arrays per channel.
    ch_names : list
        Channel names.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated figure.

    Examples
    --------
    >>> band_powers = {"Alpha": [...], "Beta": [...]}
    >>> fig = plot_band_power(band_powers, ch_names)
    """
    try:
        if not isinstance(band_powers, dict):
            raise TypeError(
                f"band_powers must be dict, got {type(band_powers).__name__}"
            )

        fig = go.Figure()

        for band_name, powers in band_powers.items():
            if len(powers) != len(ch_names):
                raise ValueError(
                    f"Band '{band_name}' has {len(powers)} values, "
                    f"but ch_names has {len(ch_names)}"
                )

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

        logger.debug(f"Created band power plot for {len(band_powers)} bands")
        return fig

    except Exception as e:
        logger.error(f"Failed to create band power plot: {str(e)}")
        raise


def plot_connectivity_matrix(conn_matrix: np.ndarray, ch_names: list) -> go.Figure:
    """Plot connectivity matrix as a heatmap.

    Parameters
    ----------
    conn_matrix : np.ndarray
        Connectivity matrix, shape (n_channels, n_channels).
    ch_names : list
        Channel names.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated figure.

    Examples
    --------
    >>> fig = plot_connectivity_matrix(conn_matrix, ch_names)
    """
    try:
        if not isinstance(conn_matrix, np.ndarray):
            raise TypeError(
                f"conn_matrix must be ndarray, got {type(conn_matrix).__name__}"
            )

        if conn_matrix.shape[0] != conn_matrix.shape[1]:
            raise ValueError(f"conn_matrix must be square, got {conn_matrix.shape}")

        if conn_matrix.shape[0] != len(ch_names):
            raise ValueError(
                f"conn_matrix size {conn_matrix.shape[0]} != "
                f"n_channels {len(ch_names)}"
            )

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

        logger.debug("Created connectivity matrix heatmap")
        return fig

    except Exception as e:
        logger.error(f"Failed to create connectivity matrix: {str(e)}")
        raise


def plot_ica_components(
    ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw, n_components: Optional[int] = None
) -> plt.Figure:
    """Plot ICA component time courses.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.BaseRaw
        The raw EEG data (any MNE Raw format).
    n_components : int or None
        Number of components to plot. If None, plots all.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    Raises
    ------
    TypeError
        If parameters are not the correct types.
    ValueError
        If n_components is invalid.

    Examples
    --------
    >>> fig = plot_ica_components(ica, raw, n_components=10)
    """
    try:
        if not is_mne_ica(ica):
            raise TypeError(f"ica must be ICA object, got {type(ica).__name__}")

        if not is_mne_raw(raw):
            raw_type = type(raw).__name__
            raise TypeError(f"raw must be MNE Raw object, got {raw_type}")

        sources = ica.get_sources(raw, verbose=False)
        data = sources.get_data()
        times = sources.times

        if n_components is not None:
            if not isinstance(n_components, int) or n_components <= 0:
                raise ValueError(
                    f"n_components must be positive int, got {n_components}"
                )
            if n_components > data.shape[0]:
                raise ValueError(
                    f"n_components {n_components} exceeds "
                    f"number of ICA components {data.shape[0]}"
                )
            data = data[:n_components]

        n_comp = data.shape[0]
        fig, axes = plt.subplots(
            n_comp, 1, figsize=(12, max(1.5 * n_comp, 4)), sharex=True
        )

        if n_comp == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(times, data[i], linewidth=PLOT_SIGNAL_LINEWIDTH, color="darkblue")
            ax.set_ylabel(f"IC{i}", fontsize=8, rotation=0, labelpad=30, va="center")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("ICA Components", fontsize=14)
        fig.tight_layout()

        logger.debug(f"Created ICA components plot for {n_comp} components")
        return fig

    except Exception as e:
        logger.error(f"Failed to create ICA plot: {str(e)}")
        raise
