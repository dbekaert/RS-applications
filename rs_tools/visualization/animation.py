"""Animated GIF export for time-series imagery.

Renders a sequence of RGB frames as an animated GIF with per-frame
annotations (sensor, orbit direction, UTC time) and an optional
scale bar.

Supports two modes:

* **Pre-built frames** — pass a list of RGB arrays to
  ``save_timeseries_gif()``.
* **Lazy / incremental** — pass an iterable of data items and a
  compositing callable to ``save_timeseries_gif_lazy()`` so that only
  one frame is in memory at a time.

After writing, ``compress_gif()`` is called automatically when
``gifsicle`` is available on the system, reducing file size
significantly.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rs_tools.visualization.scalebar import add_scalebar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GIF compression
# ---------------------------------------------------------------------------

def compress_gif(
    gif_path: Union[str, Path],
    lossy: int = 30,
    colors: int = 256,
) -> None:
    """Compress a GIF in-place using ``gifsicle`` if available.

    Parameters
    ----------
    gif_path : str or Path
        Path to the GIF file.
    lossy : int
        Lossy compression level (0 = lossless, higher = smaller).
    colors : int
        Maximum number of colours in the palette (max 256).
    """
    if shutil.which("gifsicle") is None:
        logger.debug("gifsicle not found — skipping compression")
        return

    gif_path = str(gif_path)
    size_before = os.path.getsize(gif_path)
    try:
        subprocess.run(
            [
                "gifsicle",
                f"--lossy={lossy}",
                "-O3",
                "--colors",
                str(min(colors, 256)),
                gif_path,
                "-o",
                gif_path,
            ],
            check=True,
            capture_output=True,
        )
        size_after = os.path.getsize(gif_path)
        ratio = (1 - size_after / size_before) * 100 if size_before else 0
        print(
            f"  Compressed with gifsicle: "
            f"{size_before / 1024**2:.1f} → {size_after / 1024**2:.1f} MB "
            f"({ratio:.0f}% smaller)"
        )
    except subprocess.CalledProcessError:
        logger.debug("gifsicle compression failed — keeping original")


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_frame(
    rgb: np.ndarray,
    label: Optional[str],
    title: Optional[str],
    scalebar_km: Optional[float],
    pixel_size_m: Optional[float],
    figsize: Tuple[int, int],
    dpi: int,
) -> Image.Image:
    """Render a single annotated frame as a PIL Image."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if label:
        ax.text(
            0.02,
            0.02,
            label,
            transform=ax.transAxes,
            fontsize=11,
            color="white",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="black",
                alpha=0.7,
            ),
            verticalalignment="bottom",
        )

    if pixel_size_m is not None:
        add_scalebar(ax, pixel_size_m, length_km=scalebar_km)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    del fig, ax
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # .convert("RGB") copies pixels so buf can be freed
    buf.close()
    return img


# ---------------------------------------------------------------------------
# Pre-built frame list (original API, kept for backwards compatibility)
# ---------------------------------------------------------------------------

def save_timeseries_gif(
    frames: Sequence[np.ndarray],
    output_path: Union[str, Path],
    labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    scalebar_km: Optional[float] = None,
    pixel_size_m: Optional[float] = None,
    fps: float = 2.0,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100,
    lossy: int = 30,
) -> Path:
    """Save a list of RGB frames as an animated GIF.

    Parameters
    ----------
    frames : sequence of np.ndarray
        List of ``(H, W, 3)`` RGB arrays with values in [0, 1].
    output_path : str or Path
        Destination path for the ``.gif`` file.
    labels : sequence of str, optional
        Per-frame annotation strings (e.g. sensor / orbit / datetime).
        Must have the same length as *frames* if provided.
    title : str, optional
        Title shown at the top of each frame.
    scalebar_km : float, optional
        Fixed scale-bar length in km.  If *None* and *pixel_size_m* is
        given, the length is auto-determined.
    pixel_size_m : float, optional
        Ground distance per pixel in metres (needed for the scale bar).
    fps : float
        Frames per second.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution for rendering.
    lossy : int
        Lossy compression level for ``gifsicle`` (0 = lossless).

    Returns
    -------
    pathlib.Path
        Path to the written GIF file.
    """
    output_path = Path(output_path)
    duration_ms = int(1000.0 / fps)

    pil_frames: List[Image.Image] = []
    for i, rgb in enumerate(frames):
        label = labels[i] if labels else None
        img = _render_frame(
            rgb,
            label=label,
            title=title,
            scalebar_km=scalebar_km,
            pixel_size_m=pixel_size_m,
            figsize=figsize,
            dpi=dpi,
        )
        pil_frames.append(img)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {len(pil_frames)}-frame GIF → {output_path} ({size_mb:.1f} MB)")
    compress_gif(output_path, lossy=lossy)
    return output_path


# ---------------------------------------------------------------------------
# Lazy / incremental GIF writing (one frame at a time)
# ---------------------------------------------------------------------------

def _downsample_rgb(rgb: np.ndarray, max_pixels: int) -> np.ndarray:
    """Downsample an (H, W, 3) array so max(H, W) <= *max_pixels*."""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_pixels:
        return rgb
    scale = max_pixels / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    # Use PIL for fast high-quality downsampling
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_timeseries_gif_lazy(
    items: Iterable,
    output_path: Union[str, Path],
    composite_fn: Callable,
    title: Optional[str] = None,
    scalebar_km: Optional[float] = None,
    pixel_size_m: Optional[float] = None,
    fps: float = 2.0,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100,
    lossy: int = 30,
    max_pixels: int = 1200,
) -> Path:
    """Render and save GIF frames one at a time to minimise peak memory.

    Instead of pre-building all RGB composites, this function accepts
    an iterable of data items and a callable that produces an
    ``(rgb, label)`` tuple from each item.  Only one frame is ever
    materialised at a time.

    Parameters
    ----------
    items : iterable
        Data items to iterate over (e.g. ``LoadedItem`` objects).
    output_path : str or Path
        Destination path for the ``.gif`` file.
    composite_fn : callable
        ``composite_fn(item) -> (rgb, label)`` where *rgb* is an
        ``(H, W, 3)`` float array and *label* is a string.
    title : str, optional
        Title shown at the top of each frame.
    scalebar_km : float, optional
        Fixed scale-bar length in km.
    pixel_size_m : float, optional
        Ground distance per pixel in metres.
    fps : float
        Frames per second.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution for rendering.
    lossy : int
        Lossy compression level for ``gifsicle`` (0 = lossless).
    max_pixels : int
        Maximum dimension (height or width) of the composited frame
        in pixels.  Larger composites are downsampled before rendering
        to keep memory usage bounded.  Set to 0 to disable.

    Returns
    -------
    pathlib.Path
        Path to the written GIF file.
    """
    output_path = Path(output_path)
    duration_ms = int(1000.0 / fps)

    # Write each rendered frame to a temp PNG on disk so that at most
    # one composite + one PIL image is in memory at any time.
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="gif_frames_")
    frame_paths: List[str] = []
    n_frames = 0

    try:
        for item in items:
            rgb, label = composite_fn(item)
            if max_pixels and max(rgb.shape[:2]) > max_pixels:
                rgb = _downsample_rgb(rgb, max_pixels)
            img = _render_frame(
                rgb,
                label=label,
                title=title,
                scalebar_km=scalebar_km,
                pixel_size_m=pixel_size_m,
                figsize=figsize,
                dpi=dpi,
            )
            del rgb  # free composite immediately

            frame_path = os.path.join(tmp_dir, f"frame_{n_frames:04d}.png")
            img.save(frame_path)
            del img
            frame_paths.append(frame_path)
            n_frames += 1

            # Matplotlib leaks memory across many fig create/close
            # cycles; periodic gc keeps the footprint bounded.
            if n_frames % 5 == 0:
                gc.collect()

        if n_frames == 0:
            raise ValueError("No frames produced from items iterable")

        # Assemble GIF from disk PNGs.
        # PIL's save_all with a generator can buffer internally, so
        # we quantize + write each frame as a single-frame GIF first,
        # then use a streaming concat to avoid loading all frames at
        # once.  When gifsicle is available it does this natively;
        # otherwise we fall back to PIL's save_all with the generator.
        _gifsicle = shutil.which("gifsicle")

        if _gifsicle:
            # Convert PNGs → individual GIFs, merge with gifsicle
            gif_singles: List[str] = []
            for fp in frame_paths:
                gp = fp.replace(".png", ".gif")
                Image.open(fp).convert("RGB").save(gp)
                gif_singles.append(gp)

            subprocess.run(
                [
                    _gifsicle,
                    "--merge",
                    f"--delay={duration_ms // 10}",
                    "--loop",
                    *gif_singles,
                    "-o",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
            )
        else:
            # Fallback: PIL generator approach — still better than
            # loading all into a list since PIL processes one at a time.
            first_frame = Image.open(frame_paths[0]).convert("RGB")

            def _load_remaining():
                for fp in frame_paths[1:]:
                    frm = Image.open(fp).convert("RGB")
                    yield frm

            first_frame.save(
                output_path,
                save_all=True,
                append_images=_load_remaining(),
                duration=duration_ms,
                loop=0,
            )
            del first_frame

    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved {n_frames}-frame GIF → {output_path} ({size_mb:.1f} MB)")
    compress_gif(output_path, lossy=lossy)
    return output_path
