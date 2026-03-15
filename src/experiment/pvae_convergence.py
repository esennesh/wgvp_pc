import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typing

import src.utils as utils

def create_figure(nrows=1, ncols=1, figsize=None, sharex="none", sharey="none",
                  layout=None, wspace=None, hspace=None, width_ratios=None,
                  height_ratios=None, reshape=False, style="ticks", dpi=None,
                  constrained=True, **kwargs):
    set_style(style=style)
    figsize = figsize or [
        m * d for m, d in zip((ncols, nrows),
                              plt.rcParams.get("figure.figsize"))
    ]
    dpi = dpi if dpi else plt.rcParams.get("figure.dpi")
    layout = "constrained" if constrained else layout
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex,
                             sharey=sharey, layout=layout, figsize=figsize,
                             gridspec_kw={"wspace": wspace, "hspace": hspace,
                                          "width_ratios": width_ratios,
                                          "height_ratios": height_ratios},
                             dpi=dpi, **kwargs)
    if reshape:
        axes = np.array(axes).reshape((nrows, ncols))
    return fig, axes

def _iter_ax(axes):
    if not isinstance(axes, typing.Iterable):
        return [axes]
    elif isinstance(axes, np.ndarray):
        return axes.flat

def make_grid(x, grid_size, scaling=None, pad=1, pad_val=np.nan, normalize=True,
              **kwargs):
    if len(x.shape) == 3:
        x = x[:, np.newaxis]
    assert len(x.shape) == 4
    x = x.transpose(0, 2, 3, 1)
    b, h, w, c = x.shape

    if scaling is None:
        scaling = [1.0] * b
    assert len(scaling) == b

    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    n_rows, n_cols = grid_size

    grid = np.ones(((h + pad) * n_rows - pad, (w + pad) * n_cols - pad, c))
    grid *= pad_val

    for idx in range(min(n_rows * n_cols, b)):
        i = idx // n_cols
        j = idx % n_cols
        a = (h + pad) * i
        b = (w + pad) * j

        y = x[idx]
        if normalize:
            y = normalize_img(y, **kwargs)
        y *= scaling[idx]  # apply manual scaling
        grid[a:a + h, b:b + w] = y

    return grid

def normalize_img(x: np.ndarray, method='min-max', val_range=(0, 1)):
    if method == 'min-max':
        xmin = np.min(x)
        xmax = np.max(x)

        numen = x - xmin
        denum = xmax - xmin
        x_nrm = numen / denum

        a, b = min(val_range), max(val_range)
        x_nrm = x_nrm * (b - a) + a
    elif method == 'abs-max':
        x_nrm = x / np.max(np.abs(x))
    else:
        raise ValueError(method)

    return x_nrm

def plot_convergence(metrics: dict, nrows=2, items=None, interval=None,
                     display=True, **kwargs):
    defaults = {"figsize_x": 5.0, "figsize_y": 3.0, "legend_fontsize": 13,
                "color": "C0", "marker": '.', "markersize": 6, "lw": 1}
    kwargs = {**defaults, **kwargs}
    items = items or ["du_norm", "kl", "nelbo", "r2", "sse", "%-zeros"]
    ncols = int(np.ceil(len(items) / nrows))
    figsize = (kwargs["figsize_x"] * ncols, kwargs["figsize_y"] * nrows)

    fig, axes = create_figure(nrows=nrows, ncols=ncols, figsize=figsize,
                              sharex="col")
    for ax, key in zip(axes.flat, items):
        kws = {"ax": ax, "data": metrics[key], "kwargs": kwargs, "label": key,
               "interval": interval, "xscale": "log"}
        if key in ["r2", "%-zeros"]:
            kws["yscale"] = "linear"
            kws["max_good"] = True
        elif key in ["kl", "nelbo"]:
            kws["yscale"] = "linear"
            kws["max_good"] = False
        else:
            kws["yscale"] = "log"
        subplot(**kws)
    trim_axes(axes, len(items))

    if display:
        plt.show()
    else:
        plt.close()
    return fig, axes

def show_decoder(datamodule, parameters, order=None, method="abs-max",
                 add_title=False, display=True, **kwargs):
    phi = np.array(parameters["decoder$params"]["kernel"].squeeze())
    if order is not None:
        phi = phi[order, :]
    phi = phi.reshape(phi.shape[0], *datamodule.shape[1:])

    pad = 1
    if kwargs.get("pad", None) is None:
        kwargs["pad"] = pad
    if kwargs.get("dpi", None) is None:
        kwargs["dpi"] = 200

    return plot_grid(phi, display=display, method=method,
                     title=None if not add_title else "$\\Phi$", **kwargs)

def plot_grid(imgs, display=True, method="min-max", nrows=None, title=None,
              **kwargs):
    defaults = dict(dpi=160, figsize=(8, 4), title_fontsize=8, title_y=1.01)
    kwargs = {k: kwargs.get(k, defaults.get(k, None)) for k
              in defaults.keys() | kwargs.keys()}
    if nrows is None:
        a = np.log2(len(imgs))
        a = int(np.ceil(a))
        if a % 2 == 1:
            a += 1
        if a <= 6:
            exponent = a // 3 - 1
        elif len(imgs) == 128:
            exponent = 2
        else:
            exponent = a // 2 - 1
        nrows = int(2 ** exponent)
        nrows = max(1, nrows)

    kws_grid = utils.filter_kwargs(make_grid, kwargs)
    ncols = int(np.ceil(len(imgs) / nrows))
    grid = make_grid(imgs, grid_size=(nrows, ncols), method=method,
                     normalize=False if method == 'none' else True, **kws_grid)
    fig, ax = create_figure(figsize=kwargs["figsize"], dpi=kwargs["dpi"],
                            layout="tight")
    cmap = kwargs.get("cmap", "Greys_r")
    if method == "abs-max":
        vmin, vmax = -1, 1
    elif method == "min-max":
        vmin, vmax = 0, 1
    elif method == "none":
        vmin, vmax = None, None
    kws_show = {"cmap": cmap, "vmax": kwargs.get("vmax", vmax),
                "vmin": kwargs.get("vmin", vmin)}
    ax.imshow(grid, **kws_show)
    ax.set_title(fontsize=kwargs.get("title_fontsize", None), label=title,
                 y=kwargs.get("title_y", None))
    remove_ticks(ax)
    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax

def remove_ticks(axes, full=True):
    for ax in _iter_ax(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        if full:
            try:
                map(lambda z: z.set_visible(False), ax.spines.values())
            except AttributeError:
                continue

def set_style(context: str = 'notebook', style: str = 'ticks',
              palette: str = None, font: str = 'sans-serif'):
    sns.set_theme(context=context, style=style, palette=palette, font=font)
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['figure.figsize'] = (3.0, 2.0)
    matplotlib.rcParams['image.interpolation'] = 'none'
    matplotlib.rcParams['font.family'] = font

def subplot(ax, data, label, interval, kwargs, max_good=False, xscale="log",
            yscale="log"):
    data = np.array(data)
    interval = interval or range(len(data))
    interval = range(interval.start, min(interval.stop, len(data)),
                     interval.step)

    y = data[interval]
    if np.isfinite(y).sum() == 0:
        return ax
    best_i = np.nanargmax(y) if max_good else np.nanargmin(y)

    if interval.start == 0:
        xs = [i + 1 for i in interval]
        shifted = True
    else:
        xs = list(interval)
        shifted = False

    ax.plot(xs, y, color=kwargs["color"], label=label, lw=kwargs["lw"],
            marker=kwargs["marker"], markersize=kwargs["markersize"])
    i = best_i + 1 if shifted else best_i
    label = " ".join([f"{'max' if max_good else 'min'}",
                      f"(i = {best_i + 1}): {y[best_i]:0.2f}"])
    ax.axvline(i, color='g', ls='--', label=label)

    fmt = '0.2g' if y[-1] > 1000 else '0.2f'
    label = f"final: {y[-1]:{fmt}}"
    ax.axhline(y[-1], color='r', ls='--', label=label)

    ax.set(xscale=xscale, yscale=yscale)
    ax.legend(fontsize=kwargs['legend_fontsize'])
    ax.grid()
    return ax

def trim_axes(axes, n):
    axes = axes.flat
    for ax in axes[n:]:
        ax.remove()
    return axes[:n]
