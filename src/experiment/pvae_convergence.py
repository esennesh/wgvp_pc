import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
