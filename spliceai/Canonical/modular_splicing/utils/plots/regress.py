import numpy as np


def regress(x_label, x, y, ax, *, scatter_kwargs):
    """
    Plot a scatter plot of x vs y, and then fit a line to the data and plot it.

    Ignores any data point where x or y is NaN.

    Parameters
    ----------
    x_label : str
        The label for the x-axis.
    x : np.ndarray
        The x-values.
    y : np.ndarray
        The y-values.
    ax : matplotlib.Axis
        The axis to plot on.
    scatter_kwargs : dict
        Keyword arguments to pass to ax.scatter.

    Returns
    -------
    m, r2: float
        The slope and R^2 of the regression line.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    (m, b), *_ = np.linalg.lstsq(np.array([x, np.ones_like(x)]).T, y, rcond=None)
    min_x, max_x = x.min(), x.max()
    rang = max_x - min_x
    min_x -= rang / 10
    max_x += rang / 10
    ax.scatter(x, y, **scatter_kwargs)
    r = np.linspace(min_x, max_x, 2)

    r2 = np.corrcoef(x, y)[0, 1] ** 2
    ax.plot(
        r,
        r * m + b,
        label=f"Best fit: y = {m:.2f}x + {b:.2f} [R^2={r2:.2%}]",
        color="black",
    )
    ax.set_xlim(min_x, max_x)
    ax.legend()
    ax.set_xlabel(x_label)
    return dict(m=m, r2=r2)
