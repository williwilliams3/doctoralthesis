import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("figs/extra")
FIGSIZE = (2.6, 1.7)
DPI = 400
FILL_LEVELS = np.array([-12.0, -8.5, -6.0, -4.0, -2.75, -1.75, -1.0, -0.45, 1e-6])
FUNNEL_FILL_LEVELS = np.array([-4.2, -3.5, -2.9, -2.3, -1.8, -1.35, -1.0, -0.65, 1e-6])
FILL_COLORS = [
    "#f9f9f9",
    "#ededed",
    "#dfdfdf",
    "#cdcdcd",
    "#b5b5b5",
    "#979797",
    "#727272",
    "#424242",
]
LINE_LEVELS = FILL_LEVELS[1:-1]
FUNNEL_LINE_LEVELS = FUNNEL_FILL_LEVELS[1:-1]
AXIS_PADDING = 0.04


class Distribution:

    def __init__(self, dim=2):
        self.dim = dim
        self.name = "Distribution"
        self.fill_levels = FILL_LEVELS
        self.line_levels = LINE_LEVELS
        self.grid_shape = (360, 240)
        self.line_ymin = None

    def logdensity_fn(self, x):
        raise NotImplementedError


class Funnel(Distribution):

    def __init__(self, dim=2):
        super().__init__(dim=dim)
        self.dim = dim
        self.name = "Funnel"
        self.mean = 0.0
        self.sigma = 3.0
        self.xlim = [-7.0, 7.0]
        self.ylim = [-10.0, 4.0]
        self.true_dist_levels = [-10.0, -5.0, -2.0]
        self.fill_levels = FUNNEL_FILL_LEVELS
        self.line_levels = FUNNEL_LINE_LEVELS
        self.grid_shape = (720, 520)
        self.line_ymin = -7.6

    def logdensity_fn(self, x):
        mean = self.mean
        sigma = self.sigma
        dim = self.dim
        return jss.norm.logpdf(x[dim - 1], loc=0.0, scale=sigma) + jnp.sum(
            jss.norm.logpdf(x[: dim - 1], loc=mean, scale=jnp.exp(0.5 * x[dim - 1]))
        )


class Rosenbrock(Distribution):

    def __init__(self, dim=2):
        super().__init__(dim=dim)
        self.dim = dim
        self.name = "Rosenbrock"
        self.a = 1.0
        self.b = 20.0
        self.dim = dim
        self.xlim = [-2, 4.0]
        self.ylim = [-1, 16]
        self.true_dist_levels = [-5.0, -3.0, -1.5]

    def logdensity_fn(self, theta):
        b = self.b
        a = self.a
        theta = jnp.array(theta)
        # First term
        logpdf = -((theta[0] - a) ** 2)
        # Terms dependent on previous term
        logpdf -= b * jnp.sum((theta[1:] - theta[:-1] ** 2) ** 2)
        return logpdf


class Squiggle(Distribution):

    def __init__(self, dim=2):
        super().__init__(dim=dim)
        self.dim = dim
        self.name = "Squiggle"
        self.a = 1.5
        self.mean = jnp.zeros(dim)
        self.diagonal = jnp.array([5] + (dim - 1) * [0.05])
        self.xlim = [-11, 11]
        self.ylim = [-2.2, 2.2]
        self.true_dist_levels = [-5.0, -3.0, -1.5]

    @property
    def Sigma(self):
        return jnp.diag(self.diagonal)

    def logdensity_fn(self, x):
        mean = self.mean
        Sigma = self.Sigma
        g = jnp.insert(x[1:] + jnp.sin(self.a * x[0]), 0, x[0])
        return jss.multivariate_normal.logpdf(g, mean, Sigma)


def evaluate_logdensity_grid(distribution):
    num_x, num_y = distribution.grid_shape
    x_values = jnp.linspace(distribution.xlim[0], distribution.xlim[1], num_x)
    y_values = jnp.linspace(distribution.ylim[0], distribution.ylim[1], num_y)
    grid_x, grid_y = jnp.meshgrid(x_values, y_values, indexing="xy")
    points = jnp.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    logdensity = jax.vmap(distribution.logdensity_fn)(points).reshape(grid_x.shape)
    return np.asarray(grid_x), np.asarray(grid_y), np.asarray(logdensity)


def padded_limits(bounds, padding=AXIS_PADDING):
    lower, upper = bounds
    width = upper - lower
    return [lower - padding * width, upper + padding * width]


def plot_distribution_contours(distribution, output_dir=OUTPUT_DIR):
    fill_levels = distribution.fill_levels
    line_levels = distribution.line_levels
    grid_x, grid_y, logdensity = evaluate_logdensity_grid(distribution)
    relative_logdensity = logdensity - np.max(logdensity)
    masked_logdensity = np.ma.masked_less(relative_logdensity, fill_levels[0])
    line_logdensity = masked_logdensity
    if distribution.line_ymin is not None:
        line_logdensity = np.ma.masked_where(
            grid_y < distribution.line_ymin, masked_logdensity
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.contourf(
        grid_x,
        grid_y,
        masked_logdensity,
        levels=fill_levels,
        colors=FILL_COLORS,
        antialiased=True,
    )
    contour_lines = ax.contour(
        grid_x,
        grid_y,
        line_logdensity,
        levels=line_levels,
        colors="black",
        linestyles="solid",
        linewidths=0.55,
        antialiased=True,
    )
    contour_lines.set_capstyle("round")
    contour_lines.set_joinstyle("round")

    ax.set_xlim(padded_limits(distribution.xlim))
    ax.set_ylim(padded_limits(distribution.ylim))
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    output_path = output_dir / f"{distribution.name.lower()}_contour.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    distributions = [Funnel(), Rosenbrock(), Squiggle()]
    for distribution in distributions:
        output_path = plot_distribution_contours(distribution)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
