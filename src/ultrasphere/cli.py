from logging import DEBUG, INFO, basicConfig, getLogger
from typing import Annotated, Literal

from array_api_compat import array_namespace
import array_api_extra as xpx
from array_api._2024_12 import Array
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

from joblib.parallel import Parallel, delayed

# import tracemalloc
from pandas import DataFrame, concat
from rich import print
from rich.logging import RichHandler
from tqdm import trange
from tqdm_joblib import tqdm_joblib

from ultrasphere.coordinates import SphericalCoordinates
from ultrasphere.harmonics import sph_harm

app = typer.Typer()


LOG = getLogger(__name__)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    level = INFO
    if verbose:
        level = DEBUG
    basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def plot_harm(n_end: Annotated[int, typer.Argument(help="Log the values")] = 5) -> None:
    """Plot the real part of the spherical harmonics."""
    n_div = 32
    fig, axes = plt.subplots(n_end, n_end * 2 - 1, subplot_kw={"projection": "3d"})
    theta = xp.linspace(0, np.pi, n_div)
    phi = xp.linspace(0, 2 * np.pi, 2 * n_div)
    theta, phi = xp.meshgrid(theta, phi)
    Y = sph_harm(
        phi=phi, theta=theta, n_end=n_end, condon_shortley_phase=True, concat=True
    )
    for n in range(n_end):
        for m in range(-n, n + 1):
            r = xp.abs(xp.real(Y[..., m, n]))
            # r = xp.abs(Y[..., m, n])
            x = r * xp.sin(theta) * xp.cos(phi)
            y = r * xp.sin(theta) * xp.sin(phi)
            z = r * xp.cos(theta)
            ax = axes[n, m + n_end - 1]
            ax.plot_surface(
                x.numpy(), y.numpy(), z.numpy(), rstride=1, cstride=1, cmap="viridis"
            )
            ax.set_title(f"n={n}, m={m}")
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
    fig.tight_layout()
    fig.savefig("sph_harm.svg")


def _task(
    branching_types: str,
    k: Array,
    x: Array,
    t_angle: Array,
    n_end: int,
    n_end_add_max: int,
    condon_shortley_phase: bool,
    size: int,
    ratio: float,
    from_: Literal["regular", "singular"],
    to_: Literal["regular", "singular"],
    use_triplet: bool,
) -> DataFrame:
    xp.set_backend("numpy")
    c = from_branching_types(branching_types)
    dfs = []
    t = t_angle * ratio
    y = x + t
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)
    t_spherical = c.from_euclidean(t)
    y_RS = harmonics_regular_singular(c,
        y_spherical,
        k=k,
        harmonics=harmonics(c, 
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=to_,
    )
    x_RS = harmonics_regular_singular(c,
        x_spherical,
        k=k,
        harmonics=harmonics(c, 
            x_spherical,
            n_end=n_end_add_max,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=from_,
    )
    if use_triplet:
        coef = harmonics_translation_coef_using_triplet(c,
            t_spherical,
            n_end=n_end,
            n_end_add=n_end_add_max,
            k=k,
            condon_shortley_phase=condon_shortley_phase,
            is_type_same=from_ == to_,
        )
    else:
        coef = harmonics_translation_coef(c,
            t,
            n_end=n_end,
            n_end_add=n_end_add_max,
            k=k,
            condon_shortley_phase=condon_shortley_phase,
        )
    x_RS_add = x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim]

    for n_end_add in range(n_end_add_max, 0, -1):
        x_RS_add = expand_cut(c, x_RS_add, n_end_add)
        coef = expand_cut(c, coef, n_end_add)
        y_RS_approx = xp.sum(
            x_RS_add * coef,
            axis=tuple(range(-c.s_ndim, 0)),
        )
        ae = xp.abs(y_RS - y_RS_approx)
        rel = xp.abs(1 - y_RS_approx / (y_RS + 1e-8))
        index = index_array_harmonics_all(c, 
            n_end=n_end,
            include_negative_m=True,
            expand_dims=True,
            as_array=True,
        )[c.root_index, ...][None, ...].repeat(size, axis=0)
        ae_flatten = flatten_harmonics(c,ae).flatten()
        rel_flatten = flatten_harmonics(c,rel).flatten()
        index_flatten = flatten_harmonics(c,index).flatten()
        df = DataFrame(
            {
                "absolute_error": ae_flatten,
                "relative_error": rel_flatten,
                "n": index_flatten,
            }
        )
        df["ratio"] = ratio
        df["from"] = from_
        df["to"] = to_
        df["use_triplet"] = use_triplet
        df["n_end_add"] = n_end_add
        dfs.append(df)
    return concat(dfs)


@app.command()
def plot_translation_error(
    branching_types: str = typer.Option("ba", "--branching-types", "-b", "-bt"),
    size: int = typer.Option(100, "--size", "-s"),
    n_end: int = typer.Option(8, "--n-end", "-n"),
    k: str = typer.Option("1.0", "--k", "-k"),
) -> None:
    """
    Plot the error of the translation of
    elementary solutions.
    """
    xp.set_backend("numpy")
    condon_shortley_phase = False
    k = xp.asarray(complex(k)).to_numpy()
    n_end_add_max = n_end
    c = from_branching_types(branching_types)
    rng = np.random.default_rng(0)
    x = random_points(c, shape=(size,), surface=True, rng=rng).to_numpy()
    t_angle = random_points(c, shape=(size,), surface=True, rng=rng).to_numpy()

    task_args = []
    for ratio in [1 / 4, 1 / 2, 1, 2, 4]:
        for from_, to_ in [
            ("regular", "regular"),
            ("singular", "singular"),
            ("regular", "singular"),
        ]:
            for use_triplet in [True, False]:
                if from_ != to_ and not use_triplet:
                    continue
                if (from_, to_) == ("singular", "singular") and ratio > 1:
                    continue
                if (from_, to_) == ("regular", "singular") and ratio < 1:
                    continue
                task_args.append(
                    (
                        branching_types,
                        k,
                        x,
                        t_angle,
                        n_end,
                        n_end_add_max,
                        condon_shortley_phase,
                        size,
                        ratio,
                        from_,
                        to_,
                        use_triplet,
                    )
                )
    with tqdm_joblib(total=len(task_args)):
        dfs = Parallel(n_jobs=1)(delayed(_task)(*task_arg) for task_arg in task_args)

    # concatenate the results
    df = concat(dfs)

    # clean column names
    df["From→To, Type"] = (
        df["to"].map({"regular": "R", "singular": "S"})
        + "←"
        + df["from"].map({"regular": "R", "singular": "S"})
        + ", "
        + df["use_triplet"].map({True: "Triplet", False: "Integral"})
    )
    df["Max Degree (From)"] = df["n_end_add"] - 1
    df.rename(
        columns={
            "n": "Degree (To)",
            "ratio": "|t|/|x|",
            "absolute_error": "Absolute Error",
            "relative_error": "Relative Error",
        },
        inplace=True,
    )
    for error in ["Absolute Error", "Relative Error"]:
        sns.set_theme(font_scale=2.2)
        g = sns.catplot(
            df,
            x="Max Degree (From)",
            y=error,
            col="|t|/|x|",
            row="From→To, Type",
            hue="Degree (To)",
            kind="box",
            log_scale=True,
            saturation=0.5,
            # split=True,
        )
        g.set(ylim=(1e-12, 1))
        g.figure.suptitle(
            f"Branching Types: {branching_types}, "
            f"|x| = 1 ({size} random points), k = {complex(k):g}"
        )
        g.set_titles("{col_var}={col_name}
{row_name}")
        g.tight_layout()
        plt.savefig(f"{branching_types}-{n_end}-{k}-{error}.svg")
        plt.savefig(f"{branching_types}-{n_end}-{k}-{error}.png")


@app.command()
def benchmark_sph_harm(
    shape: str = typer.Option("1000", "--shape", "-s"),
    n_end: int = typer.Option(25, "--n-end", "-n"),
    backend: str = typer.Option("torch", "--backend", "-b"),
    device: str = typer.Option("cpu", "--device", "-d"),
) -> None:
    """Benchmark the spherical harmonics."""
    from array_api_compat import array_namespace
import array_api_extra as xpx
from array_api._2024_12 import Array
    from cm_time import timer
    from scipy.special import sph_harm as sp_sph_harm

    from .harmonics import sph_harm

    if device in ["gpu", "tpu"]:
        device = f"{device}:0"
    xp.set_backend(backend)
    xp.default_device(device)

    # xp
    phi = xp.random.random_uniform(
        low=0, high=2 * xp.pi, shape=[int(i) for i in shape.split(",")]
    )
    theta = xp.random.random_uniform(
        low=0, high=xp.pi, shape=[int(i) for i in shape.split(",")]
    )
    for _ in range(3):
        with timer() as t1:
            sph_harm(phi=phi, theta=theta, n_end=n_end, condon_shortley_phase=True)
        print(f"xp: {t1.elapsed}")

    # scipy
    m = xp.reshape(xp.arange(-n_end, n_end + 1),[1] * theta.ndim + [-1, 1])
    n = xp.reshape(xp.arange(n_end + 1),[1] * theta.ndim + [1, -1])
    theta = theta[..., None, None]
    phi = phi[..., None, None]
    m, n, phi, theta = (
        xp.to_numpy(m),
        xp.to_numpy(n),
        xp.to_numpy(phi),
        xp.to_numpy(theta),
    )
    for _ in range(3):
        with timer() as t2:
            sp_sph_harm(m, n, phi, theta)
        print(f"scipy: {t2.elapsed}")
    print(f"xp: {t1.elapsed}, scipy: {t2.elapsed}")


@app.command()
def benchmark_c2s(
    shape: str = typer.Option("1000", "--shape", "-s"),
    backend: str = typer.Option("torch", "--backend", "-b"),
    device: str = typer.Option("cpu", "--device", "-d"),
    dtype: str = typer.Option("float32", "--dtype", "-t"),
) -> None:
    """
    Benchmark different implementations of
    the cartesian to spherical coordinate transformation.

    Parameters
    ----------
    shape : str, optional
        The shape of the input tensor, by default 1000.
    backend : str, optional
        The backend to use, by default "torch".
    device : str, optional
        The device to use, by default "cpu".
    dtype : str, optional
        The data type to use, by default "float32".

    """
    from collections import defaultdict
    from collections.abc import Callable
    from functools import partial

    from cm_time import timer
    from xp import Array

    def forward(r: Array, theta: Array, phi: Array) -> tuple[Array, Array, Array]:
        rsin = r * xp.sin(theta)
        x = rsin * xp.cos(phi)
        y = rsin * xp.sin(phi)
        z = r * xp.cos(theta)
        return x, y, z

    def backward_copilot(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
        r = xp.linalg.vector_norm([x, y, z], axis=0)
        theta = xp.acos(z / r)
        phi = xp.atan2(y, x)
        return r, theta, phi

    def backward_atan2_vector_norm(
        x: Array, y: Array, z: Array, vector_norm: bool, r_independent: bool
    ) -> tuple[Array, Array, Array]:
        phi = xp.atan2(y, x)
        rsin = xp.linalg.vector_norm([x, y], axis=0)
        theta = xp.atan2(rsin, z)

        if vector_norm:
            r = xp.linalg.vector_norm(
                [x, y, z] if r_independent else [rsin, z], axis=0
            )
        else:
            r = (
                xp.sqrt(x**2 + y**2 + z**2)
                if r_independent
                else xp.sqrt(rsin**2 + z**2)
            )
        return r, theta, phi

    functions: dict[
        str, Callable[[Array, Array, Array], tuple[Array, Array, Array]]
    ] = {
        "copilot": backward_copilot,
        "vector_norm": partial(
            backward_atan2_vector_norm, vector_norm=True, r_independent=False
        ),
        "vector_norm_r_independent": partial(
            backward_atan2_vector_norm, vector_norm=True, r_independent=True
        ),
        "sqrt": partial(
            backward_atan2_vector_norm, vector_norm=False, r_independent=False
        ),
        "sqrt_r_independent": partial(
            backward_atan2_vector_norm, vector_norm=False, r_independent=True
        ),
    }

    xp.set_backend(backend)
    xp.default_device(device)
    xp.set_default_dtype(dtype)
    shape_ = [int(i) for i in shape.split(",")]
    ts = defaultdict(list)
    errors = defaultdict(list)
    for i in trange(20):
        x = xp.random.random_uniform(low=-1, high=1, shape=shape_)
        y = xp.random.random_uniform(low=-1, high=1, shape=shape_)
        z = xp.random.random_uniform(low=-1, high=1, shape=shape_)
        for name, func in functions.items():
            with timer() as t:
                r, theta, phi = func(x, y, z)
            if i < 3:
                # warm up
                continue
            ts[name].append(t.elapsed)
            x_, y_, z_ = forward(r, theta, phi)
            diff = xp.linalg.vector_norm([x - x_, y - y_, z - z_], axis=0)
            errors[name].append(xp.mean(diff))
    print(
        "Time: ",
        {k: f"{float(xp.mean(v)):.2e}±{float(xp.std(v)):.2e}" for k, v in ts.items()},
    )
    print(
        "Error: ",
        {
            k: f"{float(xp.mean(v)):.2e}±{float(xp.std(v)):.2e}"
            for k, v in errors.items()
        },
    )


if __name__ == "__main__":
    app()
