import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import utils


def plot_user_trajectory_var4(
    data,
    pat_id,
    var,
    time_col="days_since_first_session",
    var_error=None,
    hue_query_fit=None,
    hue_names_fit=None,
    hue_query_plot=None,
    hue_names_plot=None,
    plot_all_efforts=False,
    ax=None,
    title=None,
    figsize=(5, 3),
    do_legend=False,
    legend_loc=(0.5, 0.5),
    fs=12,
    ylabel=None,
    xlabel=None,
    fill_between=True,
    plot_regression_line=True,
    extend_regression=False,
    alpha=[0.9, 0.9],
    marker=["o", "+"],
    size=[50, 140],
    color_1="blue",
    color_2="green",
    color_3="red",
    color_4="red",
    alpha_lr=1,
    **kwargs,
):
    try:
        if var.startswith("fvc"):
            df1 = data.query('user_id == @pat_id and pftType=="fvc"').copy()
        elif var.startswith("vc"):
            df1 = data.query('user_id == @pat_id and pftType=="svc"').copy()
        else:
            df1 = data.query("user_id == @pat_id").copy()

        assert time_col in df1.columns, f"ERROR: {time_col} column not found"

        if var == "svc":
            var = "vc"
        df1 = df1.dropna(subset=[var, time_col])
    except KeyError:
        print(f"Patient with no {var.upper()}")
        return

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if hue_query_plot is not None:
        q1_plot, q2_plot = hue_query_plot
        if q1_plot is None and q2_plot is not None:
            df_plot = df1.query(f"{q2_plot}")
        elif q1_plot is not None and q2_plot is None:
            df_plot = df1.query(f"{q1_plot}")
        else:
            df_plot = df1.copy()
    else:
        df_plot = df1.copy()

    if var_error is not None:
        ax.fill_between(
            df_plot[time_col].values,
            df_plot[var].values - df_plot[var_error].values,
            df_plot[var].values + df_plot[var_error].values,
            color="gray",
            alpha=0.2,
        )

    if not plot_all_efforts:

        if hue_query_fit is None:
            l1_fit = hue_names_fit
            x1_fit = df1[time_col].values
            y1_fit = df1[var].values
        else:
            q1_fit, _ = hue_query_fit
            x1_fit = df1.query(q1_fit)[time_col].values
            y1_fit = df1.query(q1_fit)[var].values

        if hue_query_plot is None:
            l1_plot = hue_names_plot
            x1_plot = df1[time_col].values
            y1_plot = df1[var].values
        else:
            q1_plot, q2_plot = hue_query_plot
            l1_plot, l2_plot = hue_names_plot
            x1_plot = df1.query(q1_plot)[time_col].values
            y1_plot = df1.query(q1_plot)[var].values
            if q2_plot is not None:
                x2_plot = df1.query(q2_plot)[time_col].values
                y2_plot = df1.query(q2_plot)[var].values

        ax.scatter(
            x1_plot,
            y1_plot,
            color=color_2,
            marker=marker[0],
            alpha=alpha[0],
            label=l1_plot,
            s=size[0],
        )

        if len(x1_fit) < 3 or len(y1_fit) < 3:
            p_value = np.nan
        else:
            (
                y_pred,
                lower_bound,
                upper_bound,
                p_value,
                slope,
                _,
                intercept,
                _,
            ) = utils.fit_regression(x1_fit, y1_fit, output_all=True)
            if extend_regression:
                x1_fit = np.append(x1_fit, 0)
                y_pred = np.append(y_pred, intercept)
                x1_fit = np.append(x1_fit, 15)
                y_pred = np.append(y_pred, 15 * slope + intercept)

            if plot_regression_line:
                ax.plot(
                    x1_fit,
                    y_pred,
                    color=color_4,
                    alpha=alpha_lr,
                    linewidth=2,
                    label=None,
                )
                if fill_between:
                    ax.fill_between(
                        x1_fit, lower_bound, upper_bound, color="lightgrey", alpha=0.2
                    )

        if hue_query_plot is not None:
            if q2_plot is not None:
                ax.scatter(
                    x2_plot,
                    y2_plot,
                    color=color_3,
                    marker=marker[-1],
                    alpha=alpha[-1],
                    facecolors="white",
                    label=l2_plot,
                    s=size[-1],
                )

    if plot_all_efforts:
        all_eff = []
        for idx, row in df1.iterrows():
            date = row[time_col]
            efforts = row["efforts"]
            for i, effort in enumerate(efforts):
                if var in effort:
                    all_eff.append(
                        (date, row["session_id"], i, effort[var], row["is_proctored"])
                    )
                    ax.scatter(
                        date,
                        effort[var],
                        color="blue" if row["is_proctored"] else "red",
                        marker="o" if row["is_proctored"] else "+",
                        alpha=0.5,
                        label="Efforts",
                        s=100,
                    )

    ax.set_xlabel(xlabel, fontdict={"fontsize": fs})
    ax.set_ylabel(var.upper() if ylabel is None else ylabel, fontdict={"fontsize": fs})
    if time_col == "date_only":
        ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs)

    ax.set_title(
        title if title is not None else f"user_id: {pat_id[:8]}",
        fontdict={"fontsize": fs},
    )
    ax.grid(True)

    if do_legend:
        ax.legend(loc=legend_loc)

    if plot_all_efforts:
        return pd.DataFrame(
            all_eff, columns=["date", "session_id", "id", var, "proctored"]
        )


def plot_trajectory_panels3(
    data: pd.DataFrame,
    pat_list: list,
    data_dict: list,
    time_col="days_since_first_session",
    limit=2,
    n_rows=4,
    n_cols=5,
    figsize=(35 * 0.5, 20 * 0.5),
    posicion_text_der=1.19,
    plot_regression_line=True,
    fill_between=True,
    extend_regression=False,
    do_differt_color=[],
    xlim=None,
    mute_titles=True,
    mute_x_ticks=False,
    xlabel=None,
    titles=None,
    legend_loc="upper left",
    do_legend_index=0,
    **kwargs,
):
    """
    Plot trajectory panels for multiple patients, with dual y-axes.

    Args:
        data: DataFrame containing patient trajectories.
        pat_list: List of patient IDs to plot.
        data_dict: List of dicts with variable specs. First entry plots on the
            left axis; subsequent entries plot on the right (twin) axis.
            Each entry may include:
                var_name (str): column name to plot.
                label_y (str): y-axis label.
                ylim (tuple): y-axis limits.
                yticks (list): y-axis tick positions.
                yticklabels (list): y-axis tick labels.
                Plus any kwargs accepted by plot_user_trajectory_var4
                (hue_query_fit, color_1, alpha, marker, size, ...).
        time_col: Column name for the x-axis time variable.
        limit: Number of std deviations to draw as threshold lines. None to skip.
        n_rows, n_cols: Subplot grid dimensions.
        figsize: Figure size.
        posicion_text_der: x position (in axes coords) of the right-axis label text.
        xlim: x-axis limits.
        mute_titles: Hide per-panel titles.
        mute_x_ticks: Hide x tick labels except on the last row.
        do_differt_color: List of patient IDs to highlight with green spines.
        xlabel: x-axis label.
        titles: List of per-panel title strings (overrides default).

    Returns:
        fig: The matplotlib Figure.
    """
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    axs = axs.flatten()
    twin_axs = {}
    less_pat = len(pat_list) < n_rows * n_cols
    do_legend = kwargs.pop("do_legend", True)

    for i, pat in enumerate(pat_list):
        for i2, value in enumerate(data_dict):
            key = value["var_name"]
            for k in [
                "hue_query_fit", "hue_names_fit", "hue_query_plot", "hue_names_plot",
                "alpha", "marker", "size",
                "color_1", "color_2", "color_3", "color_4", "alpha_lr",
            ]:
                if k in value:
                    kwargs[k] = value[k]

            # i2 > 0 plots on the right (twin) axis so data uses the right scale
            if i2 > 0:
                if i not in twin_axs:
                    twin_axs[i] = axs[i].twinx()
                plot_ax = twin_axs[i]
            else:
                plot_ax = axs[i]

            plot_user_trajectory_var4(
                data,
                pat,
                key,
                time_col=time_col,
                fill_between=fill_between,
                plot_regression_line=plot_regression_line,
                extend_regression=extend_regression,
                ax=plot_ax,
                title=titles[i] if titles is not None else None,
                do_legend=False,
                **kwargs,
            )

            if limit is not None:
                datpat = data.query("user_id == @pat")
                m, s = datpat[key].mean(), datpat[key].std()
                plot_ax.axhline(limit * s + m, color="red", linestyle="dashdot", label=f"{limit} std")
                plot_ax.axhline(-1 * limit * s + m, color="red", linestyle="dashdot")
                handles, labels = plot_ax.get_legend_handles_labels()
                plot_ax.legend(handles, labels)

            # X labels and ticks (always on primary axis)
            if i >= (len(pat_list) - n_cols):
                axs[i].set_xlabel(xlabel)
            if mute_x_ticks and i < (n_rows - 1) * n_cols:
                axs[i].set_xticklabels("")

            # Y labels
            label = value.get("label_y", key)
            if i2 > 0:
                plot_ax.set_ylabel("")
            else:
                if i % n_cols != 0:
                    axs[i].set_ylabel("")
                else:
                    axs[i].set_ylabel(label, fontsize=15)

            if mute_titles:
                axs[i].set_title("")

            # Left axis limits and ticks
            if i2 == 0:
                if "ylim" in value:
                    axs[i].set_ylim(value["ylim"])
                if "yticks" in value:
                    axs[i].set_yticks(value["yticks"])
                if "yticklabels" in value:
                    axs[i].set_yticklabels(value["yticklabels"])
            if xlim is not None:
                axs[i].set_xlim(xlim)

        # Turn off unused subplots
        for i3 in range(len(pat_list), n_rows * n_cols):
            axs[i3].axis("off")

        if pat in do_differt_color:
            for spine in axs[i].spines.values():
                spine.set_color("green")
            print(f"{i}: {pat}")

    # Apply right-axis settings after full loop so sharey doesn't clobber them
    for i2, value in enumerate(data_dict):
        if i2 == 0:
            continue
        for i, tax in twin_axs.items():
            if "ylim" in value:
                tax.set_ylim(value["ylim"])
            if "yticks" in value:
                tax.set_yticks(value["yticks"])
            if i % n_cols == n_cols - 1:
                if "yticklabels" in value:
                    tax.set_yticklabels(value["yticklabels"])
            else:
                tax.set_yticklabels([])

    left_label = data_dict[0].get("label_y", data_dict[0]["var_name"])
    right_label = data_dict[1].get("label_y", data_dict[1]["var_name"]) if len(data_dict) > 1 else ""

    if do_legend:
        handles, labels = axs[do_legend_index].get_legend_handles_labels()
        if do_legend_index in twin_axs:
            h2, l2 = twin_axs[do_legend_index].get_legend_handles_labels()
            handles += h2
            labels += l2
        axs[do_legend_index].legend(handles, labels, loc=legend_loc)
    for i in range(len(pat_list)):
        axs[i].set_xticks([0, 4, 8, 12, 16])
        if i % n_cols == 0:
            axs[i].set_ylabel(left_label, fontsize=16)
        if i % n_cols == n_cols - 1:
            axs[i].set_ylabel("")
            axs[i].text(
                posicion_text_der, 0.5, right_label,
                va="center", ha="left", rotation=270, fontsize=15,
                transform=axs[i].transAxes,
            )
        if less_pat and i >= (len(pat_list) - n_cols):
            axs[i].set_xlabel(xlabel)
            axs[i].set_xticklabels([str(int(t)) for t in axs[i].get_xticks()])

    return fig
