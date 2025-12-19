import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn2

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def barplot_sentiment_repartition(full_title_df, full_body_df):
    # Compute counts
    title_counts = full_title_df["LINK_SENTIMENT"].value_counts().sort_index()
    body_counts = full_body_df["LINK_SENTIMENT"].value_counts().sort_index()

    # Define color palette and labels
    colors = { -1: "red", 1: "royalblue" }
    labels = { -1: "Negative (-1)", 1: "Positive (+1)" }

    # --- Plot for TITLE ---
    plt.figure(figsize=(6,4))
    sns.barplot(
        x=title_counts.index.astype(str),
        y=title_counts.values,
        hue=title_counts.index.astype(str),             # rm palette warning
        palette=[colors[i] for i in title_counts.index],
        errorbar=None,
        legend=False  
    )

    plt.title("Number of Positive and Negative Crosslinks — TITLE", fontsize=13, weight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Posts")
    plt.legend(handles=[
        plt.Rectangle((0,0),1,1, color=colors[-1], label=labels[-1]),
        plt.Rectangle((0,0),1,1, color=colors[1], label=labels[1])
    ], title="Sentiment")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()

    # --- Plot for BODY ---
    plt.figure(figsize=(6,4))
    sns.barplot(
        x=title_counts.index.astype(str),
        y=title_counts.values,
        hue=title_counts.index.astype(str),             # rm palette warning
        palette=[colors[i] for i in title_counts.index],
        errorbar=None,
        legend=False  
    )
    plt.title("Number of Positive and Negative Crosslinks — BODY", fontsize=13, weight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Posts")
    plt.legend(handles=[
        plt.Rectangle((0,0),1,1, color=colors[-1], label=labels[-1]),
        plt.Rectangle((0,0),1,1, color=colors[1], label=labels[1])
    ], title="Sentiment")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()

def  create_2D_coordinates(df):
    embeddings = df.iloc[:, 1:301].to_numpy()

    coords = umap.UMAP(n_components=2, random_state=0, n_jobs=1).fit_transform(embeddings) # n_jobs=1 removes warning
        
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    return df

def plot_negative_links_over_time(df, label="Title"):
    """
    Plots the evolution of the number of negative crosslinks (-1 sentiment)
    over time (monthly aggregated).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing LINK_SENTIMENT and a datetime column (e.g., LINK_DATE).
    label : str
        Label used in the plot title (e.g., "Title" or "Body").
    """

    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"]):
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    # Filter only negative links
    neg_df = df[df["LINK_SENTIMENT"] == -1].copy()

    # Group by month
    monthly_counts = (
        neg_df
        .groupby(pd.Grouper(key="TIMESTAMP", freq="W"))
        .size()
        .reset_index(name="count")
    )

    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=monthly_counts,
        x="TIMESTAMP",
        y="count",
        color="red",
        linewidth=2.2
    )

    plt.title(f"Evolution of Negative Crosslinks Over Time — {label}", fontsize=13, weight="bold")
    plt.xlabel("Time (Monthly)")
    plt.ylabel("Number of Negative Links")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_embedded_clustering(df):
    embeddings = df.iloc[:, 1:301].to_numpy()
    coords = umap.UMAP(n_components=2, random_state=0, n_jobs=1).fit_transform(embeddings) # n_jobs=1 removes warning

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    plt.scatter(df["x"], df["y"], s=5, alpha=0.4)
    plt.show()

def combine_datasets(title_df, body_df, sub_df):
    embeddings = sub_df.iloc[:, 1:301].to_numpy()
    coords = umap.UMAP(n_components=2, random_state=0, n_jobs=1).fit_transform(embeddings)
    
    coords = sub_df[["SUBREDDIT", "x", "y"]].copy()
    coords = coords.rename(columns={"x": "sub_x", "y": "sub_y"})

    #-------------------------
    # Title combination 
    #------------------------- 
    
    ## ADD coords source 
    title_df = title_df.merge(
        coords,
        left_on="SOURCE_SUBREDDIT",
        right_on="SUBREDDIT",
        how="left"
    )

    title_df = title_df.rename(columns={
        "sub_x": "source_x",
        "sub_y": "source_y"
    })

    title_df = title_df.drop(columns=["SUBREDDIT"])


    ## ADD coords target

    title_df = title_df.merge(
        coords,
        left_on="TARGET_SUBREDDIT",
        right_on="SUBREDDIT",
        how="left"
    )

    title_df = title_df.rename(columns={
        "sub_x": "target_x",
        "sub_y": "target_y"
    })

    title_df = title_df.drop(columns=["SUBREDDIT"])
    
    #-------------------------
    # BODY combination 
    #-------------------------   

    body_df = body_df.merge(
        coords,
        left_on="SOURCE_SUBREDDIT",
        right_on="SUBREDDIT",
        how="left"
    )

    body_df = body_df.rename(columns={
        "sub_x": "source_x",
        "sub_y": "source_y"
    })

    body_df = body_df.drop(columns=["SUBREDDIT"])


    ## ADD coords target

    body_df = body_df.merge(
        coords,
        left_on="TARGET_SUBREDDIT",
        right_on="SUBREDDIT",
        how="left"
    )

    body_df = body_df.rename(columns={
        "sub_x": "target_x",
        "sub_y": "target_y"
    })

    body_df = body_df.drop(columns=["SUBREDDIT"])

    return title_df, body_df

def plot_data_sent_clustering(
        df, 
        title="Subreddit 2D Projection — Month by Month",
        time_col="TIMESTAMP", 
        x_col="source_x", 
        y_col="source_y",
        snapshot: str | None = None
    ):
    plot_df = df.copy()

    # Ensure MONTH exists
    if "MONTH" not in plot_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
            # plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
            plot_df[time_col] = pd.to_datetime(plot_df[time_col], utc=True).dt.tz_localize(None)
        # plot_df["MONTH"] = plot_df[time_col].dt.to_period("M").astype(str)
        plot_df["MONTH"] = plot_df[time_col].dt.strftime("%Y-%m")


    # Build a truly chronological order for the slider
    # (use PeriodIndex to avoid lexicographic issues)
    month_order = (pd.PeriodIndex(plot_df["MONTH"], freq="M")
                     .sort_values()
                     .astype(str)
                     .unique()
                     .tolist())

    # Force MONTH to be an ordered categorical (controls slider order)
    plot_df["MONTH"] = pd.Categorical(plot_df["MONTH"],
                                      categories=month_order,
                                      ordered=True)

    # Ensure SENT_LABEL exists
    if "SENT_LABEL" not in plot_df.columns:
        plot_df["SENT_LABEL"] = (
            plot_df["LINK_SENTIMENT"]
            .round()
            .map({-1: "Negative", 1: "Positive"})
            .fillna("Neutral")
        )

    existing_labels = [lab for lab in ["Negative", "Positive", "Neutral"]
                       if lab in plot_df["SENT_LABEL"].unique()]
    color_map = {"Positive": "royalblue", "Negative": "red", "Neutral": "lightgray"}

    if snapshot is not None:
        if snapshot not in month_order:
            raise ValueError(
                f"snapshot='{snapshot}' not found. "
                f"Valid months include, e.g., {month_order[:5]} ... {month_order[-5:]}"
            )

        snap_df = plot_df[plot_df["MONTH"] == snapshot]

        fig = px.scatter(
            snap_df,
            x=x_col, y=y_col,
            hover_name="SOURCE_SUBREDDIT",
            opacity=0.85,
            color="SENT_LABEL",
            color_discrete_map={k: color_map[k] for k in existing_labels},
            category_orders={"SENT_LABEL": existing_labels},
            title=f"{title} — Snapshot {snapshot}",
            width=900, height=700,
        )
        fig.update_layout(template="plotly_white",
                          xaxis_title="UMAP 1", yaxis_title="UMAP 2")
        fig.show("png")
        return
    
    fig = px.scatter(
        plot_df,
        x=x_col, y=y_col,
        animation_frame="MONTH",
        hover_name="SOURCE_SUBREDDIT",
        opacity=0.85,
        color="SENT_LABEL",
        color_discrete_map={k: color_map[k] for k in existing_labels},
        category_orders={"MONTH": month_order, "SENT_LABEL": existing_labels},
        title=title,
        width=900, height=700,
    )

    fig.update_layout(template="plotly_white",
                      xaxis_title="UMAP 1", yaxis_title="UMAP 2")
    fig.show()

def plot_data_link_sent_clustering(
    df: pd.DataFrame,
    title="Subreddit link graph — month by month",
    frame_duration_ms=700,
    snapshot: str | None = None
):
    required = [
        "LINK_SENTIMENT", "source_x", "source_y", "target_x", "target_y",
        "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df.copy()

    # Ensure MONTH exists
    if "MONTH" not in d.columns:
        if "TIMESTAMP" not in d.columns:
            raise ValueError("MONTH not found and TIMESTAMP is missing to compute it.")
        if not pd.api.types.is_datetime64_any_dtype(d["TIMESTAMP"]):
            # d["TIMESTAMP"] = pd.to_datetime(d["TIMESTAMP"], errors="coerce")
            d["TIMESTAMP"] = pd.to_datetime(d["TIMESTAMP"], utc=True).dt.tz_localize(None)
        # d["MONTH"] = d["TIMESTAMP"].dt.to_period("M").astype(str)
        d["MONTH"] = d["TIMESTAMP"].dt.strftime("%Y-%m")

    # ---- Chronological order for slider/frames ----
    month_order = (pd.PeriodIndex(d["MONTH"], freq="M")
                     .sort_values()
                     .astype(str)
                     .unique()
                     .tolist())
    if not month_order:
        raise ValueError("No MONTH values found to animate.")

    # Force MONTH as ordered categorical so Plotly keeps the order
    d["MONTH"] = pd.Categorical(d["MONTH"], categories=month_order, ordered=True)

    def build_segments(dframe):
        pos = dframe[dframe["LINK_SENTIMENT"] > 0]
        neg = dframe[dframe["LINK_SENTIMENT"] < 0]

        pos_x, pos_y = [], []
        for _, r in pos.iterrows():
            pos_x += [r["source_x"], r["target_x"], None]
            pos_y += [r["source_y"], r["target_y"], None]

        neg_x, neg_y = [], []
        for _, r in neg.iterrows():
            neg_x += [r["source_x"], r["target_x"], None]
            neg_y += [r["source_y"], r["target_y"], None]

        return pos_x, pos_y, neg_x, neg_y

    def build_nodes(dframe):
        src = dframe[["SOURCE_SUBREDDIT", "source_x", "source_y"]].rename(
            columns={"SOURCE_SUBREDDIT": "name", "source_x": "x", "source_y": "y"}
        )
        tgt = dframe[["TARGET_SUBREDDIT", "target_x", "target_y"]].rename(
            columns={"TARGET_SUBREDDIT": "name", "target_x": "x", "target_y": "y"}
        )
        nodes = pd.concat([src, tgt], ignore_index=True)
        nodes = nodes.dropna(subset=["x", "y"]).drop_duplicates(subset=["name"])
        return nodes

    # Initial frame (first month chronologically)
    m0 = month_order[0]
    df0 = d[d["MONTH"] == m0]
    pos_x0, pos_y0, neg_x0, neg_y0 = build_segments(df0)
    nodes0 = build_nodes(df0)

    # If a snapshot month is requested, plot only that snapshot
    if snapshot is not None:
        if snapshot not in month_order:
            raise ValueError(
                f"snapshot_month '{snapshot}' not found in dataset. "
                f"Available months: {month_order[:5]} ... {month_order[-5:]}"
            )

        df_snap = d[d["MONTH"] == snapshot]
        pos_x, pos_y, neg_x, neg_y = build_segments(df_snap)
        nodes = build_nodes(df_snap)

        fig = go.Figure(
            data=[
                go.Scatter(x=pos_x, y=pos_y, mode="lines",
                        line=dict(color="royalblue", width=1),
                        name="Positive", hoverinfo="skip"),
                go.Scatter(x=neg_x, y=neg_y, mode="lines",
                        line=dict(color="red", width=1),
                        name="Negative", hoverinfo="skip"),
                go.Scatter(x=nodes["x"], y=nodes["y"], mode="markers",
                        marker=dict(size=4, color="black"),
                        name="Subreddits", hovertext=nodes["name"], hoverinfo="text"),
            ]
        )

        fig.update_layout(
            width=900, height=700,
            template="plotly_white",
            title=f"{title} — snapshot {snapshot}",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            showlegend=True
        )
        fig.show("png")
        return

    fig = go.Figure(
        data=[
            go.Scatter(x=pos_x0, y=pos_y0, mode="lines",
                       line=dict(color="royalblue", width=1),
                       name="Positive", hoverinfo="skip"),
            go.Scatter(x=neg_x0, y=neg_y0, mode="lines",
                       line=dict(color="red", width=1),
                       name="Negative", hoverinfo="skip"),
            go.Scatter(x=nodes0["x"], y=nodes0["y"], mode="markers",
                       marker=dict(size=4, color="black"),
                       name="Subreddits", hovertext=nodes0["name"], hoverinfo="text"),
        ]
    )

    # Frames in chronological order
    frames = []
    for m in month_order:
        dfm = d[d["MONTH"] == m]
        pos_x, pos_y, neg_x, neg_y = build_segments(dfm)
        nodes = build_nodes(dfm)
        frames.append(
            go.Frame(
                name=m,
                data=[
                    go.Scatter(x=pos_x, y=pos_y),
                    go.Scatter(x=neg_x, y=neg_y),
                    go.Scatter(x=nodes["x"], y=nodes["y"], hovertext=nodes["name"]),
                ]
            )
        )
    fig.frames = frames

    # Slider with months in chronological order
    fig.update_layout(
        width=900, height=700,
        template="plotly_white",
        title=title,
        showlegend=True,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ],
            showactive=False
        )],
        sliders=[dict(
            active=0,
            steps=[
                dict(method="animate",
                     args=[[m], {"mode": "immediate",
                                 "frame": {"duration": 0, "redraw": True},
                                 "transition": {"duration": 0}}],
                     label=m)
                for m in month_order
            ]
        )]
    )

    fig.show()


def compute_incoming_rate(df):
    df = df.copy()
    # df['week_start'] = pd.to_datetime(df['TIMESTAMP']).dt.to_period('W').dt.start_time
    df['week_start'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_localize(None).dt.to_period('W').dt.start_time


    incoming_neg = df[df['LINK_SENTIMENT'] == -1].groupby(['TARGET_SUBREDDIT', 'week_start']).size()
    incoming_total = df.groupby(['TARGET_SUBREDDIT', 'week_start']).size()
    incoming_rate = (incoming_neg / incoming_total).fillna(0)

    return incoming_rate

def compute_outgoing_rate(df):
    df = df.copy()
    # df['week_start'] = pd.to_datetime(df['TIMESTAMP']).dt.to_period('W').dt.start_time
    df['week_start'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_localize(None).dt.to_period('W').dt.start_time

    outgoing_neg = df[df['LINK_SENTIMENT'] == -1].groupby(['SOURCE_SUBREDDIT', 'week_start']).size()
    outgoing_total = df.groupby(['SOURCE_SUBREDDIT', 'week_start']).size()
    outgoing_rate = (outgoing_neg / outgoing_total).fillna(0)

    return outgoing_rate

def find_spike_candidates(df):
    df = df.copy()

    incoming_rate = compute_incoming_rate(df)
    outgoing_rate = compute_outgoing_rate(df)

    candidates = []
    for sub in df['SOURCE_SUBREDDIT'].unique():
        in_data = incoming_rate.get(sub, pd.Series(dtype=float))
        out_data = outgoing_rate.get(sub, pd.Series(dtype=float))

        if len(in_data) > 20 and len(out_data) > 20:
            if in_data.std() > 0.1 and out_data.std() > 0.1:
                candidates.append({
                    'subreddit': sub,
                    'in_mean': in_data.mean(),
                    'in_std': in_data.std(),
                    'out_mean': out_data.mean(),
                    'out_std': out_data.std()
                })

    candidates_df = pd.DataFrame(candidates).sort_values('in_std', ascending=False)

    return candidates_df


def analyze_subreddit_spike(body_df, incoming_rate, outgoing_rate, subreddit):
    # extract series for the chosen subreddit
    sub_in = incoming_rate[subreddit].reset_index()
    sub_in.columns = ['week', 'rate']
    sub_out = outgoing_rate[subreddit].reset_index()
    sub_out.columns = ['week', 'rate']

    # merge incoming and outgoing rates
    merged = pd.merge(sub_in, sub_out, on='week', suffixes=('_in', '_out'))

    # compute spike thresholds
    in_threshold = merged['rate_in'].mean() + 2 * merged['rate_in'].std()
    out_threshold = merged['rate_out'].mean() + 2 * merged['rate_out'].std()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(merged['week'], merged['rate_in'], 'r-o', label='Incoming', markersize=3)
    ax.plot(merged['week'], merged['rate_out'], 'b-s', label='Outgoing', markersize=3)
    ax.axhline(in_threshold, color='r', linestyle='--', alpha=0.3)
    ax.axhline(out_threshold, color='b', linestyle='--', alpha=0.3)
    ax.set_title(f'Cascade Pattern Example: {subreddit}')
    ax.set_ylabel('Negativity Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # correlation across lags
    print(f"\nLag correlations for {subreddit}")
    for lag in [0, 1, 2, 3]:
        corr = merged['rate_in'].corr(merged['rate_out'].shift(-lag))
        print(f"Lag {lag} weeks: r = {corr:.3f}")

    return merged


def plot_top_negative_sources(
    df,
    label_col="LINK_SENTIMENT",
    source_col="SOURCE_SUBREDDIT",
    negative_value=-1,
    top_n=20,
    title=None,
    annotate=True,
    figsize=(12, 6),
    verbose=True,
):
    
    def resolve_col(df, preferred, aliases=()):
        norm = {str(c): str(c).lower().replace(" ", "").replace("_", "") for c in df.columns}
        candidates = [preferred] + list(aliases)
        targets = [t.lower().replace(" ", "").replace("_", "") for t in candidates]
        for col, colnorm in norm.items():
            if colnorm in targets:
                return col
        raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")

    # 0) copy & strip column names
    df = df.copy()
    df.columns = df.columns.map(lambda c: str(c).strip())

    # 1) resolve
    label_col_res  = resolve_col(df, label_col,  ("POST_LABEL","SENTIMENT","LINKSENTIMENT","LABEL"))
    source_col_res = resolve_col(df, source_col, ("SOURCE SUBREDDIT","SRC_SUBREDDIT","SRCSUBREDDIT","SOURCE"))
    if verbose:
        print(f"Resolved columns → sentiment: '{label_col_res}', source: '{source_col_res}'")

    # 2) mask & series
    mask = df[label_col_res] == negative_value
    src_series = df[source_col_res]
    neg_src = src_series[mask]
    if neg_src.empty:
        raise ValueError(f"No rows where {label_col_res} == {negative_value}")

    # 3) counts (robust naming)
    counts = (
        neg_src.value_counts(dropna=False)
               .reset_index(name="negative_count")         # <— ensure column exists
               .rename(columns={"index": source_col_res})
               .sort_values("negative_count", ascending=False)
               .head(top_n)
    )
    counts["negative_count"] = pd.to_numeric(counts["negative_count"], errors="coerce").fillna(0).astype(int)

    # 4) share
    total_neg  = int(neg_src.shape[0])
    top_n_count = int(counts["negative_count"].sum())
    pct = float(100.0 * top_n_count / total_neg) if total_neg else 0.0

    # 5) plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=counts,
        x=source_col_res,
        y="negative_count",
        hue=source_col_res,
        order=counts[source_col_res].tolist(),
        palette="Reds_r",
        legend=False
    )

    ax.set_xlabel("Source Subreddit")
    ax.set_ylabel("Number of Negative Hyperlinks")

    main_title = title or f"Top {top_n} Subreddits Sending the Most Negative Hyperlinks"
    subtitle   = f"(Top {top_n} share: {pct:.2f}% = {top_n_count}/{total_neg})"
    ax.set_title(f"{main_title}\n{subtitle}", fontsize=13, pad=16)

    plt.xticks(rotation=75, ha="right")

    if annotate:
        for p in ax.patches:
            h = p.get_height()
            if pd.notnull(h) and h > 0:
                ax.annotate(
                    f"{int(h)}",
                    (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points",
                )

    plt.tight_layout()
    plt.show()



def plot_top_negative_targets(
    df,
    label_col="LINK_SENTIMENT",
    target_col="TARGET_SUBREDDIT",
    negative_value=-1,
    top_n=20,
    title=None,
    annotate=True,
    figsize=(12, 6),
    verbose=True,
):
    
    def resolve_col(df, preferred, aliases=()):
        norm = {str(c): str(c).lower().replace(" ", "").replace("_", "") for c in df.columns}
        candidates = [preferred] + list(aliases)
        targets = [t.lower().replace(" ", "").replace("_", "") for t in candidates]
        for col, colnorm in norm.items():
            if colnorm in targets:
                return col
        raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")

    # 0) copy & strip column names
    df = df.copy()
    df.columns = df.columns.map(lambda c: str(c).strip())

    # 1) resolve
    label_col_res   = resolve_col(df, label_col,  ("POST_LABEL","SENTIMENT","LINKSENTIMENT","LABEL"))
    target_col_res  = resolve_col(df, target_col, ("TARGET SUBREDDIT","TGT_SUBREDDIT","TGTSUBREDDIT","TARGET"))
    if verbose:
        print(f"Resolved columns → sentiment: '{label_col_res}', target: '{target_col_res}'")

    # 2) mask & series
    mask = df[label_col_res] == negative_value
    tgt_series = df[target_col_res]
    neg_tgt = tgt_series[mask]
    if neg_tgt.empty:
        raise ValueError(f"No rows where {label_col_res} == {negative_value}")

    # 3) counts
    counts = (
        neg_tgt.value_counts(dropna=False)
               .reset_index(name="negative_count")
               .rename(columns={"index": target_col_res})
               .sort_values("negative_count", ascending=False)
               .head(top_n)
    )
    counts["negative_count"] = pd.to_numeric(counts["negative_count"], errors="coerce").fillna(0).astype(int)

    # 4) share
    total_neg  = int(neg_tgt.shape[0])
    top_n_count = int(counts["negative_count"].sum())
    pct = float(100.0 * top_n_count / total_neg) if total_neg else 0.0

    # 5) plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=counts,
        x=target_col_res,
        y="negative_count",
        hue=target_col_res,
        order=counts[target_col_res].tolist(),
        palette="Reds_r",
        legend=False
    )

    ax.set_xlabel("Target Subreddit")
    ax.set_ylabel("Number of Negative Hyperlinks")

    main_title = title or f"Top {top_n} Subreddits Receiving the Most Negative Hyperlinks"
    subtitle   = f"(Top {top_n} share: {pct:.2f}% = {top_n_count}/{total_neg})"
    ax.set_title(f"{main_title}\n{subtitle}", fontsize=13, pad=16)

    plt.xticks(rotation=75, ha="right")

    if annotate:
        for p in ax.patches:
            h = p.get_height()
            if pd.notnull(h) and h > 0:
                ax.annotate(
                    f"{int(h)}",
                    (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9,
                    xytext=(0, 3), textcoords="offset points",
                )

    plt.tight_layout()
    plt.show()




def show_negativity_ratio(
    df,
    label_col="LINK_SENTIMENT",
    source_col="SOURCE_SUBREDDIT",
    negative_value=-1,
    top_n=20,
    min_links=50,  # <-- NEW PARAMETER
    figsize=(12, 6),
):
    """
    Compute and plot Negativity Ratio = (# negative posts) / (total # posts)
    for subreddits with more than `min_links` total posts.


    Parameters
    ----------
    df : pd.DataFrame
        Merged hyperlink dataset.
    label_col : str
        Column indicating sentiment (+1 / -1).
    source_col : str
        Column with source subreddit names.
    negative_value : int or str
        Value denoting negative sentiment (default: -1).
    top_n : int
        Number of top subreddits to display (default: 20).
    min_links : int
        Minimum total hyperlinks required to include subreddit.
    figsize : tuple
        Figure size (default: (12, 6)).


    Returns
    -------
    pd.DataFrame
        DataFrame containing subreddit, negative_posts, total_posts, and negativity_ratio.
    """

    # --- Compute counts
    total_counts = df[source_col].value_counts()
    neg_counts = df.loc[df[label_col] == negative_value, source_col].value_counts()

    # --- Combine
    ratio_df = pd.DataFrame({
        "total_posts": total_counts,
        "negative_posts": neg_counts
    }).fillna(0)

    # --- Filter by minimum total posts
    ratio_df = ratio_df[ratio_df["total_posts"] > min_links]

    if ratio_df.empty:
        print(f"⚠️ No subreddits with more than {min_links} hyperlinks found.")
        return ratio_df

    # --- Compute ratio
    ratio_df["negativity_ratio"] = ratio_df["negative_posts"] / ratio_df["total_posts"]

    # --- Sort and select top N
    ratio_df = ratio_df.sort_values("negativity_ratio", ascending=False).head(top_n)

    # --- Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=ratio_df,
        x=ratio_df.index,
        y="negativity_ratio",
        hue=ratio_df.index,
        legend=False,
        palette="Reds_r"
    )

    ax.set_title(
        f"Top {top_n} Subreddits by Negativity Ratio\n"
        f"(only showing those with > {min_links} total hyperlinks)",
        fontsize=13, pad=20
    )
    ax.set_xlabel("Source Subreddit")
    ax.set_ylabel("Negativity Ratio = (# Negative Posts) / (Total Posts)")
    plt.xticks(rotation=75, ha="right")

    # Annotate ratios on bars
    for p in ax.patches:
        h = p.get_height()
        if pd.notnull(h):
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=9,
                xytext=(0, 3), textcoords="offset points"
            )

    plt.tight_layout()


    plt.show()


def show_negativity_ratio_target(
    df,
    label_col="LINK_SENTIMENT",
    target_col="TARGET_SUBREDDIT",
    negative_value=-1,
    top_n=20,
    min_links=50,
    figsize=(12, 6),
):
    """
    Compute and plot Negativity Ratio = (# negative posts) / (total # posts)
    for target subreddits with more than `min_links` total posts.

    Parameters
    ----------
    df : pd.DataFrame
        Merged hyperlink dataset.
    label_col : str
        Column indicating sentiment (+1 / -1).
    target_col : str
        Column with target subreddit names.
    negative_value : int or str
        Value denoting negative sentiment (default: -1).
    top_n : int
        Number of top subreddits to display (default: 20).
    min_links : int
        Minimum total hyperlinks required to include subreddit.
    figsize : tuple
        Figure size (default: (12, 6)).

    Returns
    -------
    pd.DataFrame
        DataFrame containing subreddit, negative_posts, total_posts, and negativity_ratio.
    """

    # --- Compute counts
    total_counts = df[target_col].value_counts()
    neg_counts = df.loc[df[label_col] == negative_value, target_col].value_counts()

    # --- Combine
    ratio_df = pd.DataFrame({
        "total_posts": total_counts,
        "negative_posts": neg_counts
    }).fillna(0)

    # --- Filter by minimum total posts
    ratio_df = ratio_df[ratio_df["total_posts"] > min_links]

    if ratio_df.empty:
        print(f"⚠️ No target subreddits with more than {min_links} hyperlinks found.")
        return ratio_df

    # --- Compute ratio
    ratio_df["negativity_ratio"] = ratio_df["negative_posts"] / ratio_df["total_posts"]

    # --- Sort and select top N
    ratio_df = ratio_df.sort_values("negativity_ratio", ascending=False).head(top_n)

    # --- Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=ratio_df,
        x=ratio_df.index,
        y="negativity_ratio",
        hue=ratio_df.index,
        legend=False,
        palette="Reds_r"
    )

    ax.set_title(
        f"Top {top_n} Target Subreddits by Negativity Ratio\n"
        f"(only showing those with > {min_links} total hyperlinks)",
        fontsize=13, pad=20
    )
    ax.set_xlabel("Target Subreddit")
    ax.set_ylabel("Negativity Ratio = (# Negative Posts) / (Total Posts)")
    plt.xticks(rotation=75, ha="right")

    # Annotate ratios on bars
    for p in ax.patches:
        h = p.get_height()
        if pd.notnull(h):
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=9,
                xytext=(0, 3), textcoords="offset points"
            )

    plt.tight_layout()
    plt.show()


def merge_hyperlink_data(title_df, body_df):
    """
    Merge the title and body hyperlink DataFrames into a single dataset,
    keeping ALL hyperlinks (titles + bodies).

    No deduplication between title and body datasets, since they represent
    different hyperlink occurrences.

    Returns
    -------
    merged_df : pd.DataFrame
        Combined dataset containing all hyperlinks.
    """
    # Find common columns to align both DataFrames
    common_cols = list(set(title_df.columns).intersection(set(body_df.columns)))

    # Concatenate vertically (stack all rows)
    merged_df = pd.concat([title_df[common_cols], body_df[common_cols]], axis=0, ignore_index=True)

    # Optional: ensure proper sorting by time if TIMESTAMP exists
    if "TIMESTAMP" in merged_df.columns:
        merged_df = merged_df.sort_values("TIMESTAMP").reset_index(drop=True)

    return merged_df


def filter_subreddits_by_interaction_threshold(
    df,
    source_col="SOURCE_SUBREDDIT",
    target_col="TARGET_SUBREDDIT",
    threshold=50,
    verbose=False
):
    """
    Filter out subreddits with fewer than `threshold` total interactions (in + out).

    Parameters
    ----------
    df : pd.DataFrame
        The hyperlink dataset (merged title + body).
    source_col : str
        Name of the source subreddit column.
    target_col : str
        Name of the target subreddit column.
    threshold : int
        Minimum number of total hyperlinks (in + out) required for a subreddit to be kept.
    verbose : bool
        If True, print summary statistics.

    Returns
    -------
    filtered_df : pd.DataFrame
        A new DataFrame containing only rows where both source and target
        subreddits meet the interaction threshold.
    stats : dict
        Dictionary summarizing how many subreddits and rows were removed.
    """

    # --- 1. Count total interactions (in + out) per subreddit
    out_counts = df[source_col].value_counts()
    in_counts = df[target_col].value_counts()
    total_links = out_counts.add(in_counts, fill_value=0)

    # --- 2. Identify subreddits above the threshold
    kept_subreddits = total_links[total_links >= threshold].index
    removed_subreddits = total_links[total_links < threshold].index

    # --- 3. Filter dataset
    filtered_df = df[
        df[source_col].isin(kept_subreddits) &
        df[target_col].isin(kept_subreddits)
    ].copy()

    # --- 4. Summary stats
    stats = {
        "initial_subreddits": len(total_links),
        "kept_subreddits": len(kept_subreddits),
        "removed_subreddits": len(removed_subreddits),
        "initial_rows": len(df),
        "kept_rows": len(filtered_df),
        "removed_rows": len(df) - len(filtered_df),
        "threshold": threshold
    }

    if verbose:
        print(f"Applied threshold: {threshold} interactions")
        print(f"   Total subreddits before: {stats['initial_subreddits']:,}")
        print(f"   → Kept: {stats['kept_subreddits']:,} "
              f"({100*stats['kept_subreddits']/stats['initial_subreddits']:.2f}%)")
        print(f"   → Removed: {stats['removed_subreddits']:,} "
              f"({100*stats['removed_subreddits']/stats['initial_subreddits']:.2f}%)")
        print(f"   Rows before: {stats['initial_rows']:,}, after: {stats['kept_rows']:,}")

    return filtered_df




#-----------------------------
# Linguistic property analysis
#-----------------------------


def plot_liwc_subset_means_plot(
    df: pd.DataFrame,
    liwc_subsets: dict,
    as_percent: bool = False,
    sentiment_col: str = "LINK_SENTIMENT",
    label_map: dict = {1: "Positive", -1: "Negative"},
    title: str = "LIWC Subsets by Sentiment",
    ylabel: str = "Mean score",
    figsize=(12, 6),
    rotate_xticks: int = 20
) -> None:
    """
    Version optimisée : calcule les moyennes par subset × sentiment sans créer un gros tidy_df.
    Ne retourne rien : trace seulement un barplot.
    """
    if sentiment_col not in df.columns:
        raise ValueError(f"Colonne sentiment manquante: {sentiment_col}")

    # 1) Colonnes utiles (toutes les colonnes LIWC présentes dans df)
    available_subsets = {name: [c for c in cols if c in df.columns]
                         for name, cols in liwc_subsets.items()}
    available_subsets = {k: v for k, v in available_subsets.items() if v}
    if not available_subsets:
        raise ValueError("Aucun subset LIWC valide trouvé dans le DataFrame.")

    all_cols = sorted({c for cols in available_subsets.values() for c in cols})

    # 2) Conversion numérique une seule fois
    X = df[all_cols].apply(pd.to_numeric, errors="coerce")

    # 3) Prépare les labels de sentiment
    sent_raw = df[sentiment_col]
    sent_labeled = sent_raw.map(label_map).fillna(sent_raw)

    # 4) Agrégation directe (subset × sentiment)
    rows = []
    # indices par sentiment (évite groupby coûteux)
    sent_values = sent_labeled.unique()
    sent_masks = {sv: (sent_labeled == sv).to_numpy() for sv in sent_values}

    for subset_name, cols in available_subsets.items():
        # moyenne par ligne pour le subset (rapide via numpy)
        arr = X[cols].to_numpy(dtype=float, copy=False)
        row_mean = np.nanmean(arr, axis=1)  # taille = nb_posts

        # moyenne par sentiment
        for sv in sent_values:
            mask = sent_masks[sv]
            m = np.nanmean(row_mean[mask]) if mask.any() else np.nan
            rows.append({"subset": subset_name, "sentiment": sv, "mean_score": m})

    agg_df = pd.DataFrame(rows)

    # 5) % optionnel
    if as_percent:
        agg_df["mean_score"] = agg_df["mean_score"] * 100.0
        if ylabel == "Mean score":
            ylabel = "Mean score (%)"

    # 6) Plot rapide (pas de CI)
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=agg_df,
        x="subset",
        y="mean_score",
        hue="sentiment",
        errorbar=None,
        dodge=True
    )
    ax.set_title(title)
    ax.set_xlabel("LIWC Subset")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotate_xticks, ha="right")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()
      

def plot_subset_means_by_sentiment(
    df: pd.DataFrame,
    subset_cols: list,
    sentiment_col: str = "LINK_SENTIMENT",
    label_map: dict = {1: "Positive", -1: "Negative"},
    title: str = "Average LIWC Features by Sentiment",
    xlabel: str = "Feature",
    ylabel: str = "Mean score (proportion)",
    figsize=(10, 6),
    rotate_xticks: int = 20
) -> None:
    """
    Agrège (moyenne) par sentiment pour les colonnes de `subset_cols`
    et trace un barplot. Ne retourne rien.
    """
    # Colonnes disponibles
    present = [c for c in subset_cols if c in df.columns]
    if not present:
        raise ValueError("Aucune des colonnes du subset n'est présente dans le DataFrame.")

    # Numérise et calcule les moyennes par sentiment
    X = df[present].apply(pd.to_numeric, errors="coerce")
    means = (
        pd.concat([df[[sentiment_col]], X], axis=1)
        .groupby(sentiment_col, as_index=False)[present]
        .mean()
    )

    # Labels lisibles
    means[sentiment_col] = means[sentiment_col].map(label_map).fillna(means[sentiment_col])

    # Format long pour seaborn
    long_df = means.melt(
        id_vars=sentiment_col,
        var_name="Feature",
        value_name="Mean_Score"
    )

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=long_df,
        x="Feature",
        y="Mean_Score",
        hue=sentiment_col,
        errorbar=None,
        dodge=True
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rotate_xticks, ha="right")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()

def show_inter_A_C(merge_df, sub_df,a_src="SOURCE_SUBREDDIT",  a_tgt="TARGET_SUBREDDIT", c_id="SUBREDDIT",
            labels=("A: Merged", "C: Embeddings"),
            title="Subreddit overlap: Merged vs Embeddings",
            save_path=None):
    
    # If sub_df is a set, convert it to a DF with just the subreddit names.
    if isinstance(sub_df, set):
        sub_df = pd.DataFrame({"USER_ID": sorted(sub_df)})
    
    def _normalize(series):
        return set(
            series.dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"^r\/", "", regex=True)
                .tolist())

    def subs_from_edges_autodetect(df, source_col=None, target_col=None):
        """
        Return set of subreddits from an edge-style DF.
        - If source_col/target_col are given, use them.
        - Else, auto-detect any columns containing 'subreddit' (case-insensitive).
        If 2+ found, prefer those containing 'source' and 'target'.
        If only 1 found, use just that column.
        """
        if source_col is None and target_col is None:
            cand = [c for c in df.columns if "subreddit" in c.lower()]
            if not cand:
                raise ValueError(
                    "No subreddit-like columns found in merge_df. "
                    "Look for columns containing 'subreddit' or pass a_src/a_tgt explicitly."
                )
            # Prefer explicit source/target if present
            srcs = [c for c in cand if "source" in c.lower()]
            tgts = [c for c in cand if "target" in c.lower()]

            if srcs and tgts:
                source_col = srcs[0]
                target_col = tgts[0] if tgts[0] != source_col else (tgts[1] if len(tgts) > 1 else None)
            elif len(cand) >= 2:
                source_col, target_col = cand[0], cand[1]
            else:
                source_col, target_col = cand[0], None

        src = _normalize(df[source_col]) if source_col in df.columns else set()
        tgt = _normalize(df[target_col]) if (target_col and target_col in df.columns) else set()
        return src | tgt, source_col, (target_col if target_col else None)


    def subs_from_embeddings(df, id_col=None):
        if id_col is None:
            id_col = df.columns[0]  # first column is subreddit name in your C
        return _normalize(df[id_col])


    A, used_src, used_tgt = subs_from_edges_autodetect(merge_df, a_src, a_tgt)
    C = subs_from_embeddings(sub_df, c_id)

    plt.figure(figsize=(7,7))
    venn2([A, C], set_labels=labels)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    intersection = A & C

    pct_intersection_A = len(intersection) / len(A) * 100 if len(A) > 0 else 0
    pct_intersection_C = len(intersection) / len(C) * 100 if len(C) > 0 else 0
    pct_total_overlap  = len(intersection) / len(A | C) * 100 if len(A | C) > 0 else 0
    

    #print(f"Detected A columns -> source: {used_src!r}, target: {used_tgt!r}")
    #print(f"{labels[0]} only: {len(A - C)}")
    #print(f"{labels[1]} only: {len(C - A)}")
    #print(f"Intersection: {len(A & C)}")
    #print(f"Total in {labels[0]}: {len(A)}")
    #print(f"Total in {labels[1]}: {len(C)}")
    print(f"→ Intersection = {pct_intersection_A:.2f}% of {labels[0]}")
    print(f"→ Intersection = {pct_intersection_C:.2f}% of {labels[1]}")
    #print(f"→ Overall overlap (relative to union): {pct_total_overlap:.2f}%")