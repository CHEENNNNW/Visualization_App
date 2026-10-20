"""Interactive CSV visualization dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="CSV Visualization Dashboard", layout="wide")


QUESTIONS = {
    "Histogram": [
        "What is the shape of the distribution?",
        "Are there outliers?",
        "Is the distribution symmetric or skewed?",
    ],
    "Bar chart": [
        "Which categories are largest or smallest?",
        "Are differences economically meaningful?",
        "What business decision might this comparison inform?",
    ],
    "Box plot": [
        "Do the groups differ in median?",
        "Which groups show more variability?",
        "Are there notable outliers?",
    ],
    "Scatter plot": [
        "Is there an apparent relationship between X and Y?",
        "Is the relationship linear?",
        "Are there clusters or unusual observations?",
        "Do color or size reveal additional patterns?",
    ],
}


def finite_values(series: pd.Series) -> pd.Series:
    """Return finite numeric values for reliable range defaults."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.dropna()


def range_controls(
    label: str, series: pd.Series, key: str
) -> tuple[float, float] | None:
    """Show optional numeric axis bounds initialized from the data."""
    values = finite_values(series)
    if values.empty:
        st.sidebar.caption(f"{label} range unavailable: no finite numeric values.")
        return None

    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        padding = abs(minimum) * 0.05 or 1.0
        minimum -= padding
        maximum += padding

    if not st.sidebar.checkbox(f"Set custom {label} range", key=f"{key}_enabled"):
        return None

    lower, upper = st.sidebar.columns(2)
    low = lower.number_input(
        f"{label} minimum", value=minimum, key=f"{key}_min"
    )
    high = upper.number_input(
        f"{label} maximum", value=maximum, key=f"{key}_max"
    )
    if low >= high:
        st.sidebar.warning(f"{label} minimum must be less than its maximum.")
        return None
    return (float(low), float(high))


def optional_category(
    label: str, columns: list[str], key: str
) -> str | None:
    """Render an optional categorical selector."""
    return st.sidebar.selectbox(
        label, ["None", *columns], key=key
    ) if columns else None


def build_chart(
    frame: pd.DataFrame,
    chart_type: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Render controls, prepare chart data, and display the selected chart."""
    fig = None
    if chart_type == "Histogram":
        if not numeric_columns:
            st.warning("A histogram requires at least one numeric column.")
            return

        variable = st.sidebar.selectbox("Numeric variable", numeric_columns)
        bins = st.sidebar.slider("Number of bins", 5, 100, 30)
        color = optional_category("Color/group by", categorical_columns, "hist_color")
        opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.75, 0.05)
        x_range = range_controls("X", frame[variable], "hist_x")
        y_range = range_controls("Y", pd.Series([0, len(frame)]), "hist_y")

        columns = [variable] + ([color] if color else [])
        plot_data = frame[columns].dropna(subset=columns)
        if plot_data.empty:
            st.warning("There are no complete rows for the selected histogram.")
            return
        if len(plot_data) < len(frame):
            st.caption(f"Excluded {len(frame) - len(plot_data)} row(s) with missing values.")

        fig = px.histogram(
            plot_data,
            x=variable,
            color=color,
            nbins=bins,
            opacity=opacity,
            marginal="box",
            labels={variable: variable},
            hover_data=plot_data.columns.tolist(),
        )
        if x_range:
            fig.update_xaxes(range=x_range)
        if y_range:
            fig.update_yaxes(range=y_range)
        fig.update_yaxes(title="Count")

    elif chart_type == "Bar chart":
        if not categorical_columns:
            st.warning("A bar chart requires at least one categorical/text column.")
            return

        category = st.sidebar.selectbox("Category (x-axis)", categorical_columns)
        measure_options = ["Count of observations", *numeric_columns]
        measure = st.sidebar.selectbox("Measure", measure_options)
        aggregation = None
        if measure != "Count of observations":
            aggregation = st.sidebar.selectbox(
                "Aggregation function", ["mean", "median", "sum", "count"]
            )
        color = optional_category(
            "Color/group by",
            [column for column in categorical_columns if column != category],
            "bar_color",
        )
        orientation = st.sidebar.radio(
            "Bar layout", ["Grouped", "Stacked"], horizontal=True
        )
        sort_categories = st.sidebar.checkbox("Sort categories by plotted value")

        grouping = [category] + ([color] if color else [])
        required = grouping + ([measure] if measure != "Count of observations" else [])
        plot_data = frame[required].dropna(subset=required)
        if plot_data.empty:
            st.warning("There are no complete rows for the selected bar chart.")
            return
        if len(plot_data) < len(frame):
            st.caption(f"Excluded {len(frame) - len(plot_data)} row(s) with missing values.")

        if measure == "Count of observations":
            grouped = (
                plot_data.groupby(grouping, observed=True, sort=False)
                .size()
                .rename("Value")
                .reset_index()
            )
        else:
            grouped = (
                plot_data.groupby(grouping, observed=True, sort=False)[measure]
                .agg(aggregation)
                .rename("Value")
                .reset_index()
            )

        category_order = None
        if sort_categories:
            ordering = grouped.groupby(category, observed=True)["Value"].sum()
            category_order = ordering.sort_values(ascending=False).index.tolist()

        fig = px.bar(
            grouped,
            x=category,
            y="Value",
            color=color,
            barmode="group" if orientation == "Grouped" else "stack",
            category_orders={category: category_order} if category_order else None,
            labels={
                category: category,
                "Value": (
                    "Count"
                    if measure == "Count of observations"
                    else f"{aggregation.title()} of {measure}"
                ),
            },
            hover_data=grouped.columns.tolist(),
        )

    elif chart_type == "Box plot":
        if not numeric_columns:
            st.warning("A box plot requires at least one numeric column.")
            return

        variable = st.sidebar.selectbox("Numeric variable", numeric_columns)
        category = optional_category(
            "Compare groups by", categorical_columns, "box_category"
        )
        color_options = [
            column for column in categorical_columns if column != category
        ]
        color = optional_category("Color by", color_options, "box_color")
        points = st.sidebar.selectbox(
            "Show individual points", ["None", "Outliers only", "All points"]
        )
        y_range = range_controls("Y", frame[variable], "box_y")

        columns = [variable] + ([category] if category else []) + ([color] if color else [])
        plot_data = frame[columns].dropna(subset=columns)
        if plot_data.empty:
            st.warning("There are no complete rows for the selected box plot.")
            return
        if len(plot_data) < len(frame):
            st.caption(f"Excluded {len(frame) - len(plot_data)} row(s) with missing values.")

        fig = px.box(
            plot_data,
            x=category,
            y=variable,
            color=color,
            points={"None": False, "Outliers only": "outliers", "All points": "all"}[
                points
            ],
            labels={variable: variable},
            hover_data=plot_data.columns.tolist(),
        )
        if y_range:
            fig.update_yaxes(range=y_range)

    else:
        if len(numeric_columns) < 2:
            st.warning("A scatter plot requires at least two numeric columns.")
            return

        x_variable = st.sidebar.selectbox("X-axis variable", numeric_columns)
        y_choices = [column for column in numeric_columns if column != x_variable]
        y_variable = st.sidebar.selectbox("Y-axis variable", y_choices)
        color = optional_category("Point color", list(frame.columns), "scatter_color")
        size = optional_category("Point size (numeric)", numeric_columns, "scatter_size")
        symbol = optional_category(
            "Marker style (categorical)", categorical_columns, "scatter_symbol"
        )
        hover_columns = st.sidebar.multiselect(
            "Additional hover information",
            [column for column in frame.columns if column not in (x_variable, y_variable)],
        )
        opacity = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.8, 0.05)
        x_range = range_controls("X", frame[x_variable], "scatter_x")
        y_range = range_controls("Y", frame[y_variable], "scatter_y")
        trendline = st.sidebar.checkbox("Add trend line")

        required = [x_variable, y_variable]
        for column in (color, size, symbol):
            if column and column not in required:
                required.append(column)
        plot_data = frame[required].copy()
        for column in (x_variable, y_variable, size):
            if column:
                plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        if plot_data.empty:
            st.warning("There are no complete numeric rows for the selected scatter plot.")
            return
        if len(plot_data) < len(frame):
            st.caption(f"Excluded {len(frame) - len(plot_data)} row(s) with missing or invalid values.")

        if trendline and len(plot_data) < 2:
            st.warning("At least two complete observations are required for an OLS trend line.")
            trendline = False
        if trendline:
            try:
                import statsmodels.api  # noqa: F401
            except ImportError:
                st.error(
                    "OLS trend lines require statsmodels. Install it with "
                    "`pip install statsmodels`."
                )
                return

        fig = px.scatter(
            frame.loc[plot_data.index],
            x=x_variable,
            y=y_variable,
            color=color,
            size=size,
            symbol=symbol,
            hover_data=hover_columns or None,
            opacity=opacity,
            trendline="ols" if trendline else None,
            labels={x_variable: x_variable, y_variable: y_variable},
        )
        if x_range:
            fig.update_xaxes(range=x_range)
        if y_range:
            fig.update_yaxes(range=y_range)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Questions to consider"):
            for question in QUESTIONS[chart_type]:
                st.markdown(f"- {question}")


st.title("General-Purpose Data Visualization Dashboard")
st.write("Upload a CSV file, explore its columns, and build an interactive chart.")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as error:
    st.error(f"Could not read this CSV file: {error}")
    st.stop()

if data.empty or len(data.columns) == 0:
    st.warning("The uploaded CSV does not contain any data rows or columns.")
    st.stop()

numeric_columns = data.select_dtypes(include="number").columns.tolist()
categorical_columns = [
    column for column in data.columns if column not in numeric_columns
]

# Normalize text-like data to avoid mixed Python types in Plotly categorical axes.
chart_data = data.copy()
for column in categorical_columns:
    chart_data[column] = chart_data[column].astype("string")

metric_columns = st.columns(2)
metric_columns[0].metric("Rows", f"{len(data):,}")
metric_columns[1].metric("Columns", f"{len(data.columns):,}")
st.subheader("Data preview")
st.dataframe(data.head(10), use_container_width=True)
st.caption(
    f"Numeric columns ({len(numeric_columns)}): "
    f"{', '.join(numeric_columns) if numeric_columns else 'None'}"
)
st.caption(
    f"Categorical/text columns ({len(categorical_columns)}): "
    f"{', '.join(categorical_columns) if categorical_columns else 'None'}"
)

with st.sidebar:
    st.header("Chart controls")
    selected_chart = st.selectbox("Chart type", list(QUESTIONS))

try:
    build_chart(
        chart_data, selected_chart, numeric_columns, categorical_columns
    )
except (ValueError, TypeError, KeyError, OverflowError) as error:
    st.error(
        "The selected chart could not be created from these values. "
        f"Check the selected columns and their contents. Details: {error}"
    )
