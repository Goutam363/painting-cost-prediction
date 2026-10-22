import datetime as dt
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "linear_regression_model.joblib"
DATA_PATH = BASE_DIR / "painting_dataset_cleaned.csv"

st.set_page_config(
    page_title="Painting Cost Predictor",
    page_icon="🎨",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["project_start_date"]
    )

    return df[
        [
            "item_name",
            "quantity",
            "building_type",
            "city",
            "state",
            "project_start_date",
            "amount_inr"
        ]
    ]


model = load_model()
df = load_data()

items = sorted(df["item_name"].dropna().unique())
building_types = sorted(df["building_type"].dropna().unique())
states = sorted(df["state"].dropna().unique())

DEFAULT_ITEM = "Interior emulsion paint (2 coats)"
DEFAULT_BUILDING = "Residential - Independent House"
DEFAULT_QUANTITY = 1600.0
DEFAULT_STATE = "West Bengal"
DEFAULT_CITY = "Kolkata"

def get_default_index(options, default_value):
    """Return the index of the requested default, or 0 if unavailable."""
    options = list(options)

    if default_value in options:
        return options.index(default_value)

    return 0

st.title("🎨 Painting Work Cost Predictor")

st.write(
    "Fill in the project details and click **Predict** "
    "to estimate the painting cost."
)

col1, col2 = st.columns(2)

with col1:
    item_name = st.selectbox(
        "Painting item",
        items,
        index=get_default_index(items, DEFAULT_ITEM)
    )

    building_type = st.selectbox(
        "Building type",
        building_types,
        index=get_default_index(building_types, DEFAULT_BUILDING)
    )

    state = st.selectbox(
        "State",
        states,
        index=get_default_index(states, DEFAULT_STATE)
    )

    cities_in_state = sorted(
        df.loc[
            df["state"] == state, "city"
        ].dropna().unique()
    )

    city = st.selectbox(
        "City",
        cities_in_state,
        index=get_default_index(cities_in_state, DEFAULT_CITY)
    )

with col2:
    q_min = float(df["quantity"].min())
    q_max = float(df["quantity"].max())
    default_quantity = min(max(DEFAULT_QUANTITY, q_min), q_max)

    quantity = st.number_input(
        "Quantity or Wall area (sqm)",
        min_value=max(1.0, q_min),
        max_value=q_max,
        value=default_quantity,
        step=100.0,
        help=(
            f"Typical range in training data: "
            f"{q_min:,.0f} to {q_max:,.0f}"
        )
    )

    start_date = st.date_input(
        "Project start date",
        value=dt.date.today()
    )

if quantity > q_max or quantity < q_min:
    st.warning(
        "This quantity is outside the range the model was trained on, "
        "so the prediction may be unreliable."
    )

if DEFAULT_ITEM not in items:
    st.warning(
        f'The default painting item "{DEFAULT_ITEM}" was not found '
        "in the dataset. The first available item is selected instead."
    )

if DEFAULT_BUILDING not in building_types:
    st.warning(
        f'The default building type "{DEFAULT_BUILDING}" was not found '
        "in the dataset. The first available building type is selected instead."
    )

if DEFAULT_STATE not in states:
    st.warning(
        f'The default state "{DEFAULT_STATE}" was not found '
        "in the dataset. The first available state is selected instead."
    )

if DEFAULT_CITY not in cities_in_state and state == DEFAULT_STATE:
    st.warning(
        f'The default city "{DEFAULT_CITY}" was not found '
        f'under "{DEFAULT_STATE}". The first available city is selected instead.'
    )

def build_input() -> pd.DataFrame:
    """
    Create a one-row DataFrame containing the features
    expected by the trained Linear Regression pipeline.
    """

    return pd.DataFrame([
        {
            "item_name": item_name,
            "quantity": quantity,
            "building_type": building_type,
            "city": city,
            "state": state,
            "project_year": start_date.year,
            "project_month": start_date.month,
        }
    ])

def predict() -> float:
    input_data = build_input()
    prediction = float(model.predict(input_data)[0])
    return prediction

if st.button("Predict cost", type="primary"):

    try:
        cost = predict()

        if cost < 0:
            st.warning(
                "The model predicted a negative cost. "
                "Please check the model and input data."
            )
        else:
            st.success(f"Estimated cost: ₹ {cost:,.2f}")

            st.metric(
                "Cost per unit",
                f"₹ {cost / quantity:,.2f}"
            )

            st.caption(
                "Prediction is an estimate based on the trained model "
                "and the selected project details."
            )

    except Exception as e:
        st.error(
            "Prediction failed. Please check whether the input features "
            "match those used when training the model."
        )
        st.exception(e)

with st.expander("Show sample of training data"):
    st.dataframe(df.head(20))