import os
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from passlib.hash import pbkdf2_sha256
from passlib.exc import InvalidHashError
from sqlalchemy import create_engine, text

load_dotenv()

APP_NAME = "Grocery Store"
ACCENT_COLOR = "#1f7a3f"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="G",
    layout="wide",
)

SERVICE_Z = {"90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}
PERIOD_FACTOR = {"Daily": 365, "Weekly": 52, "Monthly": 12}
REQUIRED_INV_COLS = {"item", "current_stock", "lead_time_units", "unit_cost"}
REQUIRED_DEM_COLS = {"item", "period", "quantity"}

DB_URL = os.getenv("DATABASE_URL")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
DEMO_EMAIL = os.getenv("DEMO_EMAIL", "demo@grocery.app")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo1234")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")

st.markdown(
    f"""
    <style>
    :root {{
        --accent: {ACCENT_COLOR};
    }}
    .block-container {{padding-top: 2rem; padding-bottom: 2rem;}}
    h1, h2, h3 {{color: var(--accent);}}
    .stMetric {{background: #f6f7f9; padding: 12px; border-radius: 8px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, required: set, name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        st.error(f"{name} is missing columns: {', '.join(sorted(missing))}")
        st.stop()


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _backtest_ma(series: np.ndarray, window: int) -> tuple[list[float], list[float]]:
    preds = []
    actuals = []
    for i in range(window, len(series)):
        preds.append(float(np.mean(series[i - window:i])))
        actuals.append(float(series[i]))
    return preds, actuals


def _backtest_es(series: np.ndarray, alpha: float) -> tuple[list[float], list[float]]:
    if len(series) < 2:
        return [], []
    level = float(series[0])
    preds = []
    actuals = []
    for i in range(1, len(series)):
        preds.append(level)
        actuals.append(float(series[i]))
        level = alpha * float(series[i]) + (1 - alpha) * level
    return preds, actuals


def _backtest_metrics(demand_df: pd.DataFrame, group_cols: list, window: int, alpha: float) -> pd.DataFrame:
    rows = []
    for _, g in demand_df.sort_values(group_cols + ["period"]).groupby(group_cols):
        series = g["quantity"].to_numpy(dtype=float)
        if len(series) < 3:
            continue
        ma_pred, ma_act = _backtest_ma(series, window)
        es_pred, es_act = _backtest_es(series, alpha)
        rows.append(
            {
                **{c: g.iloc[0][c] for c in group_cols},
                "ma_mae": _mae(np.array(ma_act), np.array(ma_pred)),
                "ma_mape": _mape(np.array(ma_act), np.array(ma_pred)),
                "es_mae": _mae(np.array(es_act), np.array(es_pred)),
                "es_mape": _mape(np.array(es_act), np.array(es_pred)),
                "n_periods": len(series),
            }
        )
    return pd.DataFrame(rows)


def _data_quality_report(inventory_df: pd.DataFrame, demand_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    rows.append({"check": "inventory_missing_values", "count": int(inventory_df.isna().sum().sum())})
    rows.append({"check": "inventory_negative_stock", "count": int((inventory_df["current_stock"] < 0).sum())})
    rows.append({"check": "inventory_nonpositive_lead_time", "count": int((inventory_df["lead_time_units"] <= 0).sum())})
    rows.append({"check": "inventory_nonpositive_unit_cost", "count": int((inventory_df["unit_cost"] <= 0).sum())})
    rows.append({"check": "inventory_duplicate_rows", "count": int(inventory_df.duplicated().sum())})
    if demand_df is not None:
        rows.append({"check": "demand_missing_values", "count": int(demand_df.isna().sum().sum())})
        rows.append({"check": "demand_negative_quantity", "count": int((demand_df["quantity"] < 0).sum())})
        rows.append({"check": "demand_duplicate_rows", "count": int(demand_df.duplicated().sum())})
    return pd.DataFrame(rows)


def _compute_forecast_ma(demand_df: pd.DataFrame, window: int, group_cols: list) -> pd.DataFrame:
    forecast = (
        demand_df.sort_values(group_cols + ["period"])
        .groupby(group_cols)
        .apply(lambda g: g["quantity"].rolling(window=window, min_periods=1).mean().iloc[-1])
        .rename("avg_demand")
        .reset_index()
    )
    return forecast


def _compute_forecast_es(demand_df: pd.DataFrame, alpha: float, group_cols: list) -> pd.DataFrame:
    def _es_last(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        level = series.iloc[0]
        for val in series.iloc[1:]:
            level = alpha * val + (1 - alpha) * level
        return float(level)

    forecast = (
        demand_df.sort_values(group_cols + ["period"])
        .groupby(group_cols)["quantity"]
        .apply(_es_last)
        .rename("avg_demand")
        .reset_index()
    )
    return forecast


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return num / denom


def _db_engine():
    if not DB_URL:
        return None
    return create_engine(DB_URL, pool_pre_ping=True)


def _init_db(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS inventory (
                    id SERIAL PRIMARY KEY,
                    store TEXT,
                    item TEXT NOT NULL,
                    current_stock NUMERIC NOT NULL,
                    lead_time_units NUMERIC NOT NULL,
                    unit_cost NUMERIC NOT NULL
                );
                CREATE TABLE IF NOT EXISTS demand (
                    id SERIAL PRIMARY KEY,
                    store TEXT,
                    item TEXT NOT NULL,
                    period NUMERIC NOT NULL,
                    quantity NUMERIC NOT NULL
                );
                """
            )
        )


def _ensure_admin(engine):
    with engine.begin() as conn:
        res = conn.execute(text("SELECT 1 FROM users WHERE email=:email"), {"email": ADMIN_EMAIL}).fetchone()
        if res is None:
            conn.execute(
                text("INSERT INTO users (email, password_hash, role) VALUES (:email, :ph, :role)"),
                {"email": ADMIN_EMAIL, "ph": pbkdf2_sha256.hash(ADMIN_PASSWORD), "role": "admin"},
            )

def _ensure_demo(engine):
    with engine.begin() as conn:
        res = conn.execute(text("SELECT 1 FROM users WHERE email=:email"), {"email": DEMO_EMAIL}).fetchone()
        if res is None:
            conn.execute(
                text("INSERT INTO users (email, password_hash, role) VALUES (:email, :ph, :role)"),
                {"email": DEMO_EMAIL, "ph": pbkdf2_sha256.hash(DEMO_PASSWORD), "role": "viewer"},
            )

def _authenticate(engine, email: str, password: str):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT email, password_hash, role FROM users WHERE email=:email"), {"email": email}).fetchone()
    if not row:
        return None
    try:
        if pbkdf2_sha256.verify(password, row.password_hash):
            return {"email": row.email, "role": row.role}
    except InvalidHashError:
        if row.email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE users SET password_hash=:ph WHERE email=:email"),
                    {"email": ADMIN_EMAIL, "ph": pbkdf2_sha256.hash(ADMIN_PASSWORD)},
                )
            return {"email": row.email, "role": row.role}
    return None


def _create_user(engine, email: str, password: str, role: str):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO users (email, password_hash, role) VALUES (:email, :ph, :role)"),
            {"email": email, "ph": pbkdf2_sha256.hash(password), "role": role},
        )

def _load_inventory(engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT store, item, current_stock, lead_time_units, unit_cost FROM inventory"), engine)


def _load_demand(engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT store, item, period, quantity FROM demand"), engine)


def _save_inventory(engine, df: pd.DataFrame) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM inventory"))
    df.to_sql("inventory", engine, if_exists="append", index=False)


def _save_demand(engine, df: pd.DataFrame) -> None:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM demand"))
    df.to_sql("demand", engine, if_exists="append", index=False)


def _send_email(subject: str, body: str, to_email: str):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_FROM]):
        raise RuntimeError("SMTP settings are missing.")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


def _send_whatsapp(body: str, to_phone: str):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM]):
        raise RuntimeError("Twilio WhatsApp settings are missing.")
    from twilio.rest import Client

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=body,
        from_=TWILIO_WHATSAPP_FROM,
        to=f"whatsapp:{to_phone}" if not to_phone.startswith("whatsapp:") else to_phone,
    )


def _build_reorder_message(df: pd.DataFrame) -> str:
    lines = ["Reorder list:"]
    for _, row in df.iterrows():
        label = f"{row.get('store', '').strip()} - {row['item']}" if "store" in df.columns else row["item"]
        lines.append(f"- {label}: stock {row['current_stock']}, ROP {round(row['reorder_point'], 2)}, EOQ {row['eoq']}")
    return "\n".join(lines)


st.title(APP_NAME)
st.write("Inventory optimization for grocery stores: forecasting, safety stock, EOQ, and reorder alerts.")

engine = _db_engine()
if engine:
    _init_db(engine)
    _ensure_admin(engine)
    _ensure_demo(engine)

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("Demand period", ["Daily", "Weekly", "Monthly"], index=1)
    service_level = st.selectbox("Service level", ["90%", "95%", "97.5%", "99%"], index=1)
    order_cost = st.number_input("Order cost (per order)", min_value=0.0, value=50.0, step=5.0)
    carrying_rate = st.number_input(
        "Annual holding cost rate",
        min_value=0.0,
        value=0.25,
        step=0.05,
        help="e.g., 0.25 = 25% of unit cost per year",
    )
    forecast_method = st.selectbox("Forecast method", ["Moving average", "Exponential smoothing"], index=0)
    ma_window = st.slider("Moving average window", min_value=2, max_value=12, value=4)
    es_alpha = st.slider("Smoothing alpha", min_value=0.05, max_value=0.95, value=0.3, step=0.05)
    auto_model = st.toggle("Auto-select best model (MAPE)", value=False)

    st.header("Notifications")
    notify_email = st.text_input("Send alerts to email")
    notify_phone = st.text_input("Send alerts to WhatsApp (e.g., +234...)")


if engine is None:
    st.warning("DATABASE_URL not set. Auth + DB features are disabled. Add a Postgres URL to enable them.")

if engine is not None:
    if "user" not in st.session_state:
        st.subheader("Login")
        login_col1, login_col2 = st.columns(2)
        with login_col1:
            email = st.text_input("Email")
        with login_col2:
            password = st.text_input("Password", type="password")
        with st.expander("Demo access"):
            st.write(f"Email: {DEMO_EMAIL}")
            st.write(f"Password: {DEMO_PASSWORD}")
        if st.button("Sign in"):
            user = _authenticate(engine, email, password)
            if user:
                st.session_state["user"] = user
                st.success("Signed in.")
            else:
                st.error("Invalid credentials.")
        st.stop()

if st.session_state.get("user", {}).get("role") == "admin":
        with st.expander("Admin: Create user"):
            new_email = st.text_input("New user email", value="godwinosayamwen@gmail.com")
            new_password = st.text_input("New user password", type="password")
            new_role = st.selectbox("Role", ["manager", "admin"], index=0)
            if st.button("Create user"):
                _create_user(engine, new_email, new_password, new_role)
                st.success("User created.")


st.subheader("1) Upload data (or use sample)")
col1, col2 = st.columns(2)
with col1:
    inv_file = st.file_uploader(
        "Inventory CSV",
        type=["csv"],
        help="Required columns: item,current_stock,lead_time_units,unit_cost",
    )
with col2:
    dem_file = st.file_uploader(
        "Demand history CSV (optional)",
        type=["csv"],
        help="Required columns: item,period,quantity",
    )

st.caption("Optional for multi-store: add a 'store' column to both files.")

use_demo = st.toggle("Use demo data (Canada stores)", value=False)
use_sample = st.toggle("Use sample data", value=False)
use_db = False
if engine is not None:
    use_db = st.toggle("Use data from database", value=False)

st.subheader("Templates")
template_col1, template_col2 = st.columns(2)
inventory_template = pd.DataFrame(
    {
        "store": ["Lagos Central", "Lagos Central", "Accra North"],
        "item": ["Rice", "Beans", "Cooking Oil"],
        "current_stock": [120, 50, 60],
        "lead_time_units": [2, 2, 3],
        "unit_cost": [1.2, 1.1, 2.5],
    }
)
demand_template = pd.DataFrame(
    {
        "store": ["Lagos Central"] * 6 + ["Accra North"] * 6,
        "item": ["Rice"] * 6 + ["Cooking Oil"] * 6,
        "period": list(range(1, 7)) * 2,
        "quantity": [30, 28, 35, 32, 31, 33, 14, 13, 15, 16, 14, 15],
    }
)
with template_col1:
    st.download_button(
        "Download inventory template",
        data=inventory_template.to_csv(index=False),
        file_name="inventory_template.csv",
        mime="text/csv",
    )
with template_col2:
    st.download_button(
        "Download demand template",
        data=demand_template.to_csv(index=False),
        file_name="demand_template.csv",
        mime="text/csv",
    )

sample_inventory = pd.DataFrame(
    {
        "store": ["Lagos Central", "Lagos Central", "Accra North", "Accra North"],
        "item": ["Rice", "Maize Flour", "Cooking Oil", "Beans"],
        "current_stock": [120, 80, 60, 50],
        "lead_time_units": [2, 1, 3, 2],
        "unit_cost": [1.2, 0.8, 2.5, 1.1],
    }
)

sample_demand = pd.DataFrame(
    {
        "store": ["Lagos Central"] * 12 + ["Accra North"] * 12,
        "item": ["Rice"] * 6
        + ["Maize Flour"] * 6
        + ["Cooking Oil"] * 6
        + ["Beans"] * 6,
        "period": list(range(1, 7)) * 4,
        "quantity": [
            30,
            28,
            35,
            32,
            31,
            33,
            18,
            20,
            19,
            22,
            21,
            23,
            14,
            13,
            15,
            16,
            14,
            15,
            12,
            11,
            13,
            12,
            14,
            13,
        ],
    }
)

demo_inventory = pd.DataFrame(
    {
        "store": [
            "Calgary East",
            "Calgary West",
            "Airdrie North",
            "Edmonton South",
            "Edmonton West",
            "Airdrie South",
        ],
        "item": [
            "Rice 5kg",
            "Maize Flour 2kg",
            "Cooking Oil 1L",
            "Beans 2kg",
            "Milk 1L",
            "Tomato Paste 400g",
        ],
        "current_stock": [110, 75, 55, 48, 62, 38],
        "lead_time_units": [2, 1, 3, 2, 1, 2],
        "unit_cost": [12.5, 4.2, 6.8, 5.5, 2.1, 2.6],
    }
)

demo_demand = pd.DataFrame(
    {
        "store": ["Calgary East"] * 4 + ["Calgary West"] * 4 + ["Airdrie North"] * 4 + ["Edmonton South"] * 4 + ["Edmonton West"] * 4 + ["Airdrie South"] * 4,
        "item": [
            "Rice 5kg",
            "Rice 5kg",
            "Rice 5kg",
            "Rice 5kg",
            "Maize Flour 2kg",
            "Maize Flour 2kg",
            "Maize Flour 2kg",
            "Maize Flour 2kg",
            "Cooking Oil 1L",
            "Cooking Oil 1L",
            "Cooking Oil 1L",
            "Cooking Oil 1L",
            "Beans 2kg",
            "Beans 2kg",
            "Beans 2kg",
            "Beans 2kg",
            "Milk 1L",
            "Milk 1L",
            "Milk 1L",
            "Milk 1L",
            "Tomato Paste 400g",
            "Tomato Paste 400g",
            "Tomato Paste 400g",
            "Tomato Paste 400g",
        ],
        "period": [1, 2, 3, 4] * 6,
        "quantity": [30, 28, 35, 32, 18, 20, 19, 22, 14, 13, 15, 16, 12, 11, 13, 12, 25, 27, 26, 28, 10, 12, 11, 13],
    }
)

if use_demo:
    inventory = demo_inventory.copy()
    demand = demo_demand.copy()
elif use_sample:
    inventory = sample_inventory.copy()
    demand = sample_demand.copy()
elif use_db:
    inventory = _load_inventory(engine)
    demand = _load_demand(engine)
else:
    if inv_file is None:
        st.warning("Upload inventory CSV or enable sample data.")
        st.stop()
    inventory = pd.read_csv(inv_file)
    demand = pd.read_csv(dem_file) if dem_file is not None else None

inventory = _clean_columns(inventory)
_require_columns(inventory, REQUIRED_INV_COLS, "Inventory CSV")

if demand is not None:
    demand = _clean_columns(demand)
    _require_columns(demand, REQUIRED_DEM_COLS, "Demand CSV")

inventory = _coerce_numeric(inventory, ["current_stock", "lead_time_units", "unit_cost"])
if demand is not None:
    demand = _coerce_numeric(demand, ["period", "quantity"])

st.subheader("1b) Data quality checks")
dq_report = _data_quality_report(inventory, demand)
st.dataframe(dq_report, use_container_width=True)

if engine is not None and not use_db:
    with st.expander("Database"):
        st.write("Save the current CSV data into the database for future use.")
        can_write = st.session_state.get("user", {}).get("role") in {"admin", "manager"}
        if st.button("Save inventory + demand to DB", disabled=not can_write):
            if not can_write:
                st.error("Demo accounts cannot write to the database.")
            elif demand is None:
                st.error("Upload demand history before saving to the database.")
            else:
                _save_inventory(engine, inventory)
                _save_demand(engine, demand)
                st.success("Saved to database.")

st.subheader("2) Filters")
if "store" in inventory.columns:
    stores = sorted(inventory["store"].dropna().unique().tolist())
    filter_store = st.multiselect("Filter by store", stores, default=stores)
    if not filter_store:
        st.warning("Select at least one store to continue.")
        st.stop()
    inventory = inventory[inventory["store"].isin(filter_store)].copy()
    if demand is not None and "store" in demand.columns:
        demand = demand[demand["store"].isin(filter_store)].copy()

st.subheader("3) Forecast demand")
if demand is not None and auto_model:
    group_cols = ["item"]
    if "store" in inventory.columns:
        group_cols = ["store", "item"]
    bt = _backtest_metrics(demand, group_cols, ma_window, es_alpha)
    if not bt.empty:
        ma_score = bt["ma_mape"].mean()
        es_score = bt["es_mape"].mean()
        if np.isfinite(es_score) and (not np.isfinite(ma_score) or es_score < ma_score):
            forecast_method = "Exponential smoothing"
        else:
            forecast_method = "Moving average"
st.caption(f"Method: {forecast_method}")

group_cols = ["item"]
if "store" in inventory.columns:
    group_cols = ["store", "item"]

if demand is None:
    st.info("No demand history uploaded. Using a default: 10 units per period.")
    forecast = inventory[group_cols].drop_duplicates().assign(avg_demand=10.0)
else:
    if forecast_method == "Moving average":
        forecast = _compute_forecast_ma(demand, ma_window, group_cols)
    else:
        forecast = _compute_forecast_es(demand, es_alpha, group_cols)

st.dataframe(forecast, use_container_width=True)

if demand is not None:
    st.subheader("3b) Backtest (model evaluation)")
    bt = _backtest_metrics(demand, group_cols, ma_window, es_alpha)
    if bt.empty:
        st.info("Not enough demand history to backtest.")
    else:
        summary = pd.DataFrame(
            {
                "model": ["Moving average", "Exponential smoothing"],
                "MAPE %": [bt["ma_mape"].mean(), bt["es_mape"].mean()],
                "MAE": [bt["ma_mae"].mean(), bt["es_mae"].mean()],
            }
        )
        st.dataframe(summary, use_container_width=True)
        st.caption("Lower MAPE/MAE is better. Auto-select uses MAPE.")

st.subheader("4) Inventory optimization")

inventory = inventory.merge(forecast, on=group_cols, how="left")
if inventory["avg_demand"].isna().any():
    inventory["avg_demand"] = inventory["avg_demand"].fillna(10.0)

inventory["annual_demand"] = inventory["avg_demand"] * PERIOD_FACTOR[period]

if demand is not None:
    demand_std = demand.groupby(group_cols)["quantity"].std().rename("demand_std").reset_index()
    inventory = inventory.merge(demand_std, on=group_cols, how="left")
    inventory["demand_std"] = inventory["demand_std"].fillna(0.0)
    demand_n = demand.groupby(group_cols)["quantity"].count().rename("demand_n").reset_index()
    inventory = inventory.merge(demand_n, on=group_cols, how="left")
    inventory["demand_n"] = inventory["demand_n"].fillna(1.0)
else:
    inventory["demand_std"] = 0.0
    inventory["demand_n"] = 1.0

z = SERVICE_Z[service_level]
lead_time = inventory["lead_time_units"].clip(lower=1)
inventory["safety_stock"] = z * inventory["demand_std"] * np.sqrt(lead_time)

inventory["reorder_point"] = inventory["avg_demand"] * inventory["lead_time_units"] + inventory["safety_stock"]

inventory["demand_ci_low"] = (inventory["avg_demand"] - z * inventory["demand_std"] / np.sqrt(inventory["demand_n"])).clip(lower=0)
inventory["demand_ci_high"] = inventory["avg_demand"] + z * inventory["demand_std"] / np.sqrt(inventory["demand_n"])

inventory["eoq"] = np.sqrt(
    _safe_div(2 * inventory["annual_demand"] * order_cost, carrying_rate * inventory["unit_cost"])
)

inventory["eoq"] = inventory["eoq"].fillna(0).replace([np.inf, -np.inf], 0).round(0)

inventory["reorder_needed"] = inventory["current_stock"] <= inventory["reorder_point"]
inventory["days_of_cover"] = _safe_div(inventory["current_stock"], inventory["avg_demand"]).round(1)
inventory["reorder_cost"] = (inventory["eoq"] * inventory["unit_cost"]).round(2)

inventory["risk"] = "OK"
inventory.loc[inventory["days_of_cover"] <= inventory["lead_time_units"], "risk"] = "At Risk"
inventory.loc[inventory["reorder_needed"], "risk"] = "Reorder"

show_cols = [
    "item",
    "current_stock",
    "avg_demand",
    "demand_ci_low",
    "demand_ci_high",
    "lead_time_units",
    "safety_stock",
    "reorder_point",
    "eoq",
    "days_of_cover",
    "reorder_cost",
    "risk",
    "reorder_needed",
]
if "store" in inventory.columns:
    show_cols = ["store"] + show_cols

st.dataframe(inventory[show_cols], use_container_width=True)

st.subheader("5) Overview")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Items", int(inventory["item"].nunique()))
with metric_cols[1]:
    st.metric("Reorder items", int(inventory["reorder_needed"].sum()))
with metric_cols[2]:
    st.metric("At risk", int((inventory["risk"] == "At Risk").sum()))
with metric_cols[3]:
    st.metric("Avg days of cover", round(float(inventory["days_of_cover"].mean()), 1))

metric_cols_2 = st.columns(2)
with metric_cols_2[0]:
    st.metric("Avg safety stock", round(float(inventory["safety_stock"].mean()), 2))
with metric_cols_2[1]:
    st.metric("Avg EOQ", round(float(inventory["eoq"].mean()), 1))

st.subheader("6) Charts")
chart_cols = st.columns(2)
chart_index = "item"
if "store" in inventory.columns:
    chart_index = "item_label"
    inventory["item_label"] = inventory["store"] + " - " + inventory["item"]

with chart_cols[0]:
    st.markdown("**Stock vs reorder point**")
    chart_df = inventory.set_index(chart_index)[["current_stock", "reorder_point"]]
    st.bar_chart(chart_df, height=280)
with chart_cols[1]:
    st.markdown("**EOQ by item**")
    eoq_df = inventory.set_index(chart_index)[["eoq"]]
    st.bar_chart(eoq_df, height=280)

if "store" in inventory.columns:
    st.subheader("7) Multi-store view")
    store_summary = (
        inventory.groupby("store")
        .agg(
            items=("item", "nunique"),
            reorder_items=("reorder_needed", "sum"),
        )
        .reset_index()
    )
    st.dataframe(store_summary, use_container_width=True)

st.subheader("8) Action list")

reorders = inventory[inventory["reorder_needed"]].copy()
reorder_cols = ["item", "current_stock", "reorder_point", "eoq"]
if "store" in inventory.columns:
    reorder_cols = ["store"] + reorder_cols

if reorders.empty:
    st.success("No items need reordering right now.")
else:
    st.warning("Reorder these items:")
    st.dataframe(reorders[reorder_cols], use_container_width=True)

st.subheader("9) Export")
if reorders.empty:
    st.info("No reorder items to export yet.")
else:
    st.download_button(
        "Download reorder list (CSV)",
        data=reorders[reorder_cols].to_csv(index=False),
        file_name="reorder_list.csv",
        mime="text/csv",
    )

if not reorders.empty and (notify_email or notify_phone):
    st.subheader("10) Send alerts")
    message = _build_reorder_message(reorders)
    if notify_email and st.button("Send Email Alerts"):
        try:
            _send_email("Reorder Alerts", message, notify_email)
            st.success("Email alert sent.")
        except Exception as exc:
            st.error(f"Email failed: {exc}")
    if notify_phone and st.button("Send WhatsApp Alerts"):
        try:
            _send_whatsapp(message, notify_phone)
            st.success("WhatsApp alert sent.")
        except Exception as exc:
            st.error(f"WhatsApp failed: {exc}")

st.caption("Tip: Upload real demand history to improve forecasts and safety stock calculations.")
