from __future__ import annotations

import html
import math
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="KKBOX · Churn Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

OBS_START = "2017-02-01"
OBS_END = "2017-02-28"
NAV_ITEMS = ("Overview", "Users", "User Analysis")
COLOR = {
    "bg": "#0B1120",
    "panel": "#131F33",
    "border": "#293952",
    "text": "#EDF2FD",
    "muted": "#93A5BE",
    "violet": "#8B83FF",
    "teal": "#3DD8C7",
    "coral": "#FF7E98",
    "blue": "#6BA9FF",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #0B1120 !important;
        color: #EDF2FD;
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
    }
    .block-container {max-width: 1460px; padding-top: 2rem; padding-bottom: 2.4rem;}
    [data-testid="stHeader"] {background: transparent !important;}
    [data-testid="stSidebar"] {
        background: #0E1729 !important;
        border-right: 1px solid #253450;
    }
    [data-testid="stSidebar"] > div:first-child {padding-top: 1.35rem;}
    h1, h2, h3, h4 {font-family: 'Space Grotesk', 'Segoe UI', sans-serif !important; letter-spacing: -0.035em; color: #EFF4FF !important;}
    p, label, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] {color: inherit;}
    [data-testid="stCaptionContainer"] {color: #9BACBF !important;}
    [data-testid="stVerticalBlockBorderWrapper"] > div:has(> [data-testid="stVerticalBlock"]) {
        border-radius: 17px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div) {border-color: #283750 !important;}
    .stButton > button {
        border-radius: 11px !important;
        font-weight: 700;
        min-height: 2.65rem;
        border: 1px solid #2A3952;
        transition: all .15s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(125deg, #756EF7, #615BDD) !important;
        border: 1px solid #827BF7 !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(104,94,240,.15);
    }
    .stButton > button[kind="secondary"] {background: #17253A !important; color: #C6D6E9 !important;}
    .stButton > button:hover {border-color: #8B83FF !important; transform: translateY(-1px);}
    [data-testid="stSidebar"] .stButton > button {text-align: left; justify-content: flex-start; padding-left: 1rem;}
    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stNumberInput"] input {
        border-color: #354664 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stTextInput"] input {background: #132036 !important; color: #EDF2FD !important;}
    div[data-testid="stExpander"] {border-radius: 14px !important; border-color: #2B3B54 !important;}
    [data-testid="stAlert"] {border-radius: 12px !important;}
    [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden; border: 1px solid #2D3B52;}
    hr {border-color: #253650 !important;}
    .side-brand {margin: .05rem 0 1.9rem;}
    .side-brand-head {display:flex; align-items:center; gap:.8rem;}
    .brand-symbol {width:40px; height:40px; border-radius:13px; display:flex;
        align-items:center; justify-content:center; font-size:24px; color:#FFF;
        background:linear-gradient(145deg,#857EFF,#4945B7); box-shadow:0 5px 22px #5548C733;}
    .brand-name {font: 700 1.19rem 'Space Grotesk', sans-serif; color:#F6F7FF; line-height:1.2; letter-spacing:-.02em;}
    .brand-desc {font-size:.75rem; color:#8FA5C0; margin-top:.2rem;}
    .small-label {font-size:.68rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#7F95B2; margin-bottom:.6rem;}
    .side-foot {font-size:.75rem; color:#91A3BD; padding: 1rem .15rem; line-height:1.75;}
    .live-dot {display:inline-block; width:7px; height:7px; border-radius:50%; background:#3DD8C7; margin-right:6px; box-shadow:0 0 0 3px #3DD8C722;}
    .eyebrow {font-size:.73rem; color:#A39CFF; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.55rem;}
    .page-head {padding:.45rem .15rem .6rem;}
    .page-head h1 {font-size:2.45rem !important; line-height:1.15; margin:0 0 .38rem; font-weight:700;}
    .page-chip {display:inline-block; border-radius:40px; border:1px solid #2C4162;
        background:#17243A; padding:6px 11px; color:#B7C8E0; font-size:.76rem; font-weight:650;}
    .page-rule {height:1px; background:linear-gradient(to right,#374669,#26385100); margin:1rem 0 1.6rem;}
    .kpi {background:linear-gradient(145deg,#18253A 0%,#121E30 100%);
        border:1px solid #2A3B56; border-radius:17px; min-height:148px;
        padding:1.2rem 1.26rem 1rem; position:relative; overflow:hidden; box-shadow:0 9px 28px #03081518;}
    .kpi:before {content:''; position:absolute; right:-32px; top:-43px; width:118px; height:118px;
        background:var(--glow); border-radius:50%; filter:blur(28px); opacity:.19;}
    .kpi-top {display:flex; align-items:center; justify-content:space-between; position:relative;}
    .kpi-label {font-size:.79rem; color:#A7B8CE; font-weight:650;}
    .kpi-icon {width:29px; height:29px; display:flex; align-items:center; justify-content:center;
        font-size:16px; border-radius:9px; color:var(--accent); background:var(--tint);}
    .kpi-value {font:700 2.05rem 'Space Grotesk', sans-serif; color:#F4F8FF; margin:.76rem 0 .24rem; letter-spacing:-.05em; line-height:1.14;}
    .panel-head {display:flex; align-items:center; justify-content:space-between; padding:.12rem 0 .72rem;}
    .panel-title {font: 650 1.02rem 'Space Grotesk', sans-serif; color:#E9F0FF;}
    .panel-note {font-size:.72rem; color:#889EBB;}
    .microtag {display:inline-flex; padding:.34rem .65rem; border:1px solid #344762;
        background:#1A2A43; border-radius:40px; color:#B4C8E5; font-size:.72rem; font-weight:700;}
    .mini-split {display:flex;justify-content:space-between;align-items:center;
        margin:.72rem 0 .26rem; font-size:.84rem;}
    .mini-left {display:flex; gap:8px; align-items:center; color:#C3D2E6;}
    .mini-dot {width:8px;height:8px;border-radius:50%;display:inline-block;}
    .mini-value {color:#F1F5FF;font-weight:750;font-family:'Space Grotesk',sans-serif;}
    .track {width:100%;height:8px; background:#253750;border-radius:40px; overflow:hidden;margin:.3rem 0 .65rem;}
    .fill {height:100%; border-radius:40px;}
    .summary-strip {background:#1A2A43; border:1px solid #344763; border-radius:14px; padding:1rem 1.2rem;}
    .user-key {font-family:'Space Grotesk',monospace; color:#C5C2FF; overflow-wrap:anywhere;}
    .status {display:inline-flex; align-items:center; border-radius:30px;
        font-weight:750;font-size:.76rem;padding:.4rem .78rem;}
    .status-risk {color:#FF9CAF;background:#FF7E9816;border:1px solid #FF7E9844;}
    .status-safe {color:#59E5D7;background:#3DD8C716;border:1px solid #3DD8C744;}
    .section-heading {font:700 1.28rem 'Space Grotesk',sans-serif;color:#E8EFFC;
        margin:1.8rem 0 .25rem;}
    .footnote {margin-top:2rem; padding-top:.85rem; border-top:1px solid #283850;
        color:#7188A7;font-size:.73rem;display:flex;justify-content:space-between;}
    @media(max-width: 800px) {.page-head h1{font-size:1.85rem!important;}.kpi{min-height:123px;}.kpi-value{font-size:1.6rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

for state_key, initial_value in {
    "current_page": NAV_ITEMS[0],
    "users_page_number": 1,
    "users_filters": {
        "prediction": None,
        "min_score": None,
        "max_score": None,
        "sort_by": "churn_score",
        "order": "desc",
        "limit": 50,
    },
    "lookup_msno": "",
    "analysis_msno_input": "",
    "active_analysis_msno": "",
    "load_full_distribution": False,
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_value


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def api_request(
    base_url: str,
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 25,
) -> dict[str, Any]:
    try:
        response = requests.request(
            method=method,
            url=f"{base_url.rstrip('/')}{endpoint}",
            params=params,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Cannot connect to the API: {exc}") from exc

    if not response.ok:
        messages = {
            404: "No matching user or saved prediction was found.",
            422: "Invalid request. Check the input values.",
            503: "The backend service is temporarily unavailable.",
        }
        message = messages.get(response.status_code, f"Backend returned HTTP {response.status_code}.")
        raise ApiError(message, response.status_code)

    try:
        data = response.json()
    except ValueError as exc:
        raise ApiError(f"The backend returned invalid JSON ({endpoint}).") from exc
    if not isinstance(data, dict):
        raise ApiError(f"Unexpected API response format ({endpoint}).")
    return data


def observation_params() -> dict[str, str]:
    return {"observation_start": OBS_START, "observation_end": OBS_END}


@st.cache_data(ttl=90, show_spinner=False)
def fetch_summary(base_url: str) -> dict[str, Any]:
    return api_request(base_url, "GET", "/predictions/summary", params=observation_params())


@st.cache_data(ttl=90, show_spinner=False)
def fetch_predictions_page(base_url: str, page: int, filters: tuple) -> dict[str, Any]:
    prediction, min_score, max_score, sort_by, order, limit = filters
    params: dict[str, Any] = observation_params()
    params.update({"sort_by": sort_by, "order": order, "limit": limit,
                   "offset": (page - 1) * limit})
    if prediction is not None:
        params["prediction"] = prediction
    if min_score is not None:
        params["min_score"] = min_score
    if max_score is not None:
        params["max_score"] = max_score
    return api_request(base_url, "GET", "/predictions", params=params)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_user_saved(base_url: str, msno: str) -> dict[str, Any]:
    return api_request(base_url, "GET", "/predictions/user",
                       params={**observation_params(), "msno": msno})


@st.cache_data(ttl=300, show_spinner=False)
def fetch_user_explanation(base_url: str, msno: str, top_k: int) -> dict[str, Any]:
    return api_request(base_url, "GET", "/explain/user",
                       params={**observation_params(), "msno": msno, "top_k": top_k},
                       timeout=45)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_full_distribution(base_url: str) -> list[dict[str, Any]]:
    """Request all rows in 200-item API pages, only after explicit user action."""
    rows: list[dict[str, Any]] = []
    offset = 0
    expected_total: int | None = None
    while expected_total is None or offset < expected_total:
        page = api_request(
            base_url, "GET", "/predictions",
            params={**observation_params(), "limit": 200, "offset": offset,
                    "sort_by": "msno", "order": "asc"},
        )
        if expected_total is None:
            expected_total = int(page["total"])
        current_items = page.get("items", [])
        if not current_items:
            if offset < expected_total:
                raise ApiError("The API returned an empty page before loading all records.")
            break
        rows.extend({"churn_score": x["churn_score"], "prediction": x["prediction"]}
                    for x in current_items)
        offset += len(current_items)
    if expected_total is not None and len(rows) != expected_total:
        raise ApiError("The fetched record count does not match the reported total. Refresh and retry.")
    return rows


def show_api_error(error: ApiError) -> None:
    if error.status_code == 404:
        st.warning(str(error))
    elif error.status_code == 422:
        st.error(str(error))
    else:
        st.error(str(error))


def percent(value: Any, precision: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{precision}%}"


def count_text(value: Any) -> str:
    return f"{int(value):,}"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def kpi_card(label: str, value: str, icon: str, accent: str) -> None:
    tint = accent + "1A" 
    st.markdown(
        f'<div class="kpi" style="--accent:{accent};--glow:{accent};--tint:{tint}">'
        f'<div class="kpi-top"><span class="kpi-label">{escape(label)}</span>'
        f'<span class="kpi-icon">{escape(icon)}</span></div>'
        f'<div class="kpi-value">{escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, tag: str = "SNAPSHOT · FEB 2017") -> None:
    st.markdown(
        f'<div class="page-head"><div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<span class="page-chip">{escape(tag)}</span>', unsafe_allow_html=True)
    st.markdown('<div class="page-rule"></div>', unsafe_allow_html=True)


def section_heading(title: str) -> None:
    st.markdown(f'<div class="section-heading">{escape(title)}</div>', unsafe_allow_html=True)


def panel_header(title: str, note: str = "") -> None:
    st.markdown(
        '<div class="panel-head">'
        f'<span class="panel-title">{escape(title)}</span>'
        f'<span class="panel-note">{escape(note)}</span></div>',
        unsafe_allow_html=True,
    )


def mini_bar(label: str, value: int, total: int, color: str) -> None:
    pct = 100.0 * value / max(total, 1)
    st.markdown(
        f'<div class="mini-split"><span class="mini-left">'
        f'<span class="mini-dot" style="background:{color}"></span>{escape(label)}</span>'
        f'<span class="mini-value">{count_text(value)} · {pct:.1f}%</span></div>'
        f'<div class="track"><div class="fill" style="background:{color};width:{pct:.2f}%"></div></div>',
        unsafe_allow_html=True,
    )


def style_plot(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Segoe UI, sans-serif", color="#C1CFE4", size=12),
        margin=dict(l=12, r=14, t=10, b=12),
        hoverlabel=dict(bgcolor="#243652", font=dict(color="#EDF2FD", size=12)),
    )
    return fig


FEATURE_LABELS_EN = {
    "active_days": "Active days",
    "total_secs": "Total listening time",
    "num_100": "Nearly completed plays",
    "num_25": "Short plays (under 25%)",
    "num_unq": "Distinct-song plays",
    "secs_first_half": "Listening time, first half",
    "secs_second_half": "Listening time, second half",
    "city": "City code",
    "age": "Age",
    "registered_via": "Registration channel",
    "tx_count": "Transaction count",
    "total_paid": "Total payments",
    "auto_renew_ratio": "Auto-renewal rate",
    "cancel_count": "Cancellation count",
    "decay_ratio": "Listening time trend",
    "skip_ratio": "Short-play ratio",
    "avg_secs_per_active_day": "Listening time per active day",
    "is_age_missing": "Missing age indicator",
}

def feature_name_en(feature: str) -> str:
    return FEATURE_LABELS_EN.get(feature, feature.replace("_", " ").capitalize())


def move_to_analysis(msno: str) -> None:
    st.session_state["analysis_msno_input"] = msno
    st.session_state["active_analysis_msno"] = msno
    st.session_state["current_page"] = "User Analysis"
    st.rerun()


with st.sidebar:
    st.markdown(
        '<div class="side-brand"><div class="side-brand-head">'
        '<span class="brand-symbol">◉</span><div>'
        '<div class="brand-name">KKBOX Intelligence</div>'
        '<div class="brand-desc">Churn analytics workspace</div></div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="small-label">WORKSPACE</div>', unsafe_allow_html=True)
    for item, icon in (("Overview", "◫"), ("Users", "♙"), ("User Analysis", "◇")):
        if st.button(
            f"{icon}    {item}",
            key=f"nav_{item}",
            use_container_width=True,
            type="primary" if st.session_state["current_page"] == item else "secondary",
        ):
            st.session_state["current_page"] = item
            st.rerun()
    st.divider()
    st.markdown('<div class="small-label">CONNECTION</div>', unsafe_allow_html=True)
    base_url = st.text_input(
        "FastAPI URL",
        value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
    ).strip().rstrip("/")
    if st.button("↻    Refresh data", key="refresh", use_container_width=True):
        st.cache_data.clear()
        st.session_state["load_full_distribution"] = False
        st.rerun()
    st.markdown(
        '<div class="side-foot">'
        '<span class="live-dot"></span>API CONNECTED<br>'
        'Observation · Feb 01–28, 2017<br>'
        'Prediction · Mar 01–31, 2017<br><br>'
        'XGBoost + SHAP · Thesis prototype'
        '</div>',
        unsafe_allow_html=True,
    )

try:
    if not base_url.startswith(("http://", "https://")):
        raise ApiError("API URL must start with http:// or https://")
    api_request(base_url, "GET", "/health", timeout=5)
except ApiError as exc:
    st.error(f"Cannot connect to the backend. {exc}")
    st.code("python -m uvicorn backend.main:app --reload", language="bash")
    st.stop()


def render_overview() -> None:
    page_header(
        "ANALYTICS / POPULATION",
        "Customer churn overview",
    )
    try:
        with st.spinner("Loading overview..."):
            summary = fetch_summary(base_url)
    except ApiError as exc:
        show_api_error(exc)
        return

    total = int(summary["total_users"])
    churn = int(summary["predicted_churn"])
    non_churn = int(summary["predicted_non_churn"])
    churn_rate = float(summary["churn_rate"])
    average = summary.get("avg_churn_score")

    a, b, c, d = st.columns(4, gap="medium")
    with a:
        kpi_card("Total users", count_text(total), "◉", COLOR["blue"])
    with b:
        kpi_card("Predicted churn", count_text(churn), "↗", COLOR["coral"])
    with c:
        kpi_card("Predicted non-churn", count_text(non_churn), "✓", COLOR["teal"])
    with d:
        kpi_card("Predicted churn rate", percent(churn_rate), "%", COLOR["violet"])

    if total <= 0:
        st.info("No predictions are available for this observation period.")
        return

    section_heading("Population insights")
    left, right = st.columns([1.15, 0.85], gap="medium")
    with left:
        with st.container(border=True):
            panel_header("Churn composition", "ALL USERS")
            donut = go.Figure(go.Pie(
                labels=["Churn", "Non-Churn"],
                values=[churn, non_churn],
                hole=.77,
                marker=dict(colors=[COLOR["coral"], COLOR["teal"]],
                            line=dict(color=COLOR["panel"], width=4)),
                hovertemplate="%{label}<br>%{value:,} users · %{percent}<extra></extra>",
                textinfo="none", sort=False,
            ))
            style_plot(donut, height=322)
            donut.update_layout(
                showlegend=True,
                legend=dict(orientation="h", x=.5, xanchor="center", y=-.02,
                            font=dict(size=12, color=COLOR["muted"])),
                annotations=[dict(text=f"<b>{churn_rate:.1%}</b><br><span style='font-size:13px'>CHURN RATE</span>",
                                  x=.5, y=.5, showarrow=False,
                                  font=dict(color=COLOR["text"], size=24))],
            )
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            panel_header("Audience breakdown", "PREDICTIONS")
            st.markdown("<div style='height:9px'></div>", unsafe_allow_html=True)
            mini_bar("Churn", churn, total, COLOR["coral"])
            mini_bar("Non-Churn", non_churn, total, COLOR["teal"])
            st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="small-label">AVERAGE RISK SCORE</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font:700 2.05rem Space Grotesk,sans-serif;color:#EFF4FF">'
                f'{escape(percent(average))}</div>', unsafe_allow_html=True,
            )
            avg_width = 0 if average is None else max(0, min(float(average), 1)) * 100
            st.markdown(
                f'<div class="track" style="margin-top:10px;height:10px">'
                f'<div class="fill" style="background:linear-gradient(90deg,#7775EF,#9B95FF);width:{avg_width:.1f}%"></div>'
                '</div>', unsafe_allow_html=True,
            )

    section_heading("Risk score distribution")
    with st.container(border=True):
        head, action = st.columns([4, 1.5], vertical_alignment="center")
        with head:
            panel_header("Score distribution", "0% — 100%")
        with action:
            if st.button("↧  Load distribution", type="primary", use_container_width=True):
                st.session_state["load_full_distribution"] = True
        if st.session_state["load_full_distribution"]:
            try:
                with st.spinner("Loading risk scores..."):
                    rows = fetch_full_distribution(base_url)
                if rows:
                    scores = pd.Series([float(x["churn_score"]) for x in rows])
                    hist = go.Figure(go.Histogram(
                        x=scores, xbins=dict(start=0, end=1.00001, size=.05),
                        marker=dict(color=COLOR["violet"], line=dict(color="#C3BDFF", width=.3)),
                        hovertemplate="Risk score: %{x:.0%}<br>Users: %{y:,}<extra></extra>",
                    ))
                    style_plot(hist, height=306)
                    hist.update_layout(bargap=.16)
                    hist.update_xaxes(title="Churn score", tickformat=".0%", range=[0, 1],
                                      showgrid=False, zeroline=False)
                    hist.update_yaxes(title="Users", gridcolor="#293950", zeroline=False)
                    st.plotly_chart(hist, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No risk scores are available.")
            except ApiError as exc:
                show_api_error(exc)


def render_users() -> None:
    page_header(
        "EXPLORER / PEOPLE",
        "User explorer",
    )

    current = st.session_state["users_filters"]
    label_to_prediction = {"All": None, "Churn": 1, "Non-Churn": 0}
    prediction_to_label = {None: "All", 1: "Churn", 0: "Non-Churn"}
    sort_mapping = {"Risk score": "churn_score", "MSNO": "msno", "Predicted at": "predicted_at"}
    with st.container(border=True):
        panel_header("Filter & sort", "SERVER-SIDE")
        with st.form("users_filter_form"):
            f1, f2, f3 = st.columns([1.1, 1.1, 1.2], gap="medium")
            with f1:
                status = st.selectbox(
                    "Status", list(label_to_prediction.keys()),
                    index=list(label_to_prediction).index(prediction_to_label.get(current["prediction"], "All")),
                )
            with f2:
                sort_name = next((label for label, val in sort_mapping.items() if val == current["sort_by"]), "Risk score")
                sort_label = st.selectbox("Sort by", list(sort_mapping), index=list(sort_mapping).index(sort_name))
            with f3:
                size_options = [25, 50, 100, 200]
                size = st.selectbox("Rows per page", size_options,
                                    index=size_options.index(current["limit"]) if current["limit"] in size_options else 1)
            f4, f5, f6 = st.columns([2.2, .9, 1.1], vertical_alignment="bottom")
            with f4:
                low = 0.0 if current["min_score"] is None else float(current["min_score"])
                high = 1.0 if current["max_score"] is None else float(current["max_score"])
                score_range = st.slider("Risk score range", 0.0, 1.0, (low, high), step=.05)
            with f5:
                descending = st.checkbox("Descending", value=current["order"] == "desc")
            with f6:
                applied = st.form_submit_button("Apply filters", type="primary", use_container_width=True)
            if applied:
                st.session_state["users_filters"] = {
                    "prediction": label_to_prediction[status],
                    "min_score": None if score_range[0] == 0 else score_range[0],
                    "max_score": None if score_range[1] == 1 else score_range[1],
                    "sort_by": sort_mapping[sort_label],
                    "order": "desc" if descending else "asc",
                    "limit": size,
                }
                st.session_state["users_page_number"] = 1
                st.rerun()

    active = st.session_state["users_filters"]
    filters = (active["prediction"], active["min_score"], active["max_score"],
               active["sort_by"], active["order"], active["limit"])
    page_number = st.session_state["users_page_number"]
    try:
        with st.spinner("Loading users..."):
            response = fetch_predictions_page(base_url, page_number, filters)
    except ApiError as exc:
        show_api_error(exc)
        return
    total = int(response.get("total", 0))
    pages = max(1, math.ceil(total / active["limit"]))
    if page_number > pages:
        st.session_state["users_page_number"] = pages
        st.rerun()

    section_heading("Prediction records")
    with st.container(border=True):
        title_row, page_row = st.columns([3, 1], vertical_alignment="center")
        with title_row:
            panel_header("All predictions", f"{count_text(total)} USERS")
        with page_row:
            st.caption(f"Page {page_number} / {pages}")
        items = response.get("items", [])
        if items:
            frame = pd.DataFrame([{
                "MSNO": item["msno"],
                "Churn score (%)": float(item["churn_score"]) * 100,
                "Status": "Churn" if int(item["prediction"]) else "Non-Churn",
                "Threshold (%)": float(item["threshold"]) * 100,
                "Predicted at": item.get("predicted_at", ""),
            } for item in items])
            table_kwargs = dict(
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MSNO": st.column_config.TextColumn("MSNO", width="large"),
                    "Churn score (%)": st.column_config.ProgressColumn(
                        "Churn score (%)", format="%.1f%%", min_value=0, max_value=100),
                    "Threshold (%)": st.column_config.NumberColumn("Threshold (%)", format="%.1f"),
                },
            )
            selected_msno = ""
            try:
                selection = st.dataframe(
                    frame, on_select="rerun", selection_mode="single-row",
                    key=f"prediction_records_table_{page_number}_{hash(filters)}", **table_kwargs,
                )
                selected_rows = selection.selection.rows
                if selected_rows:
                    row_idx = selected_rows[0]
                    if 0 <= row_idx < len(frame):
                        selected_msno = str(frame.iloc[row_idx]["MSNO"])
            except TypeError: 
                st.dataframe(frame, **table_kwargs)
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            action_spacer, action_col = st.columns([3.5, 1.2])
            with action_col:
                if st.button(
                    "Analyze selected user →", type="primary", use_container_width=True,
                    disabled=not bool(selected_msno),
                ):
                    move_to_analysis(selected_msno)
        else:
            st.info("No users match the selected filters.")

    prev, mid, nxt = st.columns([1, 2, 1])
    with prev:
        if st.button("← Previous", disabled=page_number <= 1, use_container_width=True):
            st.session_state["users_page_number"] -= 1
            st.rerun()
    with mid:
        st.markdown(
            f'<p style="text-align:center;color:#A7B8CE;padding-top:9px;font-size:.83rem">'
            f'Page {page_number} of {pages} · {count_text(total)} records</p>',
            unsafe_allow_html=True,
        )
    with nxt:
        if st.button("Next →", disabled=page_number >= pages, use_container_width=True):
            st.session_state["users_page_number"] += 1
            st.rerun()

    section_heading("Quick lookup")
    with st.container(border=True):
        with st.form("lookup_user_form"):
            a, b = st.columns([4, 1.2], vertical_alignment="bottom")
            with a:
                text = st.text_input("MSNO", placeholder="Enter MSNO")
            with b:
                submitted = st.form_submit_button("Find user", type="primary", use_container_width=True)
        if submitted:
            st.session_state["lookup_msno"] = text.strip()
        msno = st.session_state["lookup_msno"]
        if msno:
            try:
                result = fetch_user_saved(base_url, msno)
                name = escape(result["msno"])
                st.markdown(
                    f'<div class="summary-strip"><b class="user-key">{name}</b> &nbsp; '
                    f'Risk score <b>{percent(result["churn_score"],2)}</b> &nbsp;·&nbsp; '
                    f'{"Churn" if int(result["prediction"]) else "Non-Churn"}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Open User Analysis ↗", type="secondary"):
                    move_to_analysis(msno)
            except ApiError as exc:
                show_api_error(exc)


def render_user_analysis() -> None:
    page_header(
        "EXPLAINABILITY / USER",
        "Individual user analysis",
    )

    with st.container(border=True):
        panel_header("Select a customer", "MSNO")
        with st.form("analysis_form"):
            inp, action = st.columns([4, 1.25], vertical_alignment="bottom")
            with inp:
                st.text_input("MSNO", key="analysis_msno_input",
                              placeholder="Enter MSNO")
            with action:
                submitted = st.form_submit_button("Analyze →", type="primary", use_container_width=True)
        if submitted:
            st.session_state["active_analysis_msno"] = st.session_state["analysis_msno_input"].strip()

    msno = st.session_state["active_analysis_msno"]
    if not msno:
        return

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin:1.4rem 0 .9rem">'
        '<span class="microtag">SELECTED USER</span>'
        f'<span class="user-key">{escape(msno)}</span></div>',
        unsafe_allow_html=True,
    )

    saved: dict[str, Any] | None = None
    saved_error: ApiError | None = None
    try:
        saved = fetch_user_saved(base_url, msno)
    except ApiError as exc:
        saved_error = exc
        if exc.status_code != 404:
            st.warning(f"Could not load saved prediction: {exc}")

    explanation: dict[str, Any] | None = None
    try:
        with st.spinner("Loading SHAP analysis..."):
            explanation = fetch_user_explanation(base_url, msno, 8)
    except ApiError as exc:
        if saved is None:
            show_api_error(exc)
        else:
            st.warning(f"Prediction loaded, but SHAP analysis is unavailable: {exc}")

    if saved is None and explanation is None:
        if saved_error is not None and saved_error.status_code == 404:
            st.info("No saved prediction was found for this MSNO.")
        return

    output = explanation if explanation is not None else saved
    assert output is not None
    score = float(output["churn_score"])
    threshold = float(output["threshold"])
    prediction = int(output["prediction"])
    risk_label = "Churn" if prediction else "Non-Churn"
    color = COLOR["coral"] if prediction else COLOR["teal"]

    a, b, c = st.columns(3, gap="medium")
    with a:
        kpi_card("Churn risk score", percent(score, 2), "◎", COLOR["violet"])
    with b:
        kpi_card("Classification", risk_label, "●", color)
    with c:
        kpi_card("Decision threshold", percent(threshold, 0), "◈", COLOR["blue"])

    section_heading("Individual risk profile")
    pleft, pright = st.columns([1.05, .95], gap="medium")
    with pleft:
        with st.container(border=True):
            panel_header("Churn risk score", "XGBOOST SCORE")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number=dict(suffix="%", font=dict(size=42, color=COLOR["text"])),
                gauge=dict(
                    axis=dict(range=[0, 100], tickwidth=0, tickcolor=COLOR["muted"], tickfont=dict(size=10)),
                    bar=dict(color=color, thickness=.25),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[dict(range=[0, 100], color="#26364D")],
                    threshold=dict(line=dict(color="#F6C96C", width=3),
                                   thickness=.85, value=threshold * 100),
                ),
            ))
            style_plot(gauge, 265)
            gauge.update_layout(margin=dict(l=20, r=20, t=25, b=20))
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    with pright:
        with st.container(border=True):
            panel_header("Risk assessment", "MAR 2017")
            css_class = "status-risk" if prediction else "status-safe"
            st.markdown(
                f'<span class="status {css_class}">● &nbsp; {risk_label}</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:.86rem;color:#B8CAE3;line-height:1.9">'
                f'<b style="color:#F1F4FC">Observation:</b> Feb 01–28, 2017<br>'
                f'<b style="color:#F1F4FC">Prediction:</b> Mar 01–31, 2017<br>'
                f'<b style="color:#F1F4FC">Status:</b> {risk_label}'
                '</div>', unsafe_allow_html=True,
            )
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    if saved and explanation:
        if (abs(float(saved["churn_score"]) - score) > 1e-4
                or int(saved["prediction"]) != prediction):
            st.warning("Saved and live predictions differ. Check the model version and rerun batch prediction if needed.")

    if explanation is None:
        return

    section_heading("Why this prediction?")
    with st.container(border=True):
        panel_header("Feature contributions", f"SHAP · {explanation.get('shap_unit','raw')}")
        top = explanation.get("top_features") or []
        if top:
            sorted_items = sorted(top, key=lambda item: abs(float(item["shap_value"])))
            names = [feature_name_en(str(item["feature"])) for item in sorted_items]
            shap_vals = [float(item["shap_value"]) for item in sorted_items]
            colors = [COLOR["coral"] if val > 0 else COLOR["teal"] for val in shap_vals]
            chart = go.Figure(go.Bar(
                y=names, x=shap_vals, orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                customdata=[str(item["feature"]) for item in sorted_items],
                hovertemplate="%{y}<br>SHAP: %{x:+.4f}<br>Feature: %{customdata}<extra></extra>",
            ))
            style_plot(chart, max(330, 48 * len(top)))
            chart.update_layout(margin=dict(l=10, r=18, t=18, b=35))
            chart.add_vline(x=0, line_width=1, line_color="#60718D")
            chart.update_xaxes(title="SHAP contribution", zeroline=False,
                               gridcolor="#293950", ticks="outside")
            chart.update_yaxes(showgrid=False, tickfont=dict(size=11))
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<span class="microtag" style="color:{COLOR["coral"]}">● Higher risk</span> &nbsp; '
                f'<span class="microtag" style="color:{COLOR["teal"]}">● Lower risk</span>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No SHAP contributions are available.")

    section_heading("Behavioral insights")
    with st.container(border=True):
        insights_vi = explanation.get("explanation_vi") or []
        if insights_vi:
            for index, note in enumerate(insights_vi, 1):
                st.markdown(
                    f'<div style="display:flex;gap:12px;align-items:flex-start;padding:.66rem .3rem;'
                    f'border-bottom:1px solid #27364C">'
                    f'<span style="color:#A5A0FF;font-weight:700;font-size:.76rem;min-width:22px">{index:02d}</span>'
                    f'<span style="color:#CCD9EB;font-size:.88rem;line-height:1.7">{escape(str(note))}</span>'
                    '</div>', unsafe_allow_html=True,
                )
        else:
            st.info("No behavioral insights are available.")

    with st.expander("Technical details · SHAP values"):
        t1, t2, t3 = st.columns(3)
        t1.metric("Baseline probability", percent(explanation.get("baseline_probability"), 2))
        t2.metric("Reconstructed score", percent(explanation.get("reconstructed_probability"), 2))
        t3.metric("SHAP unit", str(explanation.get("shap_unit", "—")))
        raw = explanation.get("contributions") or []
        if raw:
            df = pd.DataFrame(raw).drop(columns=["label_vi"], errors="ignore").rename(columns={
                "feature": "Feature", "scaled_value": "Scaled value",
                "shap_value": "SHAP", "direction": "Direction",
            })
            if "Feature" in df:
                df["Feature"] = df["Feature"].map(feature_name_en)
            if "Direction" in df:
                df["Direction"] = df["Direction"].map({
                    "increase_churn": "Higher risk",
                    "decrease_churn": "Lower risk",
                    "neutral": "Neutral",
                }).fillna("Neutral")
            st.dataframe(df, hide_index=True, use_container_width=True)


if st.session_state["current_page"] == "Overview":
    render_overview()
elif st.session_state["current_page"] == "Users":
    render_users()
else:
    render_user_analysis()

st.markdown(
    '<div class="footnote"><span>KKBOX · Churn Intelligence</span>'
    '<span>Streamlit · FastAPI · XGBoost · SHAP</span></div>',
    unsafe_allow_html=True,
)
