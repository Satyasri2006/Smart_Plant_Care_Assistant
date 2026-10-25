import streamlit as st
import pandas as pd
import joblib
import requests
import ast
import warnings
import base64
from PIL import Image
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation
import db_helper

# Suppress unpickling version warnings for clean terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================
# Load Mascot Image and Inline Icons
# =========================================
try:
    mascot_pil = Image.open("plant_mascot.png")
    with open("plant_mascot.png", "rb") as _f:
        img_b64 = base64.b64encode(_f.read()).decode("utf-8")
    MASCOT_ICON_HTML = f'<img src="data:image/png;base64,{img_b64}" class="mascot-icon" alt="mascot">'
    MASCOT_ICON_MD = f"![mascot](data:image/png;base64,{img_b64})"
except Exception:
    try:
        mascot_pil = Image.open("plant_mascot.jpg")
        with open("plant_mascot.jpg", "rb") as _f:
            img_b64 = base64.b64encode(_f.read()).decode("utf-8")
        MASCOT_ICON_HTML = f'<img src="data:image/jpeg;base64,{img_b64}" class="mascot-icon" alt="mascot">'
        MASCOT_ICON_MD = f"![mascot](data:image/jpeg;base64,{img_b64})"
    except Exception:
        mascot_pil = "🌱"
        img_b64 = ""
        MASCOT_ICON_HTML = "🌱"
        MASCOT_ICON_MD = "🌱"

# =========================================
# Page Configuration
# =========================================
st.set_page_config(
    page_title="Smart Plant Care Assistant",
    page_icon=mascot_pil,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# Load CSS Styles
# =========================================
try:
    with open("style.css", "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass

# =========================================
# Decorative Hanging Vines on Margins
# =========================================
try:
    with open("vine_decoration.png", "rb") as _vf:
        vine_b64 = base64.b64encode(_vf.read()).decode("utf-8")
    st.markdown(
        f'''
        <img src="data:image/png;base64,{vine_b64}" class="vine-decorative vine-left" alt="decorative vine left">
        <img src="data:image/png;base64,{vine_b64}" class="vine-decorative vine-right" alt="decorative vine right">
        ''',
        unsafe_allow_html=True
    )
except Exception:
    pass


# =========================================
# Cached Resource and Data Loading
# =========================================
@st.cache_resource
def load_ml_pipeline():
    """Load the trained Random Forest model and OneHotEncoder."""
    model = joblib.load("plant_watering_model.pkl")
    encoder = joblib.load("plant_encoder.pkl")
    return model, encoder


@st.cache_data
def load_plant_data():
    """Load the ML dataset and botanical care lookup table."""
    df_ml = pd.read_csv("plantfolio_final_ml_dataset.csv", encoding="utf-8", encoding_errors="replace")
    df_care = pd.read_csv("plantfolio_care_lookup.csv", encoding="utf-8", encoding_errors="replace")
    return df_ml, df_care


rf_model, encoder = load_ml_pipeline()
df, care_df = load_plant_data()

categorical_cols = [
    "Plant_Name",
    "Season",
    "Humidity_Preference",
    "Light_Preference",
    "Drainage_Preference",
    "Watering_Method"
]

numerical_cols = [
    "Temperature_Preference_Min_C",
    "Temperature_Preference_Max_C"
]


# =========================================
# Season Determination Utilities
# =========================================
def get_current_indian_season():
    """Determine the current Indian seasonal cycle and its dataset equivalent."""
    month = datetime.now().month
    if month in [1, 2]:
        return "Winter", "Winter"
    elif month in [3, 4, 5]:
        return "Summer", "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon", "Summer"
    else:
        return "Post-Monsoon", "Fall"


def parse_temperature_range(temp_pref_str):
    """Safely parse temperature preference string [min, max]."""
    try:
        val = ast.literal_eval(str(temp_pref_str))
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return float(val[0]), float(val[1])
    except Exception:
        pass
    return 18.0, 26.0


# =========================================
# Prediction & Inference Engine
# =========================================
def predict_watering_frequency(plant_name, season):
    """
    Predict watering interval using the trained Random Forest model.
    If the plant or season is missing from the ML dataset, fall back to care lookup intervals.
    """
    plant_data = df[
        (df["Plant_Name"].str.lower() == plant_name.lower()) &
        (df["Season"].str.lower() == season.lower())
    ]

    if not plant_data.empty:
        row = plant_data.iloc[0]
        input_data = pd.DataFrame([row.drop("Watering_Frequency_days")])

        input_encoded = encoder.transform(input_data[categorical_cols])
        input_encoded = pd.DataFrame(
            input_encoded,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        input_final = pd.concat(
            [
                input_data[numerical_cols].reset_index(drop=True),
                input_encoded.reset_index(drop=True)
            ],
            axis=1
        )

        prediction = rf_model.predict(input_final)[0]
        return round(float(prediction), 1), "ml_model"

    # Fallback to botanical care lookup if row is absent from ML dataset
    care_match = care_df[care_df["typeName"].str.lower() == plant_name.lower()]
    if not care_match.empty:
        care_row = care_match.iloc[0]
        interval_col = f"{season.lower()}Interval"
        if interval_col in care_row and pd.notna(care_row[interval_col]):
            return float(care_row[interval_col]), "care_lookup"

        # If specific season is NaN (e.g. winterInterval), estimate from available seasons
        if pd.notna(care_row.get("fallInterval")):
            return round(float(care_row["fallInterval"]) * 1.3, 1), "estimated_care"
        elif pd.notna(care_row.get("springInterval")):
            return round(float(care_row["springInterval"]) * 1.4, 1), "estimated_care"

    return 7.0, "default"


def compute_environmental_adjustment(
    base_interval,
    current_temp,
    current_humidity,
    sunlight_hours,
    soil_condition,
    temp_min,
    temp_max,
    humidity_pref
):
    """
    Adjust base watering frequency dynamically according to real-time environmental stress.
    Considers ambient temperature difference, ambient humidity, daily sun, and soil moisture.
    """
    factor = 1.0
    adjustment_reasons = []

    # Temperature Effect
    if current_temp is not None:
        if current_temp > temp_max:
            excess = current_temp - temp_max
            temp_penalty = min(0.30, excess * 0.03)
            factor -= temp_penalty
            adjustment_reasons.append(
                f"High temperature (+{excess:.1f}°C above optimum) accelerates evapotranspiration."
            )
        elif current_temp < temp_min:
            deficit = temp_min - current_temp
            temp_relief = min(0.35, deficit * 0.035)
            factor += temp_relief
            adjustment_reasons.append(
                f"Cooler temperature (-{deficit:.1f}°C below optimum) slows water uptake."
            )

    # Humidity Effect
    if current_humidity is not None:
        if current_humidity < 35:
            factor -= 0.12
            adjustment_reasons.append("Very dry ambient air increases water evaporation from soil.")
        elif current_humidity > 75:
            factor += 0.15
            adjustment_reasons.append("High ambient humidity keeps soil damp for longer.")

    # Sunlight Hours Effect
    if sunlight_hours >= 7:
        factor -= 0.10
        adjustment_reasons.append(f"Extended daily sunlight ({sunlight_hours:.1f} hrs) promotes high transpiration.")
    elif sunlight_hours <= 2:
        factor += 0.10
        adjustment_reasons.append(f"Low daily sunlight ({sunlight_hours:.1f} hrs) reduces plant moisture demand.")

    # Clamp factor to realistic bounds (0.45x to 1.7x baseline)
    factor = max(0.45, min(1.70, factor))
    adjusted_interval = max(1.0, round(base_interval * factor, 1))

    # Immediate Action Directive based on current soil condition
    if soil_condition == "Extremely Dry":
        action_title = "🚨 Immediate Action: Water Thoroughly Today"
        action_desc = "The soil is completely parched. Deeply water your plant until moisture drains from the bottom holes."
        action_badge = "badge-alert"
    elif soil_condition == "Dry":
        action_title = "💧 Time to Water Today"
        action_desc = "The top 1–2 inches have dried out completely. Optimal time to water your plant today."
        action_badge = "badge-good"
    elif soil_condition == "Moist":
        action_title = "🌿 Soil Well-Moistened: Hold Off"
        action_desc = "Soil has healthy moisture reserves. No watering required today; check again in 2–3 days."
        action_badge = "badge-info"
    elif soil_condition == "Very Moist":
        action_title = "⏳ Damp Soil: Do Not Water Yet"
        action_desc = "Soil is still quite moist. Allow it to aerate and dry out before applying more water."
        action_badge = "badge-warn"
    elif soil_condition == "Wet":
        action_title = "⚠️ Waterlogged Caution: Hold Water!"
        action_desc = "Soil is saturated with water. Ensure drainage holes are free to avoid root rot and fungal issues."
        action_badge = "badge-alert"
    else:
        action_title = f"{MASCOT_ICON_HTML} Check Soil Moisture"
        action_desc = f"Follow recommended cycle of watering roughly every {adjusted_interval} days."
        action_badge = "badge-info"

    return adjusted_interval, action_title, action_desc, action_badge, adjustment_reasons


# =========================================
# Environmental Suitability Checks
# =========================================
def evaluate_temperature(current_temp, temp_min, temp_max):
    if current_temp is None:
        return "Unknown", "badge-info", "Temperature reading not available."
    if current_temp < temp_min:
        return "Cooler than preferred", "badge-warn", f"{current_temp}°C is below preferred range ({temp_min:.0f}–{temp_max:.0f}°C). Growth and water uptake may slow down."
    elif current_temp > temp_max:
        return "Warmer than preferred", "badge-alert", f"{current_temp}°C exceeds preferred range ({temp_min:.0f}–{temp_max:.0f}°C). Watch out for wilting and heat stress."
    else:
        return "Ideal Temperature", "badge-good", f"{current_temp}°C is comfortably within the plant's ideal range ({temp_min:.0f}–{temp_max:.0f}°C)."


def evaluate_humidity(current_humidity, recommended_humidity):
    if current_humidity is None or not recommended_humidity:
        return "Unknown", "badge-info", "Humidity preference not available."

    rec = str(recommended_humidity).lower()
    if rec == "low":
        if current_humidity > 60:
            return "Higher than ideal", "badge-warn", f"{current_humidity}% is higher than preferred for this drought-tolerant species."
        return "Suitable", "badge-good", f"{current_humidity}% suits this plant's low-humidity preference."
    elif rec == "medium":
        if current_humidity < 40:
            return "Dry Air", "badge-warn", f"{current_humidity}% is slightly dry. Consider light misting or grouping plants."
        elif current_humidity > 75:
            return "Humid", "badge-info", f"{current_humidity}% is humid; keep adequate airflow."
        return "Suitable", "badge-good", f"{current_humidity}% aligns with medium humidity requirements (40–70%)."
    elif rec in ["high", "veryhigh"]:
        if current_humidity < 55:
            return "Insufficient Humidity", "badge-alert", f"{current_humidity}% is low for this tropical plant. Use a pebble tray or humidifier."
        return "Optimal Humidity", "badge-good", f"{current_humidity}% is ideal for high humidity lovers."
    return "Acceptable", "badge-info", f"Current humidity: {current_humidity}%."


# =========================================
# Session State Initialization
# =========================================
if "user" not in st.session_state:
    st.session_state["user"] = None
if "started" not in st.session_state:
    st.session_state["started"] = False
if "auth_tab" not in st.session_state:
    st.session_state["auth_tab"] = "login"
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "main"
if "show_suggestions" not in st.session_state:
    st.session_state["show_suggestions"] = False
if "current_temp" not in st.session_state:
    st.session_state["current_temp"] = None
if "current_humidity" not in st.session_state:
    st.session_state["current_humidity"] = None
if "weather_source" not in st.session_state:
    st.session_state["weather_source"] = None
if "active_consultation" not in st.session_state:
    st.session_state["active_consultation"] = None
if "last_saved_key" not in st.session_state:
    st.session_state["last_saved_key"] = None


# =========================================
# Helper: Save Active Consultation to History
# =========================================
def save_current_consultation_if_needed():
    """Ensure the current active consultation is saved to MongoDB history before leaving."""
    if st.session_state.get("active_consultation") and st.session_state.get("user"):
        c_data = st.session_state["active_consultation"]
        c_key = f"{c_data.get('plant_name')}_{c_data.get('season')}_{c_data.get('temperature')}_{c_data.get('humidity')}_{c_data.get('soil_condition')}"
        if st.session_state.get("last_saved_key") != c_key:
            db_helper.save_consultation_history(
                user_email=st.session_state["user"]["email"],
                user_name=st.session_state["user"]["name"],
                consultation_data=c_data
            )
            st.session_state["last_saved_key"] = c_key


# =========================================
# 1. Landing / Welcome & Authentication Screen
# =========================================
if not st.session_state.get("started", False):
    img_tag = f'<img src="data:image/png;base64,{img_b64}" class="landing-img" alt="Plant Mascot">' if img_b64 else '<div style="font-size: 80px; margin-bottom: 15px;">🪴</div>'

    landing_col_left, landing_col_right = st.columns([1, 1.15], gap="large")

    with landing_col_left:
        st.markdown(
            f'''
            <div class="landing-wrapper" style="box-shadow: none; border: none; background: transparent; padding-top: 15px;">
                {img_tag}
                <div class="landing-title">Smart Plant Care Assistant</div>
                <div class="landing-sub">Your AI companion for healthy, thriving plants with customized watering predictions and live weather intelligence.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

    with landing_col_right:
        # Display MongoDB status banner
        conn_res = db_helper.get_connection_status()
        conn_name = conn_res[0]
        conn_ok = conn_res[1]
        conn_err = conn_res[2] if len(conn_res) > 2 else (conn_name if not conn_ok else None)

        if not conn_ok:
            st.error(f"⚠️ **MongoDB Connection Notice**: {conn_err}")
        else:
            if "Atlas" in str(conn_name):
                st.caption(f"☁️ Connected to **{conn_name}**")

        # If user is not logged in: display Login / Register interface matching reference
        if st.session_state.get("user") is None:
            # Segmented Tab Selector (Green highlight instead of blue)
            tab_c1, tab_c2 = st.columns(2)
            with tab_c1:
                is_login = (st.session_state["auth_tab"] == "login")
                if st.button("Log In", use_container_width=True, key="tab_login_btn", type="primary" if is_login else "secondary"):
                    st.session_state["auth_tab"] = "login"
                    st.rerun()
            with tab_c2:
                is_reg = (st.session_state["auth_tab"] == "register")
                if st.button("Register", use_container_width=True, key="tab_reg_btn", type="primary" if is_reg else "secondary"):
                    st.session_state["auth_tab"] = "register"
                    st.rerun()

            # Render Active Form
            if st.session_state["auth_tab"] == "login":
                st.markdown(
                    '''
                    <div class="auth-header-title">Log In</div>
                    <div class="auth-header-sub">Welcome back! Please enter your details to continue.</div>
                    ''',
                    unsafe_allow_html=True
                )
                login_email = st.text_input("Email Address", placeholder="e.g. alex@example.com", key="login_email")
                login_password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

                if st.button("Log In", type="primary", use_container_width=True, key="login_submit"):
                    success, res = db_helper.authenticate_user(login_email, login_password)
                    if success:
                        st.session_state["user"] = res
                        st.session_state["started"] = True
                        st.session_state["view_mode"] = "main"
                        st.rerun()
                    else:
                        st.error(res)

            else:
                st.markdown(
                    '''
                    <div class="auth-header-title">Register</div>
                    <div class="auth-header-sub">Create an account to start chatting with your Smart Plant Care Assistant</div>
                    ''',
                    unsafe_allow_html=True
                )
                reg_name = st.text_input("Full Name (how we should call you)", placeholder="e.g. Alex", key="reg_name")
                reg_email = st.text_input("Email Address", placeholder="e.g. alex@example.com", key="reg_email")
                reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="reg_confirm")

                if st.button("Register", type="primary", use_container_width=True, key="reg_submit"):
                    if reg_password != reg_confirm:
                        st.error("Passwords do not match. Please verify and try again.")
                    else:
                        success, res = db_helper.register_user(reg_name, reg_email, reg_password)
                        if success:
                            st.session_state["user"] = res
                            st.session_state["started"] = True
                            st.session_state["view_mode"] = "main"
                            st.rerun()
                        else:
                            st.error(res)

        else:
            # User is already logged in
            current_user = st.session_state["user"]
            st.markdown(
                f'''
                <div class="auth-card" style="text-align: center; margin-top: 25px; padding: 40px 24px;">
                    <div style="font-size: 42px; margin-bottom: 10px;">👋</div>
                    <div class="auth-header-title">Hi {current_user['name']}!</div>
                    <div class="auth-header-sub">Logged in as <b>{current_user['email']}</b></div>
                    <div style="color: #66756A; font-size: 14px; margin-bottom: 25px;">
                        Your plant profiles, environmental history, and personalized care data are ready.
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.write("")
            if st.button("Ask away...", type="primary", use_container_width=True, key="start_app_btn"):
                st.session_state["started"] = True
                st.session_state["view_mode"] = "main"
                st.rerun()

            st.write("")
            if st.button("🚪 Log Out", use_container_width=True, key="landing_logout_btn"):
                st.session_state["user"] = None
                st.session_state["started"] = False
                st.rerun()

    st.stop()


# =========================================
# 2. Main Interface Header & Navigation
# =========================================
current_user = st.session_state.get("user") or {"name": "Friend", "email": ""}
user_name = current_user.get("name", "Friend")
user_email = current_user.get("email", "")

header_left, header_right = st.columns([3.8, 1.4], gap="medium")

with header_left:
    st.markdown(f'<div class="user-greeting">Hi, {user_name}!</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">{MASCOT_ICON_HTML} Your Smart Plant Care Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Personalized plant watering and care recommendations based on your plant and real-time environmental conditions.'
        '</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.write("")
    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 0.9])
    with nav_c1:
        if st.button("📜 See History", use_container_width=True, key="see_hist_nav_btn"):
            save_current_consultation_if_needed()
            st.session_state["view_mode"] = "history"
            st.rerun()
    with nav_c2:
        if st.button("🏠 Home", use_container_width=True, key="home_nav_btn"):
            save_current_consultation_if_needed()
            st.session_state["active_consultation"] = None
            st.session_state["show_suggestions"] = False
            st.session_state["started"] = False
            st.session_state["view_mode"] = "main"
            st.rerun()
    with nav_c3:
        if st.button("🚪 Logout", use_container_width=True, key="logout_nav_btn"):
            save_current_consultation_if_needed()
            st.session_state["user"] = None
            st.session_state["started"] = False
            st.session_state["active_consultation"] = None
            st.session_state["show_suggestions"] = False
            st.session_state["view_mode"] = "main"
            st.rerun()


# =========================================
# 3. Consultation History View
# =========================================
if st.session_state.get("view_mode") == "history":
    st.markdown(f'<div class="section-title">📜 Consultation History for {user_name}</div>', unsafe_allow_html=True)
    conn_type, _ = db_helper.get_connection_status()
    st.caption(f"Account: {user_email} | Stored securely in {conn_type}")

    hist_act1, hist_act2, _ = st.columns([1.2, 1.2, 2.6])
    with hist_act1:
        if st.button("← Back to Assistant", use_container_width=True, key="back_from_hist_btn"):
            st.session_state["view_mode"] = "main"
            st.rerun()
    with hist_act2:
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_hist_btn"):
            db_helper.clear_user_history(user_email)
            st.success("Your consultation history has been cleared.")
            st.rerun()

    history_records = db_helper.get_user_history(user_email)

    if not history_records:
        st.markdown(
            f"""
            <div class="plant-card" style="text-align: center; padding: 60px 20px; margin-top: 20px;">
                <div style="font-size: 48px; margin-bottom: 12px;">🌱</div>
                <div style="font-size: 20px; font-weight: 700; color: #234D35;">No History Yet</div>
                <div style="color: #66756A; margin-top: 8px;">
                    Ask for watering recommendations for any plant, and your consultation history will be saved here automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("")
        for rec in history_records:
            timestamp_str = rec.get("timestamp").strftime("%b %d, %Y - %I:%M %p") if isinstance(rec.get("timestamp"), datetime) else "Recent"
            plant_title = rec.get("plant_name", "Unknown Plant")
            season_info = rec.get("season", "Season")
            rec_interval = rec.get("recommended_interval", "N/A")
            act_title = rec.get("action_title", "Consultation Result")
            act_desc = rec.get("action_desc", "")
            soil_cond = rec.get("soil_condition", "Not specified")
            temp_val = rec.get("temperature")
            hum_val = rec.get("humidity")
            sun_val = rec.get("sunlight_hours")
            badge_class = rec.get("action_badge", "badge-info")

            st.markdown(
                f"""
                <div class="history-entry-card">
                    <div class="history-time">🗓️ {timestamp_str}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div class="history-plant-title">
                            {MASCOT_ICON_HTML} {plant_title} <span style="font-size: 14px; font-weight: 500; color: #66756A;">({season_info})</span>
                        </div>
                        <div>
                            <span class="badge {badge_class}">{soil_cond} Soil</span>
                            <span class="badge badge-good">Cycle: Every {rec_interval} Days</span>
                        </div>
                    </div>
                    <div style="font-size: 14px; font-weight: 700; color: #234D35; margin-top: 10px;">{act_title}</div>
                    <div style="font-size: 13.5px; color: #435A4B; margin-top: 4px;">{act_desc}</div>
                    <div style="font-size: 12px; color: #718078; margin-top: 8px;">
                        🌡️ Temp: {temp_val}°C &nbsp;|&nbsp; 💧 Humidity: {hum_val}% &nbsp;|&nbsp; ☀️ Sunlight: {sun_val} hrs
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# =========================================
# 4. Main Plant Care Assistant View
# =========================================
col_inputs, col_dashboard = st.columns([1, 1.25], gap="large")

# -----------------------------------------
# Left Column: Inputs & Environment
# -----------------------------------------
with col_inputs:
    st.markdown('<div class="section-title">🌿 1. Select Your Plant</div>', unsafe_allow_html=True)

    plant_names = sorted(care_df["typeName"].dropna().unique())
    selected_plant = st.selectbox(
        "Choose your plant",
        plant_names,
        index=plant_names.index("Aglaonema") if "Aglaonema" in plant_names else None,
        placeholder="Select a plant..."
    )

    st.markdown('<div class="section-title">📍 2. Live Environmental Conditions</div>', unsafe_allow_html=True)

    st.caption("Click the button below to allow browser location access and fetch live local weather:")
    location = streamlit_geolocation()

    # Process live browser GPS location when permission is granted
    if (
        location
        and location.get("latitude") is not None
        and location.get("longitude") is not None
    ):
        lat = location["latitude"]
        lon = location["longitude"]

        try:
            w_res = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m"
                },
                timeout=10
            )
            if w_res.ok:
                w_json = w_res.json()
                st.session_state["current_temp"] = float(w_json["current"]["temperature_2m"])
                st.session_state["current_humidity"] = float(w_json["current"]["relative_humidity_2m"])
                st.session_state["weather_source"] = f"Live GPS ({lat:.2f}°, {lon:.2f}°)"
            else:
                st.warning("Unable to fetch current weather data from meteorological server.")
        except Exception as e:
            st.warning(f"Weather fetch error: {e}")

    # Display live weather status badge if GPS data is present
    if st.session_state["current_temp"] is not None and st.session_state["current_humidity"] is not None:
        st.success(
            f"✅ **Live Weather Synchronized ({st.session_state['weather_source']}):**  \n"
            f"**{st.session_state['current_temp']}°C** temperature &nbsp;|&nbsp; **{st.session_state['current_humidity']}%** humidity"
        )
    else:
        st.info("👆 Please click the location pin button above and allow permission when prompted by your browser.")

    # Manual Fallback Expander
    with st.expander("🛠️ Manual Weather Override (if location permission is unavailable)", expanded=False):
        st.caption("You can manually specify ambient conditions if your browser does not have GPS enabled:")
        man_temp = st.slider("Manual Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
        man_hum = st.slider("Manual Relative Humidity (%)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
        if st.button("Apply Manual Weather"):
            st.session_state["current_temp"] = man_temp
            st.session_state["current_humidity"] = man_hum
            st.session_state["weather_source"] = "Manual Override"
            st.rerun()

    st.markdown(f'<div class="section-title">{MASCOT_ICON_HTML} 3. Plant Environment</div>', unsafe_allow_html=True)

    env_col1, env_col2 = st.columns(2)
    with env_col1:
        sunlight_hours = st.number_input(
            "☀️ Daily Sunlight (hours)",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

    with env_col2:
        soil_condition = st.selectbox(
            f"{MASCOT_ICON_MD} Soil Condition",
            [
                "Extremely Dry",
                "Dry",
                "Moist",
                "Very Moist",
                "Wet"
            ],
            index=1,
            placeholder="Select soil condition..."
        )

    detected_indian_season, dataset_season = get_current_indian_season()
    with st.expander("🗓️ Seasonal Cycle", expanded=False):
        st.write(f"Current Season: **{detected_indian_season}** (mapped to `{dataset_season}` model season)")
        season_override = st.selectbox(
            "Override Season (optional)",
            ["Auto Detect", "Spring", "Summer", "Fall", "Winter"],
            index=0
        )
        if season_override != "Auto Detect":
            dataset_season = season_override

    st.write("")
    # Primary CTA Button: Get Suggestions
    get_suggestions = st.button(f"{MASCOT_ICON_MD} Get Suggestions", use_container_width=True)
    if get_suggestions:
        if not selected_plant:
            st.error("Please choose a plant first before getting suggestions.")
            st.session_state["show_suggestions"] = False
        else:
            st.session_state["show_suggestions"] = True


# -----------------------------------------
# Right Column: AI Suggestions Dashboard
# -----------------------------------------
with col_dashboard:
    if not st.session_state.get("show_suggestions", False):
        st.markdown(
            f"""
            <div class="plant-card" style="text-align: center; padding: 70px 24px;">
                <div style="margin-bottom: 16px;"><img src="data:image/png;base64,{img_b64}" class="mascot-icon" style="height: 52px; width: auto; border-radius: 8px;" alt="mascot"></div>
                <div style="font-size: 22px; font-weight: 700; color: #234D35;">Awaiting Your Input</div>
                <div style="color: #66756A; margin-top: 10px; font-size: 15px; line-height: 1.6; max-width: 440px; margin-left: auto; margin-right: auto;">
                    Select your plant, allow browser location permission to sync live temperature & humidity, set your environment, and click <b>"Get suggestions"</b> below.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        active_temp = st.session_state.get("current_temp")
        active_hum = st.session_state.get("current_humidity")

        if active_temp is None or active_hum is None:
            active_temp = 25.0
            active_hum = 50.0
            st.warning("⚠️ Live location was not detected. Using standard ambient baseline (25°C / 50% humidity). Allow GPS access on the left for live accuracy.")

        care_matches = care_df[care_df["typeName"].str.lower() == selected_plant.lower()]
        care_info = care_matches.iloc[0] if not care_matches.empty else None

        temp_min, temp_max = (18.0, 27.0)
        humidity_pref = "medium"
        light_pref = "brightIndirect"
        watering_method = "topWatering"
        drainage_pref = "wellDraining"
        soil_ph = "neutral"
        care_tips = "Keep soil slightly moist and avoid waterlogging."

        if care_info is not None:
            temp_min, temp_max = parse_temperature_range(care_info.get("temperaturePreference", "[18, 27]"))
            humidity_pref = care_info.get("humidityPreference", "medium")
            light_pref = care_info.get("lightPreference", "brightIndirect")
            watering_method = care_info.get("wateringMethod", "topWatering")
            drainage_pref = care_info.get("drainagePreference", "wellDraining")
            soil_ph = care_info.get("soilPhPreference", "neutral")
            care_tips = str(care_info.get("careTips", care_tips))

        is_aquatic = (str(watering_method).lower() == "immersion" and str(drainage_pref).lower() == "waterloggingtolerant")

        if is_aquatic:
            st.markdown(
                f"""
                <div class="hero-card">
                    <div class="card-label">Aquatic Species Care</div>
                    <div class="hero-value">🌊 Continuous Immersion</div>
                    <div class="hero-sub">
                        <b>{selected_plant}</b> is an aquatic plant requiring continuous water immersion. Standard soil drying schedules do not apply.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.info("💧 **Care Directive:** Ensure clean water immersion at all times. Replace 20–30% of water periodically.")

        else:
            base_freq, freq_source = predict_watering_frequency(selected_plant, dataset_season)

            adj_interval, action_title, action_desc, action_badge, reasons = compute_environmental_adjustment(
                base_interval=base_freq,
                current_temp=active_temp,
                current_humidity=active_hum,
                sunlight_hours=sunlight_hours,
                soil_condition=soil_condition,
                temp_min=temp_min,
                temp_max=temp_max,
                humidity_pref=humidity_pref
            )

            # Store current consultation in session state and persist to MongoDB
            active_consultation = {
                "plant_name": selected_plant,
                "season": dataset_season,
                "temperature": active_temp,
                "humidity": active_hum,
                "weather_source": st.session_state.get("weather_source", "Ambient"),
                "sunlight_hours": sunlight_hours,
                "soil_condition": soil_condition,
                "recommended_interval": adj_interval,
                "base_interval": base_freq,
                "action_title": action_title,
                "action_desc": action_desc,
                "action_badge": action_badge,
                "reasons": reasons,
                "care_profile": {
                    "light": light_pref,
                    "watering_method": watering_method,
                    "drainage": drainage_pref,
                    "soil_ph": soil_ph,
                    "care_tips": care_tips
                }
            }
            st.session_state["active_consultation"] = active_consultation

            # Auto-save to MongoDB
            c_key = f"{selected_plant}_{dataset_season}_{active_temp}_{active_hum}_{soil_condition}"
            if st.session_state.get("last_saved_key") != c_key and user_email:
                db_helper.save_consultation_history(
                    user_email=user_email,
                    user_name=user_name,
                    consultation_data=active_consultation
                )
                st.session_state["last_saved_key"] = c_key

            # --- Hero Recommendation Card ---
            st.markdown(
                f"""
                <div class="hero-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="card-label">Recommended Watering Cycle</span>
                        <span class="badge {action_badge}">{soil_condition} Soil</span>
                    </div>
                    <div class="hero-value">Every {adj_interval:.1f} Days</div>
                    <div class="hero-sub">
                        Baseline model: <b>{base_freq:.1f} days</b> ({dataset_season} schedule)
                        {f' • Adjusted for current climate' if adj_interval != base_freq else ' • Matches ideal baseline'}
                    </div>
                    <hr style="margin: 14px 0 10px 0; border-top: 1px dashed #C5D6C8;">
                    <div style="font-size: 15.5px; font-weight: 700; color: #234D35; margin-bottom: 4px;">{action_title}</div>
                    <div style="font-size: 13.5px; color: #435A4B;">{action_desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- Environmental Diagnostics ---
            st.markdown('<div class="section-title">🌡️ Environmental Diagnostics</div>', unsafe_allow_html=True)

            temp_eval, temp_badge, temp_msg = evaluate_temperature(active_temp, temp_min, temp_max)
            hum_eval, hum_badge, hum_msg = evaluate_humidity(active_hum, humidity_pref)

            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <div class="card-label">Ambient Temperature</div>
                        <div class="card-value">{active_temp:.1f}°C</div>
                        <span class="badge {temp_badge}">{temp_eval}</span>
                        <div style="font-size: 12.5px; color: #556B5D; margin-top: 6px;">{temp_msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with diag_col2:
                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <div class="card-label">Ambient Humidity</div>
                        <div class="card-value">{active_hum:.0f}%</div>
                        <span class="badge {hum_badge}">{hum_eval}</span>
                        <div style="font-size: 12.5px; color: #556B5D; margin-top: 6px;">{hum_msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if reasons:
                with st.expander("🔍 Why was the watering frequency adjusted?", expanded=False):
                    for r in reasons:
                        st.markdown(f"- {r}")

        # --- Botanical Care Guide Card ---
        st.markdown('<div class="section-title">📖 Botanical Care Guide</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="plant-card">
                <div style="margin-bottom: 14px;">
                    <span class="badge badge-good">☀️ Light: {light_pref}</span>
                    <span class="badge badge-info">💧 Method: {watering_method}</span>
                    <span class="badge badge-good">{MASCOT_ICON_HTML} Drainage: {drainage_pref}</span>
                    <span class="badge badge-info">🧪 Soil pH: {soil_ph}</span>
                    <span class="badge badge-warn">🌡️ Preferred: {temp_min:.0f}–{temp_max:.0f}°C</span>
                </div>
                <div class="card-label" style="margin-bottom: 6px;">Care Instructions</div>
                <div class="care-tips">{care_tips}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # --- Seasonal Schedule Chips ---
        if care_info is not None and not is_aquatic:
            st.markdown('<div class="section-title">📅 Annual Seasonal Watering Schedule</div>', unsafe_allow_html=True)

            sp_int = care_info.get("springInterval")
            su_int = care_info.get("summerInterval")
            fa_int = care_info.get("fallInterval")
            wi_int = care_info.get("winterInterval")

            def format_int(val):
                return f"{float(val):.0f} days" if pd.notna(val) else "N/A"

            st.markdown(
                f"""
                <div class="season-grid">
                    <div class="season-chip {'active' if dataset_season == 'Spring' else ''}">
                        <div class="season-name">🌸 Spring</div>
                        <div class="season-val">{format_int(sp_int)}</div>
                    </div>
                    <div class="season-chip {'active' if dataset_season == 'Summer' else ''}">
                        <div class="season-name">☀️ Summer</div>
                        <div class="season-val">{format_int(su_int)}</div>
                    </div>
                    <div class="season-chip {'active' if dataset_season == 'Fall' else ''}">
                        <div class="season-name">🍂 Fall</div>
                        <div class="season-val">{format_int(fa_int)}</div>
                    </div>
                    <div class="season-chip {'active' if dataset_season == 'Winter' else ''}">
                        <div class="season-name">❄️ Winter</div>
                        <div class="season-val">{format_int(wi_int)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )