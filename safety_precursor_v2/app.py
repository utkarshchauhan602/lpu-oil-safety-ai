import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import re
from datetime import datetime, timedelta, date
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import plotly.express as px

# --------------------------- CONFIG ---------------------------
st.set_page_config(page_title="Safety Precursor Intelligence", page_icon="🛡️", layout="wide")
DB_PATH = Path("safety_precursor.db")
MODEL_CATEGORIES = [
    "Mechanical / Procedure Violation",
    "Fall & Working at Heights Hazard",
    "Chemical / Gas Leak Hazard",
    "Electrical Safety Hazard",
    "Routine / Non-Hazardous Observation",
]
SITES = ["Refinery Unit A", "Refinery Unit B", "Storage Yard Alpha", "Assembly Plant 2"]
DEPARTMENTS = ["Operations", "Maintenance", "Warehouse", "Utilities", "Production"]

# --------------------------- STYLING ---------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stSidebar"] {background: #101827;}
[data-testid="stSidebar"] * {color: #edf2f7;}
div[data-testid="stMetric"] {background: #f4f7fb; border: 1px solid #e2e8f0; padding: 14px; border-radius: 12px;}
div.stButton > button {border-radius: 9px; font-weight: 600;}
.small-muted {color:#64748b;font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)

# --------------------------- DATABASE ---------------------------
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_code TEXT UNIQUE,
            report_date TEXT,
            site TEXT,
            department TEXT,
            shift TEXT,
            equipment TEXT,
            report_text TEXT,
            category TEXT,
            risk_level TEXT,
            risk_score INTEGER,
            model_confidence REAL,
            status TEXT DEFAULT 'Open',
            corrective_action TEXT,
            source TEXT DEFAULT 'User submitted',
            is_incident INTEGER DEFAULT 0
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            action_text TEXT,
            owner TEXT,
            due_date TEXT,
            status TEXT DEFAULT 'Open',
            FOREIGN KEY(report_id) REFERENCES reports(id)
        )""")

def read_reports():
    with db() as con:
        return pd.read_sql_query("SELECT * FROM reports ORDER BY report_date DESC, id DESC", con)

def save_report(row):
    with db() as con:
        con.execute("""
        INSERT INTO reports
        (report_code, report_date, site, department, shift, equipment, report_text,
         category, risk_level, risk_score, model_confidence, status, corrective_action, source, is_incident)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["report_code"], row["report_date"], row["site"], row["department"],
            row["shift"], row["equipment"], row["report_text"], row["category"],
            row["risk_level"], int(row["risk_score"]), float(row["model_confidence"]),
            row["status"], row["corrective_action"], row["source"], int(row["is_incident"])
        ))

def seed_demo_data():
    with db() as con:
        count = con.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    if count:
        return
    rng = np.random.default_rng(27)
    samples = {
        MODEL_CATEGORIES[0]: [
            "Operator removed the press machine guard while the machine was running.",
            "Maintenance team bypassed the conveyor interlock during troubleshooting.",
            "Technician started pump maintenance without applying lockout tagout.",
            "Forklift was operated above the marked speed limit in the loading bay.",
            "Machine emergency stop was obstructed by stored material.",
            "Worker reached into moving equipment to clear a jam.",
            "Safety procedure was skipped before restarting the compressor.",
            "A contractor used a machine without completing the required permit."
        ],
        MODEL_CATEGORIES[1]: [
            "Scaffold platform has a loose plank and missing toe board at height.",
            "Worker used a ladder with a damaged side rail near the tank.",
            "Open floor edge on the mezzanine has no guardrail or warning barrier.",
            "Harness lanyard was not connected while working on the elevated platform.",
            "Access ladder to the roof is unsecured and shifts under load.",
            "Materials are stored close to an unprotected floor opening.",
            "Scaffolding inspection tag is missing from the work area.",
            "Worker was standing on an unstable surface to reach overhead piping."
        ],
        MODEL_CATEGORIES[2]: [
            "Chemical smell was reported near the chlorine cylinder valve.",
            "Small solvent leak was observed below the transfer pump flange.",
            "Gas detector alarm activated near the storage manifold.",
            "Corrosion is visible on the chemical dosing line connection.",
            "Unlabelled chemical container was found leaking in the storage area.",
            "Operator noticed vapour escaping from a damaged process hose.",
            "Acid transfer hose has a wet patch around the coupling.",
            "Ventilation was not operating in the chemical handling room."
        ],
        MODEL_CATEGORIES[3]: [
            "Damaged insulation was found on a temporary welding power cable.",
            "Electrical panel door was open with exposed terminals accessible.",
            "Extension lead is routed through standing water near the workshop.",
            "Grounding cable is disconnected from the portable welding machine.",
            "Burn marks were observed around the distribution board socket.",
            "A temporary cable has exposed copper strands near the walkway.",
            "Circuit breaker cover is missing from the utility panel.",
            "Worker reported a tingling sensation after touching metal equipment."
        ],
        MODEL_CATEGORIES[4]: [
            "Shift handover completed and all routine checks were recorded.",
            "Work area was clean and access routes remained unobstructed.",
            "Scheduled inspection completed with no abnormal findings.",
            "PPE stock was replenished and stored in the designated cabinet.",
            "Routine toolbox talk was completed before the morning shift.",
            "Fire extinguishers were inspected and tags were up to date.",
            "Production readings remained within normal operating limits.",
            "Housekeeping inspection found no safety observations."
        ]
    }
    modifiers = [
        " during the morning shift", " during night shift", " near the north access gate",
        " after routine maintenance", " during a contractor activity", " beside line 3",
        " during equipment cleaning", " before the scheduled shutdown"
    ]
    rows = []
    today = date.today()
    for i in range(240):
        cat = rng.choice(MODEL_CATEGORIES, p=[.23, .22, .19, .19, .17])
        txt = str(rng.choice(samples[cat])) + str(rng.choice(modifiers))
        days_ago = int(rng.integers(0, 90))
        risk = 1 if cat == MODEL_CATEGORIES[4] else int(rng.choice([2, 3, 4, 5], p=[.12, .28, .40, .20]))
        if risk <= 2: level = "Low"
        elif risk <= 3: level = "Medium"
        else: level = "High"
        rows.append({
            "report_code": f"DEMO-{1000+i}", "report_date": (today-timedelta(days=days_ago)).isoformat(),
            "site": str(rng.choice(SITES)), "department": str(rng.choice(DEPARTMENTS)),
            "shift": str(rng.choice(["Day", "Evening", "Night"])),
            "equipment": str(rng.choice(["Conveyor", "Pump", "Compressor", "Scaffold", "Panel", "General"])),
            "report_text": txt, "category": cat, "risk_level": level,
            "risk_score": risk*20, "model_confidence": 0.0, "status": str(rng.choice(["Open","In Progress","Closed"], p=[.35,.25,.40])),
            "corrective_action": "Review observation and apply site safety procedure.",
            "source": "Synthetic demonstration data", "is_incident": 0
        })
    with db() as con:
        con.executemany("""
        INSERT OR IGNORE INTO reports
        (report_code, report_date, site, department, shift, equipment, report_text, category,
         risk_level, risk_score, model_confidence, status, corrective_action, source, is_incident)
        VALUES (:report_code,:report_date,:site,:department,:shift,:equipment,:report_text,:category,
         :risk_level,:risk_score,:model_confidence,:status,:corrective_action,:source,:is_incident)
        """, rows)

init_db()
seed_demo_data()

# --------------------------- MODEL DATA ---------------------------
@st.cache_resource
def train_classifier():
    # Varied demonstration text generated from labelled sentence patterns.
    rng = np.random.default_rng(91)
    templates = {
        MODEL_CATEGORIES[0]: [
            "operator bypassed the safety interlock on the {machine}",
            "worker entered the danger zone while {machine} was operating",
            "lockout tagout was not applied before servicing the {machine}",
            "machine guard was removed during {task}",
            "emergency stop was blocked near the {machine}",
            "forklift was driven above the permitted speed in the {area}",
            "contractor restarted equipment without an approved permit"
        ],
        MODEL_CATEGORIES[1]: [
            "scaffold plank was loose and toe boards were missing at {area}",
            "worker was at height without connecting the fall arrest harness",
            "floor opening near {area} had no protective cover",
            "ladder was damaged and unsecured during overhead work",
            "guardrail was missing from the elevated {area}",
            "materials were stacked near an unprotected platform edge",
            "scaffold inspection tag was missing before access"
        ],
        MODEL_CATEGORIES[2]: [
            "chemical vapour was detected near the {equipment} valve",
            "solvent leak was observed below the {equipment} flange",
            "gas detector alarm activated in the {area}",
            "corrosion was found on the chemical transfer hose",
            "unknown liquid was leaking from a chemical container",
            "chlorine odour was reported near the storage manifold",
            "ventilation failed in the chemical handling room"
        ],
        MODEL_CATEGORIES[3]: [
            "damaged insulation exposed wires on the {equipment} cable",
            "electrical panel had exposed terminals near {area}",
            "extension cable was lying in water beside the {equipment}",
            "ground wire was disconnected from the welding machine",
            "burn marks were visible around the distribution board",
            "circuit breaker cover was missing in the utility room",
            "temporary power cable had exposed copper strands"
        ],
        MODEL_CATEGORIES[4]: [
            "routine shift handover was completed without abnormal findings",
            "work area was clean and walkways were unobstructed",
            "scheduled equipment inspection found no issues",
            "toolbox talk was completed before the shift started",
            "fire extinguishers were inspected and tags were current",
            "personal protective equipment stock was replenished",
            "operating readings remained within normal limits"
        ]
    }
    machines = ["press machine", "conveyor", "pump", "compressor", "forklift"]
    tasks = ["cleaning", "maintenance", "production", "troubleshooting"]
    areas = ["sector 4", "loading bay", "workshop", "tank farm", "assembly line"]
    equipment = ["pump", "welding unit", "control panel", "transfer line", "motor"]
    texts, labels = [], []
    for cat, patterns in templates.items():
        for _ in range(350):
            pattern = str(rng.choice(patterns))
            texts.append(pattern.format(machine=rng.choice(machines), task=rng.choice(tasks),
                                        area=rng.choice(areas), equipment=rng.choice(equipment)))
            labels.append(cat)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=.25, random_state=42, stratify=labels
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=6000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1200, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return pipe, X_test, y_test, pred

model, X_test, y_test, y_pred = train_classifier()

# --------------------------- INTELLIGENCE ---------------------------
def assess_risk(category, text):
    t = text.lower()
    if category == MODEL_CATEGORIES[4]:
        score = 10
    else:
        score = 38
    severe_terms = ["exposed", "gas detector alarm", "chlorine", "fall", "unprotected",
                    "electric shock", "fire", "explosion", "bypass", "without harness",
                    "moving equipment", "leak", "confined space"]
    if any(term in t for term in severe_terms):
        score += 32
    if any(term in t for term in ["near miss", "injury", "alarm", "failed", "damaged", "missing"]):
        score += 15
    score = min(score, 100)
    level = "Low" if score < 35 else "Medium" if score < 65 else "High"
    return score, level

def recommended_action(category):
    actions = {
        MODEL_CATEGORIES[0]: "Stop unsafe operation; isolate equipment; apply lockout/tagout; verify guards and permits before restart.",
        MODEL_CATEGORIES[1]: "Restrict access; install edge protection; inspect scaffold/ladder; verify fall-arrest equipment before work.",
        MODEL_CATEGORIES[2]: "Isolate the source if safe; follow site emergency procedure; use trained responders and verify atmospheric conditions.",
        MODEL_CATEGORIES[3]: "Keep personnel clear; isolate electrical supply through authorized staff; repair and test before reuse.",
        MODEL_CATEGORIES[4]: "Record the observation and continue routine monitoring."
    }
    return actions[category]

def get_alerts(df):
    alerts = []
    if df.empty:
        return alerts
    nonroutine = df[df["category"] != MODEL_CATEGORIES[4]]
    for site, group in nonroutine.groupby("site"):
        recent = group[pd.to_datetime(group["report_date"], errors="coerce") >= (pd.Timestamp.today().normalize()-pd.Timedelta(days=30))]
        high = int((recent["risk_level"] == "High").sum())
        total = len(recent)
        if high >= 3:
            alerts.append({"site": site, "level": "High", "count": high,
                           "message": f"{high} high-risk observations in the last 30 days."})
        elif total >= 5:
            alerts.append({"site": site, "level": "Medium", "count": total,
                           "message": f"{total} hazard reports in the last 30 days; review recurring patterns."})
    return alerts

def safe_date_series(df):
    out = df.copy()
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
    return out.dropna(subset=["report_date"])

# --------------------------- SIDEBAR ---------------------------
st.sidebar.title("🛡️ HSE Intelligence")
st.sidebar.caption("Safety Precursor Detection System")
page = st.sidebar.radio("Navigation", [
    "Executive Dashboard", "AI Report Analyzer", "Precursor Intelligence",
    "Site Risk Monitor", "Audit & Corrective Actions", "Model Evaluation"
])
st.sidebar.divider()
st.sidebar.caption("Data mode")
st.sidebar.info("Synthetic demonstration records are clearly labelled. Add verified site data for operational validation.")

df = read_reports()
if not df.empty:
    df_dates = safe_date_series(df)
else:
    df_dates = df

# --------------------------- DASHBOARD ---------------------------
if page == "Executive Dashboard":
    st.title("🛡️ Safety Precursor Intelligence")
    st.caption("HSE overview | Report-driven operational risk monitoring")
    total = len(df)
    high = int((df["risk_level"] == "High").sum()) if total else 0
    open_count = int((df["status"] != "Closed").sum()) if total else 0
    sites = int(df["site"].nunique()) if total else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total reports", f"{total:,}")
    c2.metric("High-risk reports", f"{high:,}")
    c3.metric("Open actions / reports", f"{open_count:,}")
    c4.metric("Sites represented", sites)
    st.divider()
    left,right = st.columns([1.3,1])
    with left:
        st.subheader("Report volume over time")
        if not df_dates.empty:
            daily = df_dates.groupby(df_dates["report_date"].dt.date).size().reset_index(name="Reports")
            st.line_chart(daily.set_index("report_date"))
        else:
            st.info("No dated reports available.")
    with right:
        st.subheader("Hazard category distribution")
        if total:
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category","Reports"]
            fig = px.pie(cat_counts, names="Category", values="Reports", hole=.45)
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Latest high-priority observations")
    latest = df[df["risk_level"] == "High"].head(8)
    if latest.empty:
        st.success("No high-risk reports currently recorded.")
    else:
        st.dataframe(latest[["report_code","report_date","site","category","risk_score","status","report_text"]], use_container_width=True, hide_index=True)
    st.caption("Dashboard values are calculated from records stored in the local SQLite database.")

# --------------------------- ANALYZER ---------------------------
elif page == "AI Report Analyzer":
    st.title("📝 AI Safety Report Analyzer")
    st.write("Submit a field observation. The NLP model classifies the text; the risk engine provides a separate rule-based screening score.")
    with st.form("report_form", clear_on_submit=False):
        col1,col2,col3 = st.columns(3)
        site = col1.selectbox("Site", SITES)
        dept = col2.selectbox("Department", DEPARTMENTS)
        shift = col3.selectbox("Shift", ["Day","Evening","Night"])
        col4,col5 = st.columns(2)
        report_date = col4.date_input("Observation date", value=date.today())
        equipment = col5.text_input("Equipment / area", "Conveyor")
        report_text = st.text_area("Field report", height=140,
            placeholder="Example: Worker noticed damaged insulation on a temporary cable near the pump...")
        submitted = st.form_submit_button("Analyze and save report", type="primary", use_container_width=True)
    if submitted:
        if len(report_text.strip()) < 12:
            st.error("Please enter a meaningful report (at least 12 characters).")
        else:
            probabilities = model.predict_proba([report_text])[0]
            order = np.argsort(probabilities)[::-1]
            category = model.classes_[order[0]]
            confidence = float(probabilities[order[0]])
            score, level = assess_risk(category, report_text)
            action = recommended_action(category)
            code = f"HSE-{datetime.now().strftime('%y%m%d%H%M%S')}"
            row = {
                "report_code": code, "report_date": report_date.isoformat(), "site": site,
                "department": dept, "shift": shift, "equipment": equipment,
                "report_text": report_text.strip(), "category": category,
                "risk_level": level, "risk_score": score, "model_confidence": confidence,
                "status": "Open", "corrective_action": action, "source": "User submitted",
                "is_incident": 0
            }
            try:
                save_report(row)
                st.session_state["last_analysis"] = row
                st.success(f"Report {code} analyzed and saved successfully.")
            except sqlite3.IntegrityError:
                st.error("Could not save this report because its report ID already exists. Please submit again.")
    if "last_analysis" in st.session_state:
        r = st.session_state["last_analysis"]
        st.divider()
        st.subheader("Analysis result")
        a,b,c = st.columns(3)
        a.metric("Predicted category", r["category"])
        b.metric("Risk level", r["risk_level"])
        c.metric("Risk screening score", f'{r["risk_score"]}/100')
        st.write(f"**Model confidence:** {r['model_confidence']:.1%}")
        st.write("**Recommended action:**", r["corrective_action"])
        st.caption("Confidence is the model's estimated class probability, not the probability that an accident will occur.")
        st.subheader("Class probabilities")
        probs = model.predict_proba([r["report_text"]])[0]
        prob_df = pd.DataFrame({"Category":model.classes_,"Probability":probs})
        st.bar_chart(prob_df.set_index("Category"))

# --------------------------- PRECURSOR INTELLIGENCE ---------------------------
elif page == "Precursor Intelligence":
    st.title("📈 Precursor Trend Intelligence")
    st.caption("Pattern-based screening of repeated hazards. Alerts indicate elevated reporting patterns, not certainty of a future accident.")
    alerts = get_alerts(df)
    if alerts:
        for alert in alerts:
            if alert["level"] == "High":
                st.error(f"🔴 {alert['site']}: {alert['message']}")
            else:
                st.warning(f"🟠 {alert['site']}: {alert['message']}")
    else:
        st.success("No site currently meets the configured alert thresholds.")
    st.subheader("Hazard reports by site and category")
    nonroutine = df[df["category"] != MODEL_CATEGORIES[4]]
    if not nonroutine.empty:
        pivot = pd.crosstab(nonroutine["site"], nonroutine["category"])
        st.dataframe(pivot, use_container_width=True)
        fig = px.bar(nonroutine, x="site", color="category", barmode="stack",
                     title="Recorded hazard observations by site")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Weekly hazard trend")
    if not df_dates.empty:
        hazards = df_dates[df_dates["category"] != MODEL_CATEGORIES[4]].copy()
        if not hazards.empty:
            hazards["week"] = hazards["report_date"].dt.to_period("W").astype(str)
            weekly = hazards.groupby("week").size().reset_index(name="Hazard reports")
            st.line_chart(weekly.set_index("week"))
        else:
            st.info("No hazard reports found.")
    st.info("Precursor lead time is not calculated from synthetic observations. To calculate it, import real dated precursor reports and subsequent incident records linked by site or event.")

# --------------------------- SITE MONITOR ---------------------------
elif page == "Site Risk Monitor":
    st.title("🚨 Site Risk Monitor")
    st.caption("Site risk indicators are calculated from saved report records.")
    if df.empty:
        st.info("No reports available.")
    else:
        summary = df.groupby("site").agg(
            Reports=("report_code","count"),
            High_Risk=("risk_level",lambda s: int((s=="High").sum())),
            Average_Score=("risk_score","mean"),
            Open=("status",lambda s: int((s!="Closed").sum()))
        ).reset_index()
        summary["Average_Score"] = summary["Average_Score"].round(1)
        summary["Risk_Band"] = summary["Average_Score"].apply(lambda x: "High" if x>=65 else "Medium" if x>=35 else "Low")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        fig = px.density_heatmap(df, x="site", y="category", z="risk_score",
                                 histfunc="avg", color_continuous_scale="RdYlGn_r",
                                 title="Average risk screening score by site and category")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Threshold-based alerts")
        for alert in get_alerts(df):
            if alert["level"] == "High": st.error(f"{alert['site']} — {alert['message']}")
            else: st.warning(f"{alert['site']} — {alert['message']}")

# --------------------------- AUDIT LOGS ---------------------------
elif page == "Audit & Corrective Actions":
    st.title("📋 Audit Logs & Corrective Actions")
    col1,col2,col3 = st.columns(3)
    search = col1.text_input("Search report text / ID")
    site_filter = col2.selectbox("Filter site", ["All"] + SITES)
    status_filter = col3.selectbox("Filter status", ["All","Open","In Progress","Closed"])
    filtered = df.copy()
    if search:
        mask = filtered["report_text"].str.contains(search, case=False, regex=False, na=False) | filtered["report_code"].str.contains(search, case=False, regex=False, na=False)
        filtered = filtered[mask]
    if site_filter != "All": filtered = filtered[filtered["site"] == site_filter]
    if status_filter != "All": filtered = filtered[filtered["status"] == status_filter]
    st.dataframe(filtered[["report_code","report_date","site","department","category","risk_level","risk_score","status","source","report_text"]], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export filtered audit logs (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="hse_audit_logs.csv", mime="text/csv")
    st.divider()
    st.subheader("Update report status")
    if not df.empty:
        selected = st.selectbox("Select report", df["report_code"].tolist())
        new_status = st.selectbox("New status", ["Open","In Progress","Closed"])
        if st.button("Save status update"):
            with db() as con:
                con.execute("UPDATE reports SET status=? WHERE report_code=?", (new_status,selected))
            st.success(f"{selected} updated to {new_status}.")
            st.rerun()
    else:
        st.info("No reports to update.")

# --------------------------- EVALUATION ---------------------------
elif page == "Model Evaluation":
    st.title("🧪 Model Evaluation")
    st.caption("Metrics below are calculated on a held-out split of the generated demonstration corpus. They are not real-world validation results.")
    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro")
    weighted = f1_score(y_test, y_pred, average="weighted")
    c1,c2,c3 = st.columns(3)
    c1.metric("Holdout accuracy", f"{acc:.3f}")
    c2.metric("Macro F1", f"{macro:.3f}")
    c3.metric("Weighted F1", f"{weighted:.3f}")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metric_df = pd.DataFrame(report).T
    st.subheader("Per-class precision, recall and F1")
    st.dataframe(metric_df.loc[MODEL_CATEGORIES, ["precision","recall","f1-score","support"]].round(3), use_container_width=True)
    cm = confusion_matrix(y_test, y_pred, labels=MODEL_CATEGORIES)
    fig = px.imshow(cm, x=MODEL_CATEGORIES, y=MODEL_CATEGORIES, text_auto=True,
                    color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(title="Confusion matrix", xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Keyword baseline comparison")
    KEYWORDS = {
        MODEL_CATEGORIES[0]: ["interlock","lockout","bypass","machine guard","forklift","emergency stop"],
        MODEL_CATEGORIES[1]: ["scaffold","harness","ladder","guardrail","floor opening","height"],
        MODEL_CATEGORIES[2]: ["chemical","gas","vapour","chlorine","solvent","leak","valve"],
        MODEL_CATEGORIES[3]: ["electrical","cable","wire","insulation","panel","grounding","circuit"],
        MODEL_CATEGORIES[4]: ["routine","handover","housekeeping","normal","no issues","toolbox talk"]
    }
    def keyword_predict(text):
        t = text.lower()
        scores = {cat: sum(1 for kw in kws if kw in t) for cat,kws in KEYWORDS.items()}
        return max(scores, key=scores.get)
    base_pred = [keyword_predict(x) for x in X_test]
    base_f1 = f1_score(y_test, base_pred, average="macro")
    comparison = pd.DataFrame({
        "Method":["Keyword baseline","TF-IDF + Logistic Regression"],
        "Macro F1":[base_f1,macro]
    })
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
    st.bar_chart(comparison.set_index("Method"))
    st.warning("Because the corpus is synthetic and generated from templates, these scores are only a software demonstration. For a defensible CA3 result, evaluate on manually labelled, unseen real reports.")

st.sidebar.divider()
st.sidebar.caption("Prototype v2.0 • Local SQLite storage")
