from pathlib import Path
import sys
from datetime import datetime
import html

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.pipeline import ensure_trained, analyze_message, load_and_clean

st.set_page_config(
    page_title="PhishShift | Security Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Polished, responsive dashboard styling ----
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#e8eef8; --muted:#9aabc3; --line:rgba(160,180,210,.17); --panel:#111d30; --panel2:#15243a; --teal:#57e0c0; --blue:#7aa7ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #172c48 0%, #0b1423 42%, #080f1a 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(8,15,26,.78); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101c2e,#0c1523); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:2.5rem; }
#MainMenu, footer {visibility:hidden;}
.hero { position:relative; overflow:hidden; border:1px solid rgba(122,167,255,.25); border-radius:24px; padding:28px 32px; background:linear-gradient(115deg,rgba(26,51,79,.96),rgba(16,35,55,.94) 55%,rgba(15,73,75,.75)); box-shadow:0 20px 60px rgba(0,0,0,.22); margin-bottom:20px; }
.hero:after {content:""; position:absolute; width:280px;height:280px;right:-60px;top:-150px;border-radius:50%;border:1px solid rgba(87,224,192,.22);box-shadow:0 0 0 28px rgba(87,224,192,.035),0 0 0 60px rgba(87,224,192,.025);}
.eyebrow {text-transform:uppercase;letter-spacing:2px;font-size:.70rem;font-weight:700;color:var(--teal);}
.hero h1 {font-family:'Manrope',sans-serif;font-size:2.35rem;font-weight:800;letter-spacing:-1.2px;color:#f5f8ff;margin:.25rem 0 .35rem;}
.hero p {color:#b7c8dc;margin:0;max-width:740px;font-size:1rem;}
.status-pill {display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border-radius:999px;border:1px solid rgba(87,224,192,.3);background:rgba(87,224,192,.08);color:#8cf1d9;font-size:.78rem;font-weight:700;white-space:nowrap;}
.status-dot {width:7px;height:7px;background:#57e0c0;border-radius:50%;box-shadow:0 0 12px #57e0c0;display:inline-block;}
.section-title {font-family:'Manrope',sans-serif;font-size:1.15rem;font-weight:800;color:#f2f6ff;margin:.3rem 0 .2rem;}
.section-sub {color:var(--muted);font-size:.87rem;margin-bottom:1rem;}
.metric-card {background:linear-gradient(145deg,rgba(21,36,58,.98),rgba(14,26,43,.98));border:1px solid var(--line);border-radius:17px;padding:17px 18px;min-height:112px;box-shadow:0 8px 26px rgba(0,0,0,.12);}
.metric-label {font-size:.78rem;color:#9eafc6;font-weight:600;letter-spacing:.2px;}
.metric-value {font-family:'Manrope',sans-serif;font-size:1.8rem;font-weight:800;color:#f2f7ff;line-height:1.35;margin-top:3px;}
.metric-foot {font-size:.75rem;color:#7f94b1;margin-top:3px;}
.panel {background:rgba(17,29,48,.78);border:1px solid var(--line);border-radius:18px;padding:20px 22px;margin:10px 0 16px;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#15243a,#101c2e);border:1px solid var(--line);padding:15px 17px;border-radius:15px;}
div[data-testid="stMetricLabel"] {color:#9eafc6;}
div[data-testid="stMetricValue"] {color:#f2f7ff;font-family:'Manrope',sans-serif;}
.stButton>button {border-radius:11px;border:1px solid rgba(87,224,192,.28);font-weight:700;min-height:2.8rem;transition:all .18s ease;}
.stButton>button[kind="primary"] {background:linear-gradient(100deg,#32c6ad,#55e0c0);color:#071820;border:0;}
.stButton>button:hover {border-color:#57e0c0;transform:translateY(-1px);}
.stTextArea textarea, .stTextInput input {background:#0b1626!important;color:#eaf1fc!important;border:1px solid rgba(150,175,210,.25)!important;border-radius:12px!important;}
.stSelectbox div[data-baseweb="select"] > div {background:#0b1626;border-color:rgba(150,175,210,.25);border-radius:10px;}
.stTabs [data-baseweb="tab-list"] {gap:8px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"] {background:rgba(17,29,48,.62);border:1px solid var(--line);border-radius:10px 10px 0 0;padding:10px 16px;color:#a9bad0;}
.stTabs [aria-selected="true"] {background:#17364a!important;color:#8cf1d9!important;border-color:rgba(87,224,192,.35)!important;}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
div[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:var(--line)!important;}
.small-note {color:#8fa2bc;font-size:.78rem;line-height:1.55;}
.risk-high {color:#ff8b9b;} .risk-medium {color:#ffd08a;} .risk-low {color:#7be7bd;} .risk-uncertain {color:#b7c9e3;}
@media(max-width:800px){.hero{padding:22px}.hero h1{font-size:1.8rem}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div style='font-size:2rem'>🛡️</div><div style='font-family:Manrope;font-size:1.25rem;font-weight:800'>PhishShift</div><div style='color:#8fa2bc;font-size:.8rem'>SECURITY INTELLIGENCE</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**WORKSPACE**")
    page = st.radio("Navigate", ["Overview", "Message Scanner", "Model Performance", "Methodology"], label_visibility="collapsed")
    st.divider()
    st.markdown("<div class='small-note'>● Local demo environment<br>Models run on your computer<br>No links are opened or fetched</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:24px;color:#657a96;font-size:.72rem'>ML-07 · CYBERSECURITY NLP<br>EDUCATIONAL PROTOTYPE</div>", unsafe_allow_html=True)

# Load artefacts
with st.spinner("Preparing local models and dashboard data…"):
    try:
        results, stats, train, test = ensure_trained()
        models_path = ROOT / "models" / "phishshift_models.joblib"
        import joblib
        models = joblib.load(models_path)
        df, cleaning_stats = load_and_clean()
        if not stats:
            stats = cleaning_stats
    except Exception as exc:
        st.error(f"Dashboard setup could not finish: {exc}")
        st.info("From the project folder, run setup.bat. If the error continues, check Python and the packages in requirements.txt.")
        st.stop()

results = results.copy()
for col in ["accuracy", "precision", "recall", "f1", "false_positive_rate"]:
    if col in results:
        results[col] = pd.to_numeric(results[col], errors="coerce")

# Hero header
left, right = st.columns([4, 1.25], vertical_alignment="center")
with left:
    st.markdown("<div class='hero'><div class='eyebrow'>ML-07 &nbsp; / &nbsp; CYBERSECURITY NLP</div><h1>Message Risk Intelligence</h1><p>Explore suspicious-message signals, review model performance, and test safe sample messages in one workspace.</p></div>", unsafe_allow_html=True)
with right:
    st.markdown("<div style='height:12px'></div><div class='status-pill'><span class='status-dot'></span> LOCAL SYSTEM READY</div><div class='small-note' style='margin-top:12px'>Model artefacts loaded<br>Offline message analysis</div>", unsafe_allow_html=True)

# Shared dataset stats
n_messages = len(df)
n_campaigns = int(df["campaign_group"].nunique()) if "campaign_group" in df else 0
n_phishing = int((df["label"] == 1).sum()) if "label" in df else 0
n_legit = int((df["label"] == 0).sum()) if "label" in df else 0
best_row = results.sort_values("f1", ascending=False).iloc[0] if len(results) and "f1" in results else None
best_name = str(best_row["model"]) if best_row is not None else "Not available"
best_f1 = float(best_row["f1"]) if best_row is not None else 0

if page == "Overview":
    st.markdown("<div class='section-title'>Live project overview</div><div class='section-sub'>A quick view of the current synthetic corpus and saved evaluation run.</div>", unsafe_allow_html=True)
    cards = st.columns(4)
    metrics = [
        ("CLEAN MESSAGES", f"{n_messages:,}", f"{stats.get('duplicates_removed', 0)} exact duplicates removed"),
        ("CAMPAIGN GROUPS", f"{n_campaigns:,}", "Used to reduce campaign leakage"),
        ("BEST HOLDOUT F1", f"{best_f1:.1%}", best_name),
        ("HOLDOUT SAMPLES", f"{int(results['test_rows'].max()) if len(results) and 'test_rows' in results else 0:,}", "Campaign-separated evaluation"),
    ]
    for col, (label, value, foot) in zip(cards, metrics):
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-foot'>{html.escape(str(foot))}</div></div>", unsafe_allow_html=True)

    # Put the live, percentage-formatted model table near the top so it is visible
    # before the visualisations and is not pushed below the fold.
    st.markdown("<div style='height:12px'></div><div class='section-title'>Live model comparison</div><div class='section-sub'>Interactive results from the campaign-separated holdout evaluation. Percentages are formatted for quick reading.</div>", unsafe_allow_html=True)
    if len(results):
        live_table = results.copy()
        for col in ["accuracy", "precision", "recall", "f1", "false_positive_rate"]:
            if col in live_table:
                live_table[col] = live_table[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        live_table = live_table.rename(columns={"model":"Model", "accuracy":"Accuracy", "precision":"Precision", "recall":"Recall", "f1":"F1 score", "false_positive_rate":"False-positive rate", "test_rows":"Test rows", "train_campaigns":"Train campaigns", "test_campaigns":"Test campaigns"})
        st.dataframe(live_table, width="stretch", hide_index=True, height=175)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    chart_col, detail_col = st.columns([1.35, 1], gap="large")
    with chart_col:
        st.markdown("<div class='panel'><div class='section-title'>Model performance snapshot</div><div class='section-sub'>Higher is better for F1, precision and recall. Lower is better for false-positive rate.</div>", unsafe_allow_html=True)
        if len(results):
            chart = results.set_index("model")[["precision", "recall", "f1", "false_positive_rate"]].rename(columns={"precision":"Precision", "recall":"Recall", "f1":"F1 score", "false_positive_rate":"False-positive rate"})
            fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#111d30")
            ax.set_facecolor("#111d30")
            chart.plot(kind="bar", ax=ax, width=.74, color=["#6fa8ff", "#59dfbd", "#b59aff", "#ffb86b"])
            ax.set_ylim(0, 1.08); ax.set_ylabel("Score / rate", color="#a9bad0"); ax.set_xlabel("")
            ax.tick_params(axis="x", colors="#d7e3f3", labelsize=8, rotation=10); ax.tick_params(axis="y", colors="#9aabc3")
            for spine in ax.spines.values(): spine.set_color("#2b3b52")
            ax.grid(axis="y", color="#304058", alpha=.55); ax.legend(facecolor="#15243a", labelcolor="#d7e3f3", fontsize=8, ncol=2, loc="upper center")
            fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with detail_col:
        st.markdown("<div class='panel'><div class='section-title'>Dataset composition</div><div class='section-sub'>Label balance after exact-text deduplication.</div>", unsafe_allow_html=True)
        label_counts = pd.Series({"Legitimate": n_legit, "Phishing": n_phishing})
        fig, ax = plt.subplots(figsize=(5, 3.25), facecolor="#111d30")
        ax.set_facecolor("#111d30")
        bars = ax.bar(label_counts.index, label_counts.values, color=["#6fa8ff", "#59dfbd"], width=.55)
        ax.set_ylabel("Messages", color="#a9bad0"); ax.tick_params(colors="#b7c8dc")
        for spine in ax.spines.values(): spine.set_color("#2b3b52")
        ax.grid(axis="y", color="#304058", alpha=.5)
        for bar, val in zip(bars, label_counts.values): ax.text(bar.get_x()+bar.get_width()/2, val, f"{val}  ({val/max(n_messages,1):.1%})", ha="center", va="bottom", color="#e8eef8", fontsize=9)
        fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
        st.markdown(f"<div class='small-note'>Legitimate: <b>{n_legit:,}</b> · Phishing: <b>{n_phishing:,}</b><br>Channels: {html.escape(', '.join(sorted(df['channel'].astype(str).unique())) if 'channel' in df else 'Not available')}</div></div>", unsafe_allow_html=True)

    st.warning("Research limitation: the bundled corpus is synthetic. These scores demonstrate the workflow only; they are not real-world phishing detection guarantees.")

elif page == "Message Scanner":
    st.markdown("<div class='section-title'>Inspect a message</div><div class='section-sub'>Paste a harmless sample to see the model's current assessment and warning cues.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        sample_choice = st.selectbox("Load a sample", ["Choose a sample…", "Suspicious account alert", "Legitimate college reminder", "Legitimate password reset", "Urgent payment review", "Unfamiliar short wording"])
        samples = {
            "Suspicious account alert": "URGENT: Your bank account will be blocked today. Verify your account immediately to avoid suspension. Link: hxxps://secure-login-example[.]com",
            "Legitimate college reminder": "Hi, your college assignment submission is scheduled for tomorrow. Please submit your work before 5 PM.",
            "Legitimate password reset": "Hello, we received a request to reset your password. If you made this request, follow the instructions on the official website. If not, you can ignore this message.",
            "Urgent payment review": "Action required: Your payment is pending. Please review your transaction status through the official application.",
            "Unfamiliar short wording": "Acct access review needed. Pls verify details soon.",
        }
        if sample_choice in samples and st.button("Use selected sample", width="stretch"): st.session_state["scan_text"] = samples[sample_choice]
        message = st.text_area("Message text", key="scan_text", height=220, placeholder="Paste an email or SMS body here…")
        analyze = st.button("Analyse message", type="primary", width="stretch")
        st.markdown("<div class='small-note'>Safety: do not paste real passwords, private messages, personal data or active suspicious links. This tool never visits URLs.</div></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='panel'><div class='section-title'>Assessment guide</div><div class='section-sub'>How to interpret the result</div><p><span class='risk-high'><b>HIGH</b></span> · Strong model signal; verify independently.</p><p><span class='risk-medium'><b>MEDIUM</b></span> · Review the context and sender.</p><p><span class='risk-low'><b>LOW</b></span> · No strong signal found; not proof of safety.</p><p><span class='risk-uncertain'><b>UNCERTAIN</b></span> · Model disagreement or borderline score.</p><hr><div class='small-note'>Risk score is an uncalibrated model score, not a probability that a message is malicious.</div></div>", unsafe_allow_html=True)
    if analyze:
        result = analyze_message(message, models)
        st.session_state["last_scan"] = {"time": datetime.now().strftime("%H:%M:%S"), "risk": result["risk"], "score": result["score"], "preview": (message[:90] + "…") if len(message) > 90 else message}
        if result["score"] is None:
            st.warning(result["note"])
        else:
            risk_class = {"High":"risk-high", "Medium":"risk-medium", "Low":"risk-low", "Uncertain":"risk-uncertain"}.get(result["risk"], "risk-uncertain")
            st.markdown(f"<div class='panel'><div class='eyebrow'>SCAN RESULT</div><div style='font-family:Manrope;font-size:2rem;font-weight:800' class='{risk_class}'>{html.escape(result['risk'])} risk</div><div style='color:#9aabc3'>Ensemble score · <b style='color:#f2f7ff'>{result['score']:.1f}%</b> <span style='font-size:.8rem'>(not a calibrated probability)</span></div></div>", unsafe_allow_html=True)
            st.markdown("**Signals found in the message**")
            for reason in result["reasons"]: st.markdown(f"- {reason}")
            st.info(result["note"])
    if "last_scan" in st.session_state:
        item = st.session_state["last_scan"]
        st.markdown("<div class='section-title'>Latest scan activity</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([item]).rename(columns={"time":"Time", "risk":"Risk category", "score":"Model score (%)", "preview":"Message preview"}), width="stretch", hide_index=True)

elif page == "Model Performance":
    st.markdown("<div class='section-title'>Evaluation lab</div><div class='section-sub'>Compare the rules baseline and two text-classification models on campaign-held-out data.</div>", unsafe_allow_html=True)
    if len(results):
        selected = st.selectbox("Highlight a model", results["model"].tolist())
        row = results[results["model"] == selected].iloc[0]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Accuracy", f"{row['accuracy']:.1%}")
        k2.metric("Precision", f"{row['precision']:.1%}")
        k3.metric("Recall", f"{row['recall']:.1%}")
        k4.metric("False-positive rate", f"{row['false_positive_rate']:.1%}")
        st.markdown("#### All models")
        st.dataframe(results, width="stretch", hide_index=True)
        st.markdown("#### Downloadable evaluation files")
        c1,c2 = st.columns(2)
        with c1:
            file = ROOT / "reports" / "model_results.csv"
            st.download_button("Download model results CSV", data=file.read_bytes(), file_name="phishshift_model_results.csv", mime="text/csv", width="stretch")
        with c2:
            file = ROOT / "reports" / "holdout_predictions.csv"
            if file.exists(): st.download_button("Download holdout predictions CSV", data=file.read_bytes(), file_name="phishshift_holdout_predictions.csv", mime="text/csv", width="stretch")
    st.caption("Campaign-separated evaluation reduces the chance that near-identical campaign templates appear in both train and test sets. Exact duplicates are removed; deeper near-duplicate similarity auditing remains a project limitation.")

else:
    st.markdown("<div class='section-title'>Methodology & responsible use</div><div class='section-sub'>A transparent summary of what this prototype does and what it does not claim.</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='panel'>
    <h4>01 · Input and preprocessing</h4><p>Message text is normalized. Exact duplicate message text is removed from the supplied corpus before model evaluation.</p>
    <h4>02 · Baseline and models</h4><p>A simple warning-cue rules baseline is compared with TF-IDF + Logistic Regression and TF-IDF + Multinomial Naive Bayes.</p>
    <h4>03 · Evaluation design</h4><p>Campaign groups are held out together, then accuracy, precision, recall, F1 score and false-positive rate are calculated.</p>
    <h4>04 · Risk communication</h4><p>The scanner displays a Low, Medium, High or Uncertain assessment with detected language cues. The score is not a calibrated probability.</p>
    <h4>05 · Safety boundaries</h4><p>The app does not click, fetch or execute URLs. Use only generated or otherwise authorised examples; never enter credentials or private mailbox content.</p>
    </div>
    """, unsafe_allow_html=True)
    st.warning("The included dataset is generated synthetic data. Before final academic submission, add source/licence documentation for any external corpus and test on a mentor-provided unseen campaign plus legitimate urgent messages.")

st.divider()
st.markdown("<div class='small-note'>PHISHSHIFT · ML-07 · Cybersecurity NLP + Robustness · Educational triage prototype only. A low-risk result is not a guarantee that a message is safe. Verify sensitive requests through a known official channel.</div>", unsafe_allow_html=True)
