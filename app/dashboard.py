import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.loader   import load_file, detect_label_column, preprocess
from src.pipeline import run_model, evaluate, UNSUPERVISED_MODELS, SUPERVISED_MODELS, ALL_MODELS

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Anomaly Detection System", page_icon="🔬", layout="wide")

# ══════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:#0d1117; color:#e6edf3; }
.main-title  { font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:#58a6ff; letter-spacing:-1px; }
.sub-title   { font-size:0.9rem; color:#8b949e; margin-top:4px; }
.sec-header  { font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#8b949e; text-transform:uppercase;
               letter-spacing:2px; border-bottom:1px solid #21262d; padding-bottom:8px; margin:28px 0 16px 0; }
.card        { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px 22px; text-align:center; }
.val         { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:600; color:#58a6ff; }
.lbl         { font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.green       { color:#3fb950!important; }
.red         { color:#f85149!important; }
.yellow      { color:#d29922!important; }
.model-card  { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 16px; margin-bottom:8px; }
.model-name  { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#58a6ff; font-weight:600; }
.model-desc  { font-size:0.78rem; color:#8b949e; margin-top:4px; }
.best-badge  { background:#1f6feb; color:white; border-radius:4px; padding:2px 8px;
               font-size:0.7rem; font-family:'IBM Plex Mono',monospace; margin-left:8px; }
.info-box    { background:#161b22; border-left:3px solid #58a6ff; border-radius:0 8px 8px 0;
               padding:10px 14px; font-size:0.85rem; color:#8b949e; margin:10px 0; }
div[data-testid="stFileUploader"] { border:2px dashed #30363d; border-radius:12px; padding:6px; background:#161b22; }
div[data-testid="stFileUploader"]:hover { border-color:#58a6ff; }
.stButton>button { background:#238636; color:white; border:none; border-radius:6px;
                   font-family:'IBM Plex Mono',monospace; font-size:0.85rem; padding:10px 20px; width:100%; }
.stButton>button:hover { background:#2ea043; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════
st.markdown('<div class="main-title">🔬 Machine Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload any dataset — compare multiple ML models — get instant results</div>', unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Upload Dataset")
    st.markdown('<div class="info-box">Supports: CSV, Excel, JSON, NPY, NPZ, Parquet, TSV, TXT, HDF5<br><br>For multi-file datasets (like MSL), upload all files together.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & Drop files here",
        type=["csv","xlsx","xls","json","npy","npz","parquet","tsv","txt","h5","hdf5"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    contamination = st.slider("Anomaly Rate", 0.01, 0.5, 0.1, 0.01,
                              help="Expected % of anomalies. Default 0.1 = 10%")

    is_timeseries = st.checkbox("Time Series Data", value=False,
                                help="Enable for sensor/telemetry data to keep repeated readings.")

    label_col_hint = st.text_input("Label column name (optional)",
                                   placeholder="e.g. faulty, label, anomaly")

    st.markdown("---")
    st.markdown("### 🤖 Select Models")
    st.markdown('<div class="info-box">Unsupervised: no labels needed<br>Supervised: requires label column</div>', unsafe_allow_html=True)

    st.markdown("**Unsupervised**")
    selected_unsupervised = []
    for name, desc in UNSUPERVISED_MODELS.items():
        checked = st.checkbox(name, value=(name == "Isolation Forest"), key=f"un_{name}",
                              help=desc)
        if checked:
            selected_unsupervised.append(name)

    st.markdown("**Supervised** *(needs labels)*")
    selected_supervised = []
    for name, desc in SUPERVISED_MODELS.items():
        checked = st.checkbox(name, value=False, key=f"su_{name}", help=desc)
        if checked:
            selected_supervised.append(name)

    run_btn = st.button("▶ Run Detection")

# ══════════════════════════════════════════════════════
#  WAITING STATE
# ══════════════════════════════════════════════════════
if not uploaded_files:
    c1, c2, c3 = st.columns(3)
    for col, icon, step, desc in zip(
        [c1,c2,c3],
        ["📁","⚙️","🚀"],
        ["Step 1","Step 2","Step 3"],
        ["Upload dataset file(s) from the sidebar",
         "Select models and adjust settings",
         "Click Run Detection — compare results instantly"]
    ):
        with col:
            st.markdown(f'<div class="card"><div style="font-size:2rem">{icon}</div>'
                        f'<div style="font-weight:600;margin-top:8px">{step}</div>'
                        f'<div style="color:#8b949e;font-size:0.82rem;margin-top:4px">{desc}</div></div>',
                        unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════
#  SHOW UPLOADED FILES
# ══════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Uploaded Files</div>', unsafe_allow_html=True)
fc = st.columns(max(len(uploaded_files), 1))
for i, f in enumerate(uploaded_files):
    with fc[i]:
        st.markdown(f'<div class="card"><div style="font-size:1.4rem">📄</div>'
                    f'<div class="model-name" style="margin-top:6px">{f.name}</div>'
                    f'<div style="color:#8b949e;font-size:0.75rem">{round(f.size/1024,1)} KB</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  HELPER — IDENTIFY FILE ROLES
# ══════════════════════════════════════════════════════
def identify_roles(files):
    roles = {"train": None, "test": None, "label": None, "single": None}
    if len(files) == 1:
        roles["single"] = files[0]
        return roles
    for f in files:
        n = f.name.lower()
        if "train" in n:   roles["train"] = f
        elif "label" in n: roles["label"] = f
        elif "test"  in n: roles["test"]  = f
    if not roles["train"] and not roles["test"]:
        roles["single"] = files[0]
    return roles

# ══════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════
if run_btn:
    selected_models = selected_unsupervised + selected_supervised
    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
        st.stop()

    with st.spinner("Loading and processing files..."):
        try:
            roles = identify_roles(uploaded_files)

            # ── MULTI-FILE (e.g. MSL) ─────────────────────────
            if roles["train"] and roles["test"]:
                train_raw = load_file(roles["train"], roles["train"].name)
                test_raw  = load_file(roles["test"],  roles["test"].name)

                y_true = None
                if roles["label"]:
                    label_data = np.load(roles["label"], allow_pickle=True)
                    y_true = pd.Series(label_data.astype(int))

                X_train = preprocess(train_raw, is_timeseries=True)
                X_test  = preprocess(test_raw,  is_timeseries=True)
                y_train = y_true.values[:len(X_train)] if y_true is not None else None
                base_df = test_raw.copy().reset_index(drop=True)

            # ── SINGLE FILE ───────────────────────────────────
            else:
                single = roles["single"] or uploaded_files[0]
                raw_df = load_file(single, single.name)
                hint   = label_col_hint.strip() or None
                label_name, y_true = detect_label_column(raw_df, hint=hint)

                feat_df = raw_df.copy()
                if label_name:
                    feat_df.drop(columns=[label_name], inplace=True)

                X_train = preprocess(feat_df, is_timeseries=is_timeseries)
                X_test  = X_train.copy()
                y_train = y_true.values if y_true is not None else None
                base_df = raw_df.copy().reset_index(drop=True)

        except Exception as e:
            st.error(f"❌ File loading error: {e}")
            st.stop()

    # ── RUN EACH SELECTED MODEL ───────────────────────
    all_results = {}
    progress    = st.progress(0, text="Running models...")

    for i, name in enumerate(selected_models):
        is_supervised = name in SUPERVISED_MODELS

        if is_supervised and y_train is None:
            st.warning(f"⚠️ **{name}** skipped — no label column found in dataset.")
            continue

        with st.spinner(f"Training {name}..."):
            try:
                preds, scores, t = run_model(
                    name, X_train, X_test,
                    y_train=y_train if is_supervised else None,
                    contamination=contamination
                )

                result = {"predictions": preds, "scores": scores, "train_time": t}

                if y_true is not None:
                    y_eval = y_true.values[:len(preds)]
                    result["metrics"] = evaluate(y_eval, preds, name, t)

                all_results[name] = result

            except Exception as e:
                st.error(f"❌ {name} failed: {e}")

        progress.progress((i + 1) / len(selected_models), text=f"Completed: {name}")

    progress.empty()

    if not all_results:
        st.error("No models ran successfully.")
        st.stop()

    st.session_state["all_results"] = all_results
    st.session_state["base_df"]     = base_df
    st.session_state["y_true"]      = y_true
    st.session_state["X_test"]      = X_test
    st.session_state["ready"]       = True

# ══════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════
if not st.session_state.get("ready"):
    st.stop()

all_results = st.session_state["all_results"]
base_df     = st.session_state["base_df"]
y_true      = st.session_state["y_true"]
model_names = list(all_results.keys())

# ── MODEL SELECTOR (if multiple ran) ─────────────────
st.markdown('<div class="sec-header">Results</div>', unsafe_allow_html=True)

active_model = st.selectbox(
    "View results for model:",
    model_names,
    label_visibility="visible"
)

res   = all_results[active_model]
preds = res["predictions"]
scores= res["scores"]
total = len(preds)
n_an  = int(preds.sum())
n_no  = total - n_an
rate  = round(n_an / total * 100, 2)

# ── METRICS ROW ───────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl, color in zip(
    [m1, m2, m3, m4],
    [f"{total:,}", f"{n_no:,}", f"{n_an:,}", f"{rate}%"],
    ["Total Records", "Normal", "Anomalies", "Anomaly Rate"],
    ["#58a6ff", "#3fb950", "#f85149", "#d29922"]
):
    with col:
        st.markdown(f'<div class="card"><div class="val" style="color:{color}">{val}</div>'
                    f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ── EVALUATION METRICS ────────────────────────────────
if "metrics" in res:
    m = res["metrics"]
    st.markdown('<div class="sec-header">Model Evaluation</div>', unsafe_allow_html=True)

    e1, e2, e3, e4, e5 = st.columns(5)
    for col, key, lbl in zip(
        [e1, e2, e3, e4, e5],
        ["accuracy", "precision", "recall", "f1_score", "train_time"],
        ["Accuracy", "Precision", "Recall", "F1 Score", "Train Time (s)"]
    ):
        with col:
            val = m[key]
            if key == "train_time":
                color = "#8b949e"
                disp  = f"{val}s"
            else:
                color = "#3fb950" if val >= 0.7 else "#d29922" if val >= 0.5 else "#f85149"
                disp  = str(val)
            st.markdown(f'<div class="card"><div class="val" style="color:{color}">{disp}</div>'
                        f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  MODEL COMPARISON TABLE (if multiple models ran)
# ══════════════════════════════════════════════════════
if len(all_results) > 1 and any("metrics" in v for v in all_results.values()):
    st.markdown('<div class="sec-header">Model Comparison</div>', unsafe_allow_html=True)

    rows = []
    for name, r in all_results.items():
        if "metrics" in r:
            m = r["metrics"]
            rows.append({
                "Model"      : name,
                "Accuracy"   : m["accuracy"],
                "Precision"  : m["precision"],
                "Recall"     : m["recall"],
                "F1 Score"   : m["f1_score"],
                "Train Time" : f"{m['train_time']}s",
                "Type"       : "Supervised" if name in SUPERVISED_MODELS else "Unsupervised"
            })

    if rows:
        comp_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)

        # Highlight best F1
        best_f1    = comp_df["F1 Score"].max()
        best_model = comp_df[comp_df["F1 Score"] == best_f1]["Model"].values[0]

        st.markdown(f'<div class="info-box">🏆 Best model by F1 Score: <b style="color:#58a6ff">{best_model}</b> ({best_f1})</div>', unsafe_allow_html=True)

        def color_f1(val):
            if isinstance(val, float):
                if val >= 0.7: return "color: #3fb950"
                if val >= 0.5: return "color: #d29922"
                return "color: #f85149"
            return ""

        st.dataframe(
            comp_df.style.applymap(color_f1, subset=["Accuracy","Precision","Recall","F1 Score"]),
            use_container_width=True,
            hide_index=True
        )

        # Bar chart comparison
        st.markdown('<div class="sec-header">F1 Score Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        colors = ["#3fb950" if v >= 0.7 else "#d29922" if v >= 0.5 else "#f85149"
                  for v in comp_df["F1 Score"]]
        bars = ax.barh(comp_df["Model"], comp_df["F1 Score"], color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 Score", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        for bar, val in zip(bars, comp_df["F1 Score"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", color="#e6edf3", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════
#  CONFUSION MATRIX + CHARTS
# ══════════════════════════════════════════════════════
viz_col1, viz_col2 = st.columns(2)

# Confusion matrix
if "metrics" in res:
    with viz_col1:
        st.markdown('<div class="sec-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(res["metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3), facecolor="#161b22")
        ax.set_facecolor("#161b22")
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Normal","Anomaly"], color="#8b949e")
        ax.set_yticklabels(["Normal","Anomaly"], color="#8b949e")
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual", color="#8b949e")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                        color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Pie chart
with viz_col2:
    st.markdown('<div class="sec-header">Anomaly Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    ax.pie([n_no, n_an], labels=["Normal","Anomaly"],
           colors=["#3fb950","#f85149"], autopct="%1.1f%%",
           startangle=90, textprops={"color":"#e6edf3"})
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Score distribution
st.markdown('<div class="sec-header">Anomaly Score Distribution</div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 3), facecolor="#161b22")
ax.set_facecolor("#161b22")
ax.hist(scores[preds == 0], bins=50, alpha=0.7, color="#3fb950", label="Normal")
ax.hist(scores[preds == 1], bins=50, alpha=0.7, color="#f85149", label="Anomaly")
ax.set_xlabel("Score", color="#8b949e")
ax.set_ylabel("Count",  color="#8b949e")
ax.set_title(f"{active_model} — Score Distribution", color="#e6edf3")
ax.tick_params(colors="#8b949e")
ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Sensor charts
sensor_cols = [c for c in ["temperature","pressure","vibration","humidity"] if c in base_df.columns]
if sensor_cols:
    st.markdown('<div class="sec-header">Sensor Analysis</div>', unsafe_allow_html=True)
    sc = st.columns(len(sensor_cols))
    result_df_temp = base_df.copy()
    result_df_temp["predicted_anomaly"] = preds
    for i, col in enumerate(sensor_cols):
        with sc[i]:
            fig, ax = plt.subplots(figsize=(4,3), facecolor="#161b22")
            ax.set_facecolor("#161b22")
            ax.hist(result_df_temp[result_df_temp["predicted_anomaly"]==0][col], bins=25, alpha=0.7, color="#3fb950", label="Normal")
            ax.hist(result_df_temp[result_df_temp["predicted_anomaly"]==1][col], bins=25, alpha=0.7, color="#f85149", label="Anomaly")
            ax.set_title(col.capitalize(), color="#e6edf3", fontsize=10)
            ax.tick_params(colors="#8b949e", labelsize=7)
            ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3")
            for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════
#  PREDICTION TABLE
# ══════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Prediction Table</div>', unsafe_allow_html=True)

result_df = base_df.copy()
result_df["predicted_anomaly"] = preds
result_df["anomaly_score"]     = scores

filter_opt = st.selectbox("Filter", ["All records","Anomalies only","Normal only"],
                          label_visibility="collapsed")

display_df = result_df.copy()
if filter_opt == "Anomalies only": display_df = display_df[display_df["predicted_anomaly"]==1]
elif filter_opt == "Normal only":  display_df = display_df[display_df["predicted_anomaly"]==0]

def highlight(row):
    if row.get("predicted_anomaly", 0) == 1:
        return ["background-color:#2d0f0f;color:#f85149"] * len(row)
    return [""] * len(row)

st.dataframe(
    display_df.head(500).style.apply(highlight, axis=1),
    use_container_width=True, height=350
)
if len(display_df) > 500:
    st.caption(f"Showing first 500 of {len(display_df):,} rows.")

# Download
csv = result_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Results as CSV", csv, "anomaly_predictions.csv", "text/csv")