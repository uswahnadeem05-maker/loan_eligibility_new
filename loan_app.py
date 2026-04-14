import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Loan Eligibility System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-box {
        background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%);
        padding: 2.5rem 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(26,35,126,0.2);
    }
    .hero-box h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-box p  { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #1565c0;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin:0; color:#1565c0; font-size:0.78rem; letter-spacing:1px; text-transform:uppercase; }
    .metric-card p  { margin:0.3rem 0 0 0; font-size:1.5rem; font-weight:700; color:#1a237e; }

    .result-eligible   { background:#e8f5e9; border-left:6px solid #2e7d32; border-radius:12px; padding:1.2rem 1.5rem; }
    .result-medium     { background:#fff8e1; border-left:6px solid #f9a825; border-radius:12px; padding:1.2rem 1.5rem; }
    .result-ineligible { background:#ffebee; border-left:6px solid #c62828; border-radius:12px; padding:1.2rem 1.5rem; }

    .section-title {
        font-size: 1.2rem; font-weight: 700; color: #1a237e;
        border-bottom: 3px solid #1565c0;
        padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0288d1);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>🏦 Loan Eligibility Prediction System</h1>
    <p>Smart dataset-driven assessment · 4,269 real applications · Instant decision</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    xlsx_path = "loan_approval_dataset.xlsx"
    csv_path  = "loan_approval_dataset.csv"

    if os.path.exists(xlsx_path):
        df = pd.read_excel(xlsx_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error("❌ Dataset not found! Place loan_approval_dataset.xlsx or .csv in the same folder.")
        st.stop()

    df.columns = df.columns.str.strip()
    rename = {
        "income_annum":             "Income",
        "loan_amount":              "Loan_Amount",
        "cibil_score":              "Credit_Score",
        "loan_status":              "Actual_Status",
        "no_of_dependents":         "Dependents",
        "loan_term":                "Loan_Term",
        "residential_assets_value": "Residential_Assets",
        "commercial_assets_value":  "Commercial_Assets",
        "luxury_assets_value":      "Luxury_Assets",
        "bank_asset_value":         "Bank_Assets",
        "education":                "Education",
        "self_employed":            "Self_Employed",
        "loan_id":                  "Loan_ID",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["Actual_Status"] = df["Actual_Status"].astype(str).str.strip().str.capitalize()
    df["Education"]     = df["Education"].astype(str).str.strip()
    df["Self_Employed"] = df["Self_Employed"].astype(str).str.strip()
    return df

df = load_data()

# ─────────────────────────────────────────
# DYNAMIC THRESHOLDS
# ─────────────────────────────────────────
income_thresh = df["Income"].quantile(0.5)
credit_high   = df["Credit_Score"].quantile(0.75)
credit_mid    = df["Credit_Score"].quantile(0.5)

# ─────────────────────────────────────────
# ELIGIBILITY SCORING
# ─────────────────────────────────────────
def eligibility_score(credit, income, loan_amount, education, self_employed, dependents):
    pts = 0
    reasons = []

    if credit >= credit_high:
        pts += 35
        reasons.append(("✅", f"Excellent CIBIL score ({int(credit)})"))
    elif credit >= credit_mid:
        pts += 18
        reasons.append(("⚠️", f"Average CIBIL score ({int(credit)})"))
    else:
        reasons.append(("❌", f"Low CIBIL score ({int(credit)})"))

    if income >= income_thresh:
        pts += 25
        reasons.append(("✅", f"Income above median (Rs. {int(income):,})"))
    elif income >= income_thresh * 0.5:
        pts += 12
        reasons.append(("⚠️", "Income slightly below median"))
    else:
        reasons.append(("❌", f"Very low income (Rs. {int(income):,})"))

    ratio = loan_amount / income if income > 0 else 99
    if ratio <= 3:
        pts += 25
        reasons.append(("✅", f"Healthy loan/income ratio ({ratio:.1f}x)"))
    elif ratio <= 6:
        pts += 12
        reasons.append(("⚠️", f"Moderate loan/income ratio ({ratio:.1f}x)"))
    else:
        reasons.append(("❌", f"Very high loan/income ratio ({ratio:.1f}x)"))

    if str(education).lower() == "graduate":
        pts += 8
        reasons.append(("✅", "Graduate — adds credibility"))
    else:
        reasons.append(("🔸", "Not Graduate — minor impact"))

    if str(self_employed).lower() == "no":
        pts += 4
        reasons.append(("✅", "Salaried — stable income"))
    else:
        reasons.append(("🔸", "Self-employed — income may vary"))

    if dependents <= 2:
        pts += 3
        reasons.append(("✅", f"Low dependents ({dependents})"))
    else:
        reasons.append(("🔸", f"High dependents ({dependents})"))

    label = "Eligible" if pts >= 70 else ("Medium Risk" if pts >= 42 else "Not Eligible")
    return label, pts, reasons

def check_row(row):
    label, _, _ = eligibility_score(
        row["Credit_Score"], row["Income"], row["Loan_Amount"],
        row.get("Education", "Graduate"), row.get("Self_Employed", "No"),
        row.get("Dependents", 0)
    )
    return label

df["Predicted_Status"] = df.apply(check_row, axis=1)

# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
approved_mask = df["Actual_Status"].str.contains("approved", case=False, na=False)
rejected_mask = df["Actual_Status"].str.contains("rejected", case=False, na=False)
eligible_mask = df["Predicted_Status"] == "Eligible"

benefit      = df[eligible_mask & approved_mask]["Loan_Amount"].sum()
loss         = df[eligible_mask & rejected_mask]["Loan_Amount"].sum()
total_apps   = len(df)
approved_cnt = approved_mask.sum()
approved_pct = round(approved_cnt / total_apps * 100, 1)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Dataset Overview")
    st.info(f"**{total_apps:,}** total applications")
    st.success(f"**{approved_cnt:,}** Approved ({approved_pct}%)")
    st.error(f"**{rejected_mask.sum():,}** Rejected ({100-approved_pct}%)")
    st.markdown("---")
    st.markdown("### 📐 Dynamic Thresholds")
    st.write(f"💰 Median Income    : **Rs. {int(income_thresh):,}**")
    st.write(f"⭐ High CIBIL (75%) : **{int(credit_high)}**")
    st.write(f"🔸 Mid  CIBIL (50%) : **{int(credit_mid)}**")
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.write(f"Avg CIBIL  : **{int(df['Credit_Score'].mean())}**")
    st.write(f"Avg Income : **Rs. {int(df['Income'].mean()):,}**")
    st.write(f"Avg Loan   : **Rs. {int(df['Loan_Amount'].mean()):,}**")
    st.write(f"Avg Term   : **{int(df['Loan_Term'].mean())} yrs**")
    st.markdown("---")
    st.markdown("### 🎓 Education")
    for edu, cnt in df["Education"].value_counts().items():
        st.write(f"{edu}: **{cnt:,}** ({round(cnt/total_apps*100,1)}%)")
    st.markdown("### 💼 Employment")
    for emp, cnt in df["Self_Employed"].value_counts().items():
        lbl = "Self Employed" if emp == "Yes" else "Salaried"
        st.write(f"{lbl}: **{cnt:,}** ({round(cnt/total_apps*100,1)}%)")

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
st.markdown('<div class="section-title">🧾 Check Applicant Eligibility</div>', unsafe_allow_html=True)

with st.form("loan_form"):
    r1, r2, r3 = st.columns(3)
    income_input = r1.number_input("💰 Annual Income (Rs.)", min_value=200000,  max_value=9900000,  step=100000, value=5000000)
    loan_input   = r2.number_input("📄 Loan Amount (Rs.)",   min_value=300000,  max_value=39500000, step=100000, value=10000000)
    credit_input = r3.number_input("📊 CIBIL Score",         min_value=300,     max_value=900,      step=10,     value=650)

    r4, r5, r6 = st.columns(3)
    edu_input = r4.selectbox("🎓 Education",      ["Graduate", "Not Graduate"])
    emp_input = r5.selectbox("💼 Self Employed?", ["No", "Yes"])
    dep_input = r6.number_input("👨‍👩‍👧 Dependents", min_value=0, max_value=5, step=1, value=1)

    submitted = st.form_submit_button("🚀 Check Eligibility Now")

if submitted:
    label, score_pct, reasons = eligibility_score(
        credit_input, income_input, loan_input, edu_input, emp_input, dep_input
    )
    st.markdown("---")
    col_res, col_det = st.columns([1.2, 1])

    with col_res:
        if label == "Eligible":
            st.markdown("""<div class="result-eligible">
                <h2 style="color:#2e7d32;margin:0">✅ Eligible for Loan</h2>
                <p style="margin:0.4rem 0 0 0;color:#1b5e20">Applicant meets all key criteria.</p>
            </div>""", unsafe_allow_html=True)
        elif label == "Medium Risk":
            st.markdown("""<div class="result-medium">
                <h2 style="color:#e65100;margin:0">⚠️ Medium Risk</h2>
                <p style="margin:0.4rem 0 0 0;color:#bf360c">May qualify with collateral or guarantor.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-ineligible">
                <h2 style="color:#c62828;margin:0">❌ Not Eligible</h2>
                <p style="margin:0.4rem 0 0 0;color:#b71c1c">Does not meet minimum criteria.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Approval Score: {score_pct} / 100**")
        st.progress(min(score_pct / 100, 1.0))
        ratio = loan_input / income_input if income_input > 0 else 0
        st.markdown(f"""
        | Detail | Value |
        |---|---|
        | 💰 Income | Rs. {income_input:,} |
        | 📄 Loan Amount | Rs. {loan_input:,} |
        | 📊 CIBIL Score | {credit_input} |
        | 🔁 Loan/Income Ratio | {ratio:.2f}x |
        | 🎓 Education | {edu_input} |
        | 💼 Self Employed | {emp_input} |
        | 👨‍👩‍👧 Dependents | {dep_input} |
        """)

    with col_det:
        st.markdown("**📝 Detailed Assessment**")
        for icon, msg in reasons:
            st.markdown(f"{icon} {msg}")

st.markdown("---")

# ─────────────────────────────────────────
# FINANCIAL OVERVIEW
# ─────────────────────────────────────────
st.markdown('<div class="section-title">💰 Financial Overview</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric-card"><h3>✅ Total Benefit</h3><p>Rs. {benefit/1e7:.1f} Cr</p></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><h3>⚠️ Risk Exposure</h3><p>Rs. {loss/1e7:.1f} Cr</p></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><h3>📋 Applications</h3><p>{total_apps:,}</p></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-card"><h3>📈 Approval Rate</h3><p>{approved_pct}%</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# VISUALIZATIONS (no heatmap)
# ─────────────────────────────────────────
st.markdown('<div class="section-title">📊 Data Visualizations</div>', unsafe_allow_html=True)

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"Approved":"#2e7d32","Rejected":"#c62828",
          "Eligible":"#2e7d32","Medium Risk":"#f9a825","Not Eligible":"#c62828"}

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**📌 Actual Loan Status**")
    counts = df["Actual_Status"].value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
           colors=[COLORS.get(k,"#90a4ae") for k in counts.index],
           startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
    ax.set_title("Actual Loan Status", fontweight="bold")
    st.pyplot(fig); plt.close(fig)

with rc2:
    st.markdown("**📌 Predicted Eligibility**")
    pcounts = df["Predicted_Status"].value_counts()
    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.bar(pcounts.index, pcounts.values,
                  color=[COLORS.get(k,"#90a4ae") for k in pcounts.index],
                  edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
                f'{int(bar.get_height()):,}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title("Predicted Eligibility", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

rc3, rc4 = st.columns(2)
with rc3:
    st.markdown("**📌 CIBIL Score Distribution**")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(df["Credit_Score"], bins=35, color="#1565c0", edgecolor="white", alpha=0.85)
    ax.axvline(credit_high, color="#2e7d32", linestyle="--", lw=2, label=f"High ({int(credit_high)})")
    ax.axvline(credit_mid,  color="#f9a825", linestyle="--", lw=2, label=f"Mid ({int(credit_mid)})")
    ax.set_xlabel("CIBIL Score"); ax.set_ylabel("Count")
    ax.set_title("CIBIL Score Distribution", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with rc4:
    st.markdown("**📌 Income vs Loan Amount**")
    fig, ax = plt.subplots(figsize=(5,4))
    for status, grp in df.groupby("Predicted_Status"):
        ax.scatter(grp["Income"]/1e5, grp["Loan_Amount"]/1e5,
                   label=status, alpha=0.35, s=10, color=COLORS.get(status,"gray"))
    ax.set_xlabel("Income (Lakh Rs.)"); ax.set_ylabel("Loan Amount (Lakh Rs.)")
    ax.set_title("Income vs Loan Amount", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

rc5, rc6 = st.columns(2)
with rc5:
    st.markdown("**📌 Education vs Loan Status**")
    edu_stat = df.groupby(["Education","Actual_Status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5,4))
    edu_stat.plot(kind="bar", ax=ax,
                  color=[COLORS.get(c,"#90a4ae") for c in edu_stat.columns],
                  edgecolor="white", width=0.6)
    ax.set_title("Education vs Loan Status", fontweight="bold")
    ax.set_xlabel(""); ax.tick_params(axis='x', rotation=15)
    ax.legend(title="Status"); ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with rc6:
    st.markdown("**📌 Self Employment vs Loan Status**")
    emp_stat = df.groupby(["Self_Employed","Actual_Status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5,4))
    emp_stat.plot(kind="bar", ax=ax,
                  color=[COLORS.get(c,"#90a4ae") for c in emp_stat.columns],
                  edgecolor="white", width=0.5)
    ax.set_title("Self Employed vs Loan Status", fontweight="bold")
    ax.set_xlabel(""); ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Status"); ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

rc7, rc8 = st.columns(2)
with rc7:
    st.markdown("**📌 Dependents vs Loan Status**")
    dep_stat = df.groupby(["Dependents","Actual_Status"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5,4))
    dep_stat.plot(kind="bar", ax=ax,
                  color=[COLORS.get(c,"#90a4ae") for c in dep_stat.columns],
                  edgecolor="white", width=0.6)
    ax.set_title("Dependents vs Loan Status", fontweight="bold")
    ax.set_xlabel("No. of Dependents"); ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Status"); ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

with rc8:
    st.markdown("**📌 Loan Term Distribution**")
    term_counts = df["Loan_Term"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(term_counts.index.astype(str), term_counts.values,
           color="#0288d1", edgecolor="white", width=0.6)
    ax.set_xlabel("Loan Term (Years)"); ax.set_ylabel("Count")
    ax.set_title("Loan Term Distribution", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig)

st.markdown("---")

# ─────────────────────────────────────────
# FULL DATASET PREVIEW — ALL 4269 ROWS
# ─────────────────────────────────────────
st.markdown('<div class="section-title">📋 Full Dataset Preview — All 4,269 Rows</div>', unsafe_allow_html=True)

show_cols = ["Loan_ID","Dependents","Education","Self_Employed",
             "Income","Loan_Amount","Loan_Term","Credit_Score",
             "Residential_Assets","Commercial_Assets","Luxury_Assets",
             "Bank_Assets","Actual_Status","Predicted_Status"]
show_cols = [c for c in show_cols if c in df.columns]

# Filters
st.markdown("**🔍 Filter Options**")
fc1, fc2, fc3 = st.columns(3)
status_filter = fc1.selectbox("Actual Status",    ["All","Approved","Rejected"])
pred_filter   = fc2.selectbox("Predicted Status", ["All","Eligible","Medium Risk","Not Eligible"])
edu_filter    = fc3.selectbox("Education",        ["All","Graduate","Not Graduate"])

filtered_df = df[show_cols].copy()
if status_filter != "All":
    filtered_df = filtered_df[filtered_df["Actual_Status"].str.contains(status_filter, case=False)]
if pred_filter != "All":
    filtered_df = filtered_df[filtered_df["Predicted_Status"] == pred_filter]
if edu_filter != "All":
    filtered_df = filtered_df[filtered_df["Education"] == edu_filter]

st.write(f"Showing **{len(filtered_df):,}** of **{total_apps:,}** rows")

def color_pred(val):
    if val == "Eligible":     return "background-color:#e8f5e9;color:#2e7d32;font-weight:600"
    if val == "Medium Risk":  return "background-color:#fff8e1;color:#e65100;font-weight:600"
    if val == "Not Eligible": return "background-color:#ffebee;color:#c62828;font-weight:600"
    return ""

def color_actual(val):
    v = str(val).lower()
    if "approved" in v: return "background-color:#e8f5e9;color:#2e7d32;font-weight:600"
    if "rejected" in v: return "background-color:#ffebee;color:#c62828;font-weight:600"
    return ""

styled_df = (filtered_df.style
             .applymap(color_pred,   subset=["Predicted_Status"])
             .applymap(color_actual, subset=["Actual_Status"]))

# Full scrollable table — all rows visible
st.dataframe(styled_df, use_container_width=True, height=600)

st.markdown("---")

# ─────────────────────────────────────────
# PROJECT EXPLANATION
# ─────────────────────────────────────────
st.markdown('<div class="section-title">📘 Project Explanation</div>', unsafe_allow_html=True)

with st.expander("🔹 About this Dataset", expanded=True):
    st.markdown(f"""
    Real dataset with **{total_apps:,} loan applications** — all 13 columns used.

    | Column | Description |
    |---|---|
    | loan_id | Unique serial number |
    | no_of_dependents | Family members dependent (0–5) |
    | education | Graduate / Not Graduate |
    | self_employed | Yes / No |
    | income_annum | Yearly income Rs. (2L – 99L) |
    | loan_amount | Loan requested Rs. (3L – 3.95Cr) |
    | loan_term | Repayment years (2–20) |
    | cibil_score | Credit score (300–900) |
    | residential_assets_value | Home/property value |
    | commercial_assets_value | Shop/office value |
    | luxury_assets_value | Car/jewellery value |
    | bank_asset_value | Bank savings/FD |
    | loan_status | ✅ Approved / ❌ Rejected |
    """)

with st.expander("🔹 Scoring Logic"):
    st.markdown(f"""
    | Factor | Condition | Points |
    |---|---|---|
    | CIBIL Score | ≥ {int(credit_high)} (75th %ile) | 35 |
    | CIBIL Score | ≥ {int(credit_mid)} (50th %ile) | 18 |
    | Income | ≥ Median Rs. {int(income_thresh):,} | 25 |
    | Loan/Income | ≤ 3x | 25 |
    | Loan/Income | ≤ 6x | 12 |
    | Education | Graduate | 8 |
    | Employment | Salaried | 4 |
    | Dependents | ≤ 2 | 3 |

    **Score ≥ 70 → ✅ Eligible &nbsp;|&nbsp; 42–69 → ⚠️ Medium Risk &nbsp;|&nbsp; < 42 → ❌ Not Eligible**
    """)

with st.expander("🔹 Key Features"):
    st.markdown("""
    - ✅ Reads your **real 4,269-row dataset** (xlsx or csv)
    - ✅ **Dynamic thresholds** — auto-adjusts to any dataset
    - ✅ **6-factor scoring** with detailed breakdown
    - ✅ **8 charts** — full visual analysis
    - ✅ **Full dataset table** — all 4,269 rows visible with scroll
    - ✅ **3 filter options** — filter by status, prediction & education
    - ✅ **Color-coded** Approved/Rejected/Eligible rows
    """)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
    "🏦 Loan Eligibility Prediction System · Streamlit · 4,269 Real Applications"
    "</div>", unsafe_allow_html=True
)
