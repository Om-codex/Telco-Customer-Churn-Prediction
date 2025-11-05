import streamlit as st
from streamlit_option_menu import option_menu
import base64
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import joblib
from PIL import Image
from io import BytesIO
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,precision_score,recall_score

# Convert image to base64
def image_to_base64(img_path):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

img_base64 = image_to_base64("churn_banner.png")
# --- CUSTOM STYLES ---
st.markdown(f"""
<style>
/* ====== GLOBAL BACKGROUND ====== */
.stApp {{
    background: radial-gradient(circle at top left, rgba(255,255,255,0.06), transparent 60%), 
                linear-gradient(135deg, #0A0A0A, #1A1A1A);
    color: #F0F0F0;
}}

.css-18e3th9, .css-1d391kg {{
    background-color: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(6px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}}
            
/* Change the sidebar background color */
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.04) !important;
}}
        

            
/* ====== HERO BANNER ====== */
.hero-banner {{
    position: relative;
    width: 100%;
    height: 320px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 5px 25px rgba(0,0,0,0.4);
    margin-bottom: 2rem;
    background: url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    filter: brightness(80%);
}}

.hero-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(15, 23, 42, 0.6);
    z-index: 1;
}}

.hero-text {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #e0f2fe;
    text-align: center;
    z-index: 2;
    opacity: 0;
    animation: fadeInGlow 1.8s ease-out forwards;
}}


.hero-text h1 {{
    font-size: 50px;
    font-weight: 800;
    text-shadow: 0 0 20px rgba(56,189,248,0.8), 0 0 40px rgba(37,99,235,0.6);
}}

/* ===== Cinematic Fade + Lift + Glow ===== */
@keyframes fadeInGlow {{
    0% {{
        opacity: 0;
        transform: translate(-50%, calc(-50% + 50px)) scale(0.98);
        text-shadow: 0 0 0 rgba(96,165,250,0);
    }}
    60% {{
        opacity: 1;
        transform: translate(-50%, -50%) scale(1);
        text-shadow: 0 0 25px rgba(96,165,250,0.7);
    }}
    80% {{
        text-shadow: 0 0 40px rgba(96,165,250,0.5), 0 0 70px rgba(59,130,246,0.3);
    }}
    100% {{
        opacity: 1;
        transform: translate(-50%, -50%) scale(1);
        text-shadow: 0 0 25px rgba(56,189,248,0.6);
    }}
}}

/* ===== BUTTON STYLE (your base + futuristic colors) ===== */
.stButton>button {{
    background-color: #3B82F6
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.75rem 2rem;
    box-shadow: 0 0 12px rgba(56,189,248,0.6);
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #3B82F6, #60A5FA) !important;
    transform: translateY(-2px);
    box-shadow: 0 0 18px rgba(96,165,250,0.4);
    color: black;
}}
/* ===== FORM SUBMIT BUTTON STYLE (same design) ===== */
.stForm .stButton>button {{
    background: linear-gradient(90deg, #38bdf8, #1e40af);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    font-size: 1.1rem;
    padding: 0.75rem 2rem;
    box-shadow: 0 0 12px rgba(56,189,248,0.6);
    transition: all 0.3s ease;
}}

.stForm .stButton>button:hover {{
    background: linear-gradient(90deg, #1e3a8a, #60a5fa);
    transform: translateY(-3px);
    box-shadow: 0 0 20px rgba(96,165,250,0.8);
}}
.footer-badge-wrapper {{
    width: 100%;
    display: flex;
    justify-content: flex-end; /* align right */
    margin-top: 40px;
    margin-bottom: 20px;
}}

.footer-badge {{
    display: flex;
    align-items: center;
    background: #00aaff; /* neon-ish blue */
    padding: 10px 18px;
    border-radius: 30px; /* pill shape */
    box-shadow: 0 0 10px #00aaff; /* neon glow */
}}

.footer-badge img {{
    width: 42px;
    height: 42px;
    border-radius: 50%; /* circle */
    margin-right: 10px;
}}

.footer-badge-text {{
    font-size: 24px;
    font-weight: 900;
    color: black; /* text color */
    letter-spacing: 0px;
}}
/* Hover Animation */
.footer-badge:hover {{
    transform: scale(1.06);
    box-shadow: 0 0 16px #00cfff;
}}


</style>
""", unsafe_allow_html=True)

logo = Image.open("logo.png")
logo_base64 = image_to_base64("logo.png")  # ensure correct path
base_logo_base64 = image_to_base64("pfp.png")



# --- PAGE CONFIG ---
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# --- SESSION STATE CONTROL ---
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False



if "stage"  not in st.session_state:
    st.session_state.stage = 'Home Page'

if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False

# --- LANDING PAGE ---
if st.session_state.stage == 'Home Page':
    # HERO BANNER SECTION
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-overlay"></div>
        <div class="hero-text" style="
            display:flex; 
            align-items:center; 
            gap:0px;
        ">
            <img src="data:image/png;base64,{logo_base64}" 
                style="height:70px;">
            <h1 style="
                margin:0; 
                padding:0; 
                line-height:1;
                font-size:50px; 
                font-weight:700;
            ">
                Telco Customer Churn Prediction
            </h1>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Subheading
    st.markdown(
        "<p style='text-align:center; color:#cbd5e1; font-weight:bold;font-size:28px; max-width:1200px; margin:auto; '>Predict churn. Prevent loss. Power growth.</p>",
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("""
    <p style='text-align:left; color:#cbd5e1; font-weight:semi-bold;font-size:25px; max-width:2000px;'>⬢ This project illustrates the complete lifecycle of a machine learning workflow — from data cleaning and feature engineering to model evaluation and prediction on Telco-customer-churn Dataset.</p>
    <p style='text-align:left; color:#cbd5e1; font-weight:semi-bold;font-size:25px; max-width:2000px; '>⬢ The goal is to explore why customers leave and how predictive models can provide actionable insights in the telecom industry.</p>
    """, unsafe_allow_html=True)

    st.divider()

    # INFORMATION SECTION 
    st.markdown("""
    <p style='text-align:left; color:#cbd5e1; font-weight:bold;font-size:28px;'>📊 Why This Project Matters</p>
    <p style='text-align:left; color:#cbd5e1; font-size:22px; max-width:2000px; margin:auto;'>Telecom companies lose millions each year due to customer churn — customers leaving for competitors.</p>
    <p style='text-align:left; color:#cbd5e1; font-size:22px; max-width:2000px; margin:auto;'>This project uses predictive analytics to identify at-risk customers before they leave.</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:left; color:#cbd5e1; font-weight:bold;font-size:28px;'>🧠 How It Works</p>
    <p style='text-align:left; color:#cbd5e1; font-size:22px; max-width:2000px; margin:auto;'>The system uses machine learning algorithms such as Logistic Regression, Decision Trees, and SVMs to analyze customer data, contracts, and billing patterns to predict churn likelihood.</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:left; color:#cbd5e1; font-weight:bold;font-size:28px;'>🚀 Business Impact</p>
    <p style='text-align:left; color:#cbd5e1; font-size:22px; max-width:2000px; margin:auto;'>Early prediction enables proactive retention campaigns, saving revenue and strengthening customer loyalty.</p>
    """, unsafe_allow_html=True)

    st.divider()

    # Button triggers stage change
    _, center, _ = st.columns([1, 0.5, 1])
    with center:
        if st.button("✨ Explore the App"):
            st.session_state.show_sidebar = True
            st.session_state.stage = 'Show_navigation'
            st.rerun()


if st.session_state.stage == 'Show_navigation':
    with st.sidebar:
        c1, c2 = st.columns([1, 4])
        with c1:
            # Fills the small column width; no deprecated args
            st.image(logo, use_container_width=True)
        with c2:
            st.markdown(
                "<h3 style='margin:0; line-height:1.2;'>Telco Customer Churn Predictor</h3>",
                unsafe_allow_html=True
            )
    if st.session_state.show_sidebar:
        with st.sidebar:
            selected = option_menu(
                "Navigation",
                ["Home", "EDA Dashboard", "Prediction", "Model Comparison", "About"],
                icons=["house", "bar-chart", "cpu", "list-task", "info-circle"],
                menu_icon="cast",
                default_index=1,
            )

        if selected == "Home":
            st.session_state.stage = "Home Page"
            st.session_state.sidebar_open = False
            st.rerun()

        elif selected == "EDA Dashboard":
            page = st.sidebar.radio("Go to", [
                "Overview",
                "Customer Demographics",
                "Services & Usage",
                "Contract & Billing"
            ])

            df = pd.read_csv('cleaned_dataset.csv')

            # ----------------- Overview Page -----------------
            if page == "Overview":
                st.title("📊 Telco Customer Churn Dashboard")
               # Compute churn counts
                churn_counts = df['Churn'].value_counts().reset_index()
                churn_counts.columns = ['Churn', 'Count']
                churn_counts['Percent'] = (churn_counts['Count'] / churn_counts['Count'].sum()) * 100

                fig = px.bar(
                    churn_counts,
                    x='Churn',
                    y='Count',
                    text=churn_counts['Percent'].apply(lambda x: f"{x:.1f}%"),
                    color='Churn',
                    color_discrete_map={'Yes': '#E76F51', 'No': '#2A9D8F'},
                    title="<b>Distribution of Customer Churn</b>"
                )

                fig.update_layout(title_x=0.38, plot_bgcolor='white')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                # KPI Metric
                churn_rate = (df['Churn'] == 'Yes').mean() * 100
                st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

                # ---- FIXED: Churn Rate by Contract ----
                churn_rate_by_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
                churn_rate_by_contract.columns = ['Contract', 'Churn_rate']

                fig3 = px.bar(
                    churn_rate_by_contract,
                    x="Churn_rate",
                    y="Contract",
                    color="Churn_rate",
                    text="Churn_rate",
                    title="<b>Churn Rate by Contract Type</b>"
                )
                fig3.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                fig3.update_layout(title_x=0.38)
                st.plotly_chart(fig3, use_container_width=True)

                # ---- Arrange KDE Plots Side-by-Side ----
                col1, col2 = st.columns(2)
                sns.set_theme(style="whitegrid")

                with col1:
                    fig1 = plt.figure(figsize=(6,3))
                    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False,
                                palette={'Yes': '#E76F51', 'No': '#2A9D8F'}, alpha=0.5)
                    plt.title('Monthly Charges by Churn')
                    plt.tight_layout()
                    st.pyplot(fig1)

                with col2:
                    fig2 = plt.figure(figsize=(6,3))
                    sns.kdeplot(data=df, x='TotalCharges', hue='Churn', fill=True, common_norm=False,
                                palette={'Yes': '#E76F51', 'No': '#2A9D8F'}, alpha=0.5)
                    plt.title('Total Charges by Churn')
                    plt.tight_layout()
                    st.pyplot(fig2)

                st.markdown("---")   # A clean horizontal divider
                st.subheader("⬢ Insights from Customer Churn Distribution")
                st.markdown("""
                - Around **26.5%** of customers have churned, while **73.5%** have remained — reflecting a clear **class imbalance**.
                - This imbalance means predictive models should prioritize **recall** to detect churners effectively.
                - Consider using **class weighting** or **oversampling** methods (e.g., SMOTE) during model training.
                - Even a relatively small churn percentage can result in **large revenue losses**, making churn prediction critical.
                """)

                st.subheader("⬢ Churn Rate by Contract Type")
                st.markdown("""
                | Contract Type | Churn Rate | Insight |
                |--------------|------------|---------|
                | **Month-to-Month** | **Highest (≈43%)** | Flexible → Easier to leave |
                | **One Year** | **Moderate (≈11%)** | Medium-term → Some retention stability |
                | **Two Year** | **Lowest (≈3%)** | Long-term → Strong retention |

                **Conclusion:** Encouraging customers to upgrade from month-to-month to **longer-term contracts** can significantly reduce churn.
                """)

                st.subheader("⬢ Insights: Monthly Charges by Churn")
                st.markdown("""
                - Churned customers often have **higher monthly charges ($70–100)**.
                - Non-churned users cluster more around **20–40** Dollars   with an additional moderate tier at **60–80** Dollars.
                - **Higher cost → Higher churn probability**, likely due to cost sensitivity.

                **Recommendation:**  
                Consider **loyalty discounts**, **bundling offers**, or **custom pricing tiers** to retain high-paying customers.
                """)

                st.subheader("⬢ Insights: Total Charges by Churn (Customer Tenure Effect)")
                st.markdown("""
                - Churned customers typically show **low total charges (< $2,000)**.
                - Non-churned customers span **much higher total values**, suggesting **longer retention**.
                - Low total charges = **Newer customers who churn early**.

                **Recommendation:**  
                Improve **onboarding**, **early engagement**, and provide **first 90-day retention incentives** to prevent early churn.
                """)



            # ----------------- Customer Demographics -----------------
            elif page == "Customer Demographics":
                st.title("👥 Customer Demographics & Churn")
                # ------- Gender & Churn Pie Charts -------
                gender_counts = df['gender'].value_counts()
                churn_counts = df['Churn'].value_counts()

                labels1 = gender_counts.index.tolist()
                values1 = gender_counts.values.tolist()

                labels2 = churn_counts.index.tolist()
                values2 = churn_counts.values.tolist()

                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{'type': 'domain'}, {'type': 'domain'}]],
                    subplot_titles=['Gender Distribution', 'Churn Distribution']
                )

                fig.add_trace(go.Pie(
                    labels=labels1,
                    values=values1,
                    marker_colors=["#5196F7", "#F3ABEC"],
                    hole=0.4,
                    textinfo='percent+label',
                    textposition='inside'
                ), row=1, col=1)

                fig.add_trace(go.Pie(
                    labels=labels2,
                    values=values2,
                    marker_colors=["#6DF83A", "#1EE4F2"],
                    hole=0.4,
                    textinfo='percent+label',
                    textposition='inside'
                ), row=1, col=2)

                fig.update_layout(
                    title="<b>Gender & Churn Distribution</b>",
                    title_x=0.35
                )

                st.plotly_chart(fig, use_container_width=True)


                # ------- Three Histograms Arranged in Columns -------

                col1, col2 = st.columns(2)

                with col1:
                    fig_senior = px.histogram(
                        df, x='Churn', color='SeniorCitizen', histnorm='percent',
                        barmode='group', text_auto=".1f",
                        title="<b>Senior Citizens by Churn</b>",
                        color_discrete_map={'Yes': "#605454", 'No': "#86B6C4"}
                    )
                    fig_senior.update_layout(title_x=0.3, bargap=0.2, showlegend=True)
                    fig_senior.update_traces(textposition='outside')
                    st.plotly_chart(fig_senior, use_container_width=True)

                with col2:
                    fig_partner = px.histogram(
                        df, x='Churn', color='Partner', histnorm='percent',
                        barmode='group', text_auto=".1f",
                        title="<b>Partner Type by Churn</b>",
                        color_discrete_map={'Yes': "#00B5AD", 'No': "#FF6B6B"}
                    )
                    fig_partner.update_layout(title_x=0.3, bargap=0.2, showlegend=True)
                    fig_partner.update_traces(textposition='outside')
                    st.plotly_chart(fig_partner, use_container_width=True)

                st.markdown("---")

                # Full width histogram
                fig_dependents = px.histogram(
                    df, x='Churn', color='Dependents', histnorm='percent',
                    barmode='group', text_auto=".1f",
                    title="<b>Dependents by Churn</b>",
                    color_discrete_map={'Yes': "#E76F51", 'No': "#2A9D8F"}
                )
                fig_dependents.update_layout(title_x=0.5, bargap=0.2, showlegend=True)
                fig_dependents.update_traces(textposition='outside')
                st.plotly_chart(fig_dependents, use_container_width=True)

                st.markdown("---")
                st.subheader("⬢ Insights from Gender and Churn Distribution")
                st.markdown("""
                - The **gender distribution** is balanced, with nearly equal proportions of male and female customers.
                - The **overall churn rate** is approximately **26.5%**, meaning about **one-fourth** of customers have left the service.
                - Both genders exhibit **similar churn behavior**, suggesting that **gender is not a meaningful predictor** of churn in this case.
                """)

                st.subheader("⬢ Insights: Senior Citizens and Churn")
                st.markdown("""
                - **Senior citizens** show a **significantly higher churn rate (≈41.7%)** compared to **non-seniors (≈23.6%)**.
                - This may be due to **affordability concerns, service clarity, or support needs**.
                
                **Key Takeaway:**  
                Provide **senior-friendly plans** and **personalized assistance** to better retain senior customers.
                """)

                st.subheader("⬢ Insights: Partner Status and Churn")
                st.markdown("""
                - Customers **with partners** churn at a **lower rate (≈19.7%)** than customers **without partners (≈33.0%)**.
                - This suggests that customers with partners may have **more stable lifestyles or shared service decisions**, reducing the likelihood of churn.
                
                **Key Takeaway:**  
                Focus **retention strategies** on customers **without partners**, as they represent a **higher churn risk group**.
                """)

                st.subheader("⬢ Insights: Dependents and Churn")
                st.markdown("""
                - Customers **without dependents** show a **much higher churn rate (≈84.5%)**.
                - Customers **with dependents** churn far less (≈15.5%), suggesting greater service stability and commitment.

                **Key Takeaway:**  
                Prioritize **customers without dependents** with retention efforts such as:
                - Loyalty rewards
                - Value-based messaging
                - Personalized plan recommendations
                """)


                


            # ----------------- Services & Usage -----------------
            elif page == "Services & Usage":
                st.title("📡 Services Subscribed & Usage Behavior")
                # 1) Customer Churn by Internet Service & Gender (Seaborn)
                # -------------------------------------------
                sns.set_theme(style="whitegrid")

                fig_cat = sns.catplot(
                    data=df,
                    x="InternetService",
                    hue="gender",
                    col="Churn",
                    kind="count",
                    palette={"Female": "#F3A4D7", "Male": "#00F7FF"},
                    height=4,
                    aspect=1
                )

                fig_cat.fig.suptitle("Customer Churn by Internet Service and Gender", fontsize=14, fontweight="bold")
                fig_cat.set_axis_labels("Internet Service Type", "Count of Customers")
                fig_cat.fig.subplots_adjust(top=0.85)

                st.pyplot(fig_cat)


                st.markdown("---")  # divider


                # -------------------------------------------
                # 2) Histograms (Displayed Side-by-Side)
                # -------------------------------------------
                col1, col2 = st.columns(2)

                # Phone Service by Churn
                with col1:
                    fig_phone = px.histogram(
                        df, x='Churn', color="PhoneService", histnorm="percent",
                        barmode='group', text_auto='.1f',
                        title="<b>Phone Service by Churn</b>",
                        color_discrete_map={'Yes': '#264653', 'No': '#E9C46A'}
                    )
                    fig_phone.update_layout(title_x=0.3, bargap=0.2)
                    fig_phone.update_traces(textposition='outside')
                    st.plotly_chart(fig_phone, use_container_width=True)

                # Online Security by Churn
                with col2:
                    fig_security = px.histogram(
                        df, x='Churn', color="OnlineSecurity", histnorm="percent",
                        barmode='group', text_auto='.1f',
                        title="<b>Online Security by Churn</b>"
                    )
                    fig_security.update_layout(title_x=0.3, bargap=0.2)
                    fig_security.update_traces(textposition='outside')
                    st.plotly_chart(fig_security, use_container_width=True)

                st.markdown("---")


                # -------------------------------------------
                # 3) Multiple Lines by Churn (Full Width)
                # -------------------------------------------
                fig_mul = px.histogram(
                    df, x='Churn', color="MultipleLines", histnorm="percent",
                    barmode='group', text_auto='.1f',
                    title="<b>Multiple Lines by Churn</b>"
                )
                fig_mul.update_layout(title_x=0.4, bargap=0.2)
                fig_mul.update_traces(textposition='outside')
                st.plotly_chart(fig_mul, use_container_width=True)

                st.markdown("---")
                st.subheader("⬢ Insights on Customer Churn by Internet Service and Gender")
                st.markdown("""
                - **DSL users** show **lower churn rates** across both genders, indicating stable satisfaction.
                - **Fiber optic users** exhibit a **notably higher churn rate**, suggesting dissatisfaction related to **pricing, performance, or alternatives**.
                - Customers **without internet service** show **very low churn**, likely due to minimal digital engagement.
                - **Gender comparison:** Churn behaviors are **consistent between males and females**, meaning **gender is not a meaningful churn driver**.

                **Key Insight:**  
                The **type of internet service** impacts churn more than gender.  
                Focus retention efforts on **fiber optic subscribers** through **pricing improvements, service upgrades, or loyalty benefits**.
                """)

                st.markdown("---")
                st.subheader("⬢ Insights: Phone Service and Churn")
                st.markdown("""
                - Customers **with phone service** have a **slightly higher churn rate** compared to those without it.
                - However, the difference is **small**, meaning phone service availability **does not strongly influence churn**.
                - Churn is likely driven by **other service experiences** (internet performance, billing transparency, customer support).

                **Key Takeaway:**  
                Phone service alone is **not a major churn factor**. Improve retention through **service quality** and **support experience** rather than phone offerings.
                """)

                st.markdown("---")
                st.subheader("⬢ Insights: Online Security and Churn")
                st.markdown("""
                - Customers **without online security** have the **highest churn rate (≈41.8%)**.
                - Those **with online security** show **much lower churn (≈14.6%)**, suggesting it increases **trust and perceived value**.
                - Customers **without internet** show very **low churn (~7.4%)**, as they engage less with online services.

                **Key Takeaway:**  
                Promote **online security add-ons** — they are strongly linked to **higher customer retention and satisfaction**.
                """)

                st.markdown("---")
                st.subheader("⬢ Insights: Multiple Lines and Churn")
                st.markdown("""
                - Customers with **multiple lines** show a **slightly higher churn rate (≈28.6%)**.
                - This suggests they may have **higher expectations**, and churn could be triggered by **pricing concerns or lack of bundled value**.

                **Key Takeaway:**  
                Offer **multi-line discounts, bundled packages, or family plans** to retain this segment and reduce churn.
                """)


                


            # ----------------- Contract & Billing -----------------
            elif page == "Contract & Billing":
                st.title("💳 Contract & Billing Insights")

                # Histogram: Contract vs Churn
                fig1 = px.histogram(
                    df, x='Churn', color='Contract',
                    barmode='group', histnorm='percent',
                    title="<b>Distribution of Contract Types by Churn</b>"
                )

                fig1.update_layout(
                    width=800, height=550,
                    title_x=0.30,
                    bargap=0.20,
                    xaxis_title='<b>Churn</b>',
                    yaxis_title='<b>Percent %</b>',
                    legend=dict(
                        orientation='h',
                        yanchor='bottom', xanchor='right',
                        y=1.02, x=1
                    )
                )

                fig1.update_traces(
                    texttemplate='%{y:.1f}%',
                    textposition='outside'
                )

                st.plotly_chart(fig1, use_container_width=True)


                # Pie Chart: Payment Methods
                payment_methods = df['PaymentMethod'].value_counts()
                labels = list(payment_methods.index)
                values = list(payment_methods.values)

                fig2 = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textinfo="percent",
                    textposition="inside",
                    insidetextorientation="horizontal",
                    textfont=dict(size=18),
                    marker_colors=["#CCF780", "#08F7E7", "#2EFB2E", "#67D2A0"]
                )])

                fig2.update_layout(
                    title_text='<b>Distribution Of Payment Methods</b>',
                    title_x=0.26
                )

                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("""
                ### ⬢ **Insights from Distribution of Contract Types by Churn**

                - Customers with **month-to-month contracts** have the **highest churn rate (42.7%)**, indicating they are more likely to leave the service.
                - **One-year (11.3%)** and **two-year (2.8%)** contract customers show **much lower churn**, suggesting **long-term contracts encourage retention**.
                - Among customers who did **not churn**, **97.2% of two-year** and **88.7% of one-year** contract holders stayed, reinforcing that **longer commitments reduce churn risk**.
                - **Overall Insight:** Promoting **long-term plans** through loyalty offers or discounts can significantly improve customer retention.

                ---

                ### ⬢ **Insights on Distribution of Payment Methods**

                - **Electronic Check (33.6%)** is the most commonly used method, showing a preference for **fast, paperless transactions**.
                - **Mailed Check (21.6%)**, **Bank Transfer (21.9%)**, and **Credit Card (22.9%)** are used at similar rates, reflecting **diverse customer preferences**.
                - **Key Insight:** While electronic payments lead, maintaining **multiple payment options** supports broader customer convenience.
                - **Potential Action:** Analyze churn by payment type — customers using **manual methods** (like mailed checks) may be more likely to churn due to **less automation and lower engagement**.
                """)


        elif selected == "Prediction":

            st.sidebar.title("Choose Prediction Mode")
            mode = st.sidebar.radio(
                "Select Option:",
                ["🔹 Quick Prediction (On Best Model)",
                "🔸 Compare Top 3 Models on Test Data"]
            )
            @st.cache_resource
            def load_models():
                model_log = joblib.load("final_log_reg.pkl")
                model_svc = joblib.load("final_svc.pkl")
                model_dt = joblib.load("final_dt.pkl")
                return model_log, model_svc, model_dt

            model_log, model_svc, model_dt = load_models()

            if mode == "🔹 Quick Prediction (On Best Model)":
                st.title(" Quick Prediction (Best Model: Logistic Regression)")
                st.write("Fill in customer details and get churn prediction instantly.")

                # ---- INPUT FORM ----
                with st.form("prediction_form"):

                    st.subheader("Customer Profile")
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
                    partner = st.selectbox("Partner", ["No", "Yes"])
                    dependents = st.selectbox("Dependents", ["No", "Yes"])
                    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)

                    st.subheader("Phone & Internet Services")
                    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
                    multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                    onlinesecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                    onlinebackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                    deviceprotection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                    techsupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                    streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                    streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

                    st.subheader("Billing Information")
                    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
                    paymentmethod = st.selectbox("Payment Method", [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ])
                    monthlycharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
                    totalcharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

                    submitted = st.form_submit_button("🔍 Predict Churn")


                # ---- IF BUTTON CLICKED ----
                if submitted:

                    input_df = pd.DataFrame([{
                        'gender': gender,
                        'SeniorCitizen': senior,
                        'Partner': partner,
                        'Dependents': dependents,
                        'tenure': tenure,
                        'PhoneService': phoneservice,
                        'MultipleLines': multiplelines,
                        'InternetService': internet,
                        'OnlineSecurity': onlinesecurity,
                        'OnlineBackup': onlinebackup,
                        'DeviceProtection': deviceprotection,
                        'TechSupport': techsupport,
                        'StreamingTV': streamingtv,
                        'StreamingMovies': streamingmovies,
                        'Contract': contract,
                        'PaperlessBilling': paperless,
                        'PaymentMethod': paymentmethod,
                        'MonthlyCharges': monthlycharges,
                        'TotalCharges': totalcharges
                    }])


                    # Predict
                    prediction = model_log.predict(input_df)[0]
                    probability = model_log.predict_proba(input_df)[0][1]

                    # ---- RESULT DISPLAY ----
                    st.subheader("Prediction Result:")

                    if prediction == 1:
                        st.warning(f"❗ **High Churn Risk** — Probability: **{probability:.2%}**")
                        st.write("Customer is likely to leave. Consider offering retention benefits or long-term plan discounts.")
                    else:
                        st.success(f"✅ **Low Churn Risk** — Probability: **{probability:.2%}**")
                        st.write("Customer is likely to stay. Maintain engagement and satisfaction strategies.")

            elif mode == "🔸 Compare Top 3 Models on Test Data":
                            # Train + Evaluate Logistic Regression, SVC, Decision Tree
                            st.title("Model Evaluation on Test Set")
                            X_train = pd.read_csv('X_train.csv')
                            X_test = pd.read_csv('X_test.csv')
                            y_test = pd.read_csv('y_test.csv')

                            if st.button('Split the Training & Test Data'):
                                with st.spinner('Splitting the Data into X_train & X_test...'):
                                    time.sleep(1) 

                                st.subheader('X_train Data:')
                                st.write(X_train)

                                st.subheader('X_test Data:')
                                st.write(X_test)

                            if st.button('Train Logistic Regression'):
                                with st.spinner('Training the Model...'):
                                    time.sleep(2)

                                    # Ensure y_test is a 1D vector, not a DataFrame
                                    y_test = y_test['Churn']

                                    y_pred = model_log.predict(X_test)

                                    # Compute metrics — positive class = 1 (Churn)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision = precision_score(y_test, y_pred)
                                    recall = recall_score(y_test, y_pred)
                                    f1 = f1_score(y_test, y_pred)
                                    classification = classification_report(y_test, y_pred)

                                    # Display Metrics
                                    st.subheader("Model Performance (Test Set)")
                                    st.write(f"**Accuracy:** {accuracy:.3f}")
                                    st.write(f"**Precision:** {precision:.3f}")
                                    st.write(f"**Recall:** {recall:.3f}")
                                    st.write(f"**F1 Score:** {f1:.3f}")
                                    st.write("**Classification Report:**")
                                    st.text(classification)

                                    # Confusion Matrix
                                    cm_data = confusion_matrix(y_test, y_pred)

                                    fig = plt.figure(figsize=(3,2))
                                    sns.heatmap(cm_data, annot=True, fmt="d", cmap="coolwarm", cbar=False,
                                                xticklabels=["No Churn (0)", "Churn (1)"],
                                                yticklabels=["No Churn (0)", "Churn (1)"])
                                    plt.xlabel("Predicted")
                                    plt.ylabel("Actual")
                                    plt.title("Confusion Matrix - Logistic Regression")
                                    st.pyplot(fig)
                            if st.button('Train SVC'):
                                with st.spinner('Training the Model...'):
                                    time.sleep(2)

                                    # Ensure y_test is a 1D vector, not a DataFrame
                                    y_test = y_test['Churn']

                                    y_pred = model_svc.predict(X_test)

                                    # Compute metrics — positive class = 1 (Churn)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision = precision_score(y_test, y_pred)
                                    recall = recall_score(y_test, y_pred)
                                    f1 = f1_score(y_test, y_pred)
                                    classification = classification_report(y_test, y_pred)

                                    # Display Metrics
                                    st.subheader("Model Performance (Test Set)")
                                    st.write(f"**Accuracy:** {accuracy:.3f}")
                                    st.write(f"**Precision:** {precision:.3f}")
                                    st.write(f"**Recall:** {recall:.3f}")
                                    st.write(f"**F1 Score:** {f1:.3f}")
                                    st.write("**Classification Report:**")
                                    st.text(classification)

                                    # Confusion Matrix
                                    cm_data = confusion_matrix(y_test, y_pred)

                                    fig = plt.figure(figsize=(3,2))
                                    sns.heatmap(cm_data, annot=True, fmt="d", cmap="coolwarm", cbar=False,
                                                xticklabels=["No Churn (0)", "Churn (1)"],
                                                yticklabels=["No Churn (0)", "Churn (1)"])
                                    plt.xlabel("Predicted")
                                    plt.ylabel("Actual")
                                    plt.title("Confusion Matrix - SVC")
                                    st.pyplot(fig)

                            if st.button('Train Decision Tree'):
                                with st.spinner('Training the Model...'):
                                    time.sleep(2)

                                    # Ensure y_test is a 1D vector, not a DataFrame
                                    y_test = y_test['Churn']

                                    y_pred = model_dt.predict(X_test)

                                    # Compute metrics — positive class = 1 (Churn)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    precision = precision_score(y_test, y_pred)
                                    recall = recall_score(y_test, y_pred)
                                    f1 = f1_score(y_test, y_pred)
                                    classification = classification_report(y_test, y_pred)

                                    # Display Metrics
                                    st.subheader("Model Performance (Test Set)")
                                    st.write(f"**Accuracy:** {accuracy:.3f}")
                                    st.write(f"**Precision:** {precision:.3f}")
                                    st.write(f"**Recall:** {recall:.3f}")
                                    st.write(f"**F1 Score:** {f1:.3f}")
                                    st.write("**Classification Report:**")
                                    st.text(classification)

                                    # Confusion Matrix
                                    cm_data = confusion_matrix(y_test, y_pred)

                                    fig = plt.figure(figsize=(3,2))
                                    sns.heatmap(cm_data, annot=True, fmt="d", cmap="coolwarm", cbar=False,
                                                xticklabels=["No Churn (0)", "Churn (1)"],
                                                yticklabels=["No Churn (0)", "Churn (1)"])
                                    plt.xlabel("Predicted")
                                    plt.ylabel("Actual")
                                    plt.title("Confusion Matrix - Decision Tree")
                                    st.pyplot(fig)                    
                           
        elif selected == "Model Comparison":
            st.title("⚖️ Model Comparison")
            metrics_results = pd.read_csv('metrics_results.csv')
            st.write(metrics_results)

            df_melted = metrics_results.melt(id_vars='Model', 
                            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                            var_name='Metric', 
                            value_name='Score')

            # Create grouped bar chart
            fig = px.bar(
                df_melted,
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title="<b>Model Performance Comparison</b>",
                text='Score'
            )

            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')

            fig.update_layout(
                title_x=0.36,
                xaxis_title='<b>Model</b>',
                yaxis_title='<b>Score</b>',
                width=950,
                height=520,
                bargap=0.25,
                legend_title_text='<b>Metric</b>'
            )

            st.plotly_chart(fig, use_container_width=True)

        elif selected == "About":
            st.title("ℹ️ About")
            st.markdown("""
                ##  ⬢ **About This App — Customer Churn Prediction**

                This application demonstrates a predictive approach for identifying customers who may discontinue their telecom services (churn).
                The model analyzes patterns in customer data to estimate churn likelihood, allowing users to explore how such insights could support proactive engagement strategies like personalized offers or improved support.

                ---

                ### 🎯 Project Objective
                The goal of this project is to:
                - Analyze customer behavior and service usage patterns
                - Detect key factors that contribute to churn
                - Build a Machine Learning model that **predicts churn probability**
                - Provide **actionable insights** to improve customer retention

                ---

                ### 🔍 What This App Can Do
                - **Make Predictions**  
                Enter customer details manually or test the model on the full dataset.
                - **Visualize Patterns**  
                View distributions and feature relationships that influence churn.
                - **Compare Models**  
                The best-performing model (Logistic Regression) was selected after evaluating multiple algorithms such as:
                - Logistic Regression  
                - Random Forest  
                - SVM  
                - KNN  
                - AdaBoost  
                - XGBoost  
                - Voting Ensemble  

                ---

                ### 🧠 Model Performance By Hyperparameter Tuning (Top Models)
                | Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
                |---------------------|----------|-----------|--------|----------|---------|
                | Logistic Regression | **0.7235** | **0.4871** | **0.8094** | **0.6082** | **0.7509** |
                | SVC                 | **0.6882** | **0.4521** | **0.8287** | **0.5850** | **0.7331** |
                | Decision Tree       | **0.7286** | **0.4926** | **0.7880** | **0.6063** | **0.7476** |
                
            

                The **Gradient Boosting Classifier** performed best in balancing both correct identification of churn customers and minimizing false predictions.

                ---

                ### 📂 Dataset Information
                - Source: *Telco Customer Churn Dataset*
                - Rows: ~7,000 customers
                - Features include:
                - Demographics (Gender, Senior Citizen, Dependents)
                - Contract & Billing details
                - Internet & phone service usage patterns
                - Monthly & Total charges

                ---

                ### 👨‍💻 Built With
                - Python
                - Plotly
                - Scikit-learn
                - Pandas & NumPy
                - Streamlit (for interactive app)
                - Matplotlib & Seaborn (for visualizations)

                ---

                ### 💡 Why This Matters
                Customer retention is **far cheaper** than customer acquisition.  
                This model helps businesses:
                - Detect at-risk customers
                - Understand why churn happens
                - Plan retention solutions proactively

                ---

                *Thank you for exploring the app! Feel free to test different scenarios and see how customer behaviors influence churn probability.* 🚀
            """)

target_url = "https://github.com/Om-codex/Telco-Customer-Churn-Prediction"
st.markdown(
    f"""
    <div class="footer-badge-wrapper">
        <a href="{target_url}" target="_blank" class="footer-badge-link">
            <div class="footer-badge">
                <img src="data:image/png;base64,{base_logo_base64}">
                <span class="footer-badge-text">Om-Codex</span>
            </div>
        </a>
    </div>
    """,
    unsafe_allow_html=True

)
