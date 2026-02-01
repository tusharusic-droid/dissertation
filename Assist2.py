import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import google.generativeai as genai
import os

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="RAN Intelligence AI", layout="wide")

from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()  # Load environment variables from .env
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. DATA LOADING & PREPARATION ---
@st.cache_data
def load_kpi_data(file):
    df = pd.read_csv(file)
    # Prepare Data: Clean trailing spaces and fix Date format
    df.columns = df.columns.str.strip()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True).dt.date
    return df

@st.cache_data
def load_descriptions():
    # Placeholder for your dictionary loading logic
    counter_data = pd.DataFrame({"Counter": ["RRC Setup", "DL BLER"], "Description": ["Radio Resource Control", "Block Error Rate"]})
    param_data = pd.DataFrame({"Parameter": ["Max_Users"], "Description": ["Max cell capacity"]})
    return counter_data, param_data

# --- 3. ML & RULE ENGINE MODULES ---
def run_isolation_forest(df, kpi_column, contamination=0.05):
    """Initializes and runs Isolation Forest for anomaly detection."""
    data = df[[kpi_column]].dropna()
    if data.empty: return df
    
    model = IsolationForest(contamination=contamination, random_state=42)
    data['is_anomaly'] = model.fit_predict(data[[kpi_column]])
    data['is_anomaly'] = data['is_anomaly'].apply(lambda x: True if x == -1 else False)
    
    return df.merge(data[['is_anomaly']], left_index=True, right_index=True, how='left')

def get_gemini_response(prompt):
    """LLM Helper Function to interact with Gemini."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {str(e)}"

#--- LTE Optimization Rule 
def load_optimization_knowledge():
    """
    Loads optimization rules from text files into a combined text context for Gemini.
    """
    rule_files = {
        "Accessibility": "LTE Acc_Rule.txt",
        "Integrity": "LTE Intg_Rule.txt",
        "Mobility": "LTE Mob_Rule.txt",
        "Retainability": "LTE Ret_Rule.txt"
    }
    
    knowledge_context = "### RAN Optimization Knowledge Base ###\n"
    
    for category, file_name in rule_files.items():
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as f:
                    content = f.read()
                    knowledge_context += f"\n--- {category} Rules ---\n{content}\n"
            except Exception as e:
                knowledge_context += f"\nError loading {category} rules: {str(e)}\n"
        else:
            knowledge_context += f"\nWarning: {file_name} not found.\n"
            
    return knowledge_context

def get_recommendations_from_rules(category_query):
    """
    Robustly parses combined JSON rule strings to extract actions.
    """
    all_actions = []
    # 1. Normalize query
    query = category_query.strip().lower()
    
    # 2. Get the full rules text
    raw_text = load_optimization_knowledge()
    
    # 3. Use JSON parsing if possible, or robust string splitting
    # Example logic to find the category block and extract actions
    try:
        # Splitting by rules to find the right block
        rule_blocks = raw_text.split('"id":')
        for block in rule_blocks:
            if f'"category": "{category_query}"' in block or f'"category":"{category_query}"' in block:
                # Extract the actions list from the string
                if '"actions": [' in block:
                    action_part = block.split('"actions": [')[1].split(']')[0]
                    # Clean up and add to list
                    actions = [a.strip().strip('"') for a in action_part.split(',')]
                    all_actions.extend(actions)
    except Exception as e:
        print(f"Error parsing rules: {e}")
        
    return list(set(all_actions)) # Return unique actions

# --- 4. MAIN APPLICATION ---
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # 1. Data Source Section
        with st.expander("📂 Data Source Manager", expanded=True):
            source = st.selectbox("Select Source:", ["Default Sample", "Upload CSV"])
            kpi_data = None
            if source == "Default Sample":
                if os.path.exists("KPI Sample.csv"):
                    kpi_data = load_kpi_data("KPI Sample.csv")
                else: st.error("Sample CSV not found.")
            else:
                uploaded = st.file_uploader("Upload CSV", type="csv")
                if uploaded: kpi_data = load_kpi_data(uploaded)

            if kpi_data is None:
                st.info("Awaiting data input...")
                st.stop()
            else:
                st.success("Data Loaded Successfully!")

        # 2. Dictionary Search (Collapsed)
        with st.expander("🔍 KPI Dictionary", expanded=False):
            c_desc, p_desc = load_descriptions()
            search = st.text_input("Search Name:")
            if search:
                res = c_desc[c_desc['Counter'].str.contains(search, case=False)]
                if not res.empty: st.info(res.iloc[0]['Description'])

    # --- LAYOUT: VISUALIZATION AND CHAT ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("📡 RAN Performance Dashboard")
        tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🤖 ML Anomalies", "📏 Rule Engine"])

        with tab1:
            st.subheader("Selection for Visualization Type")
            viz_type = st.radio("Choose View:", ["Trend", "Heatmap", "Scatter"], horizontal=True)
            num_cols = kpi_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

            if viz_type == "Trend":
                target = st.selectbox("Select KPI:", num_cols)
                fig, ax = plt.subplots()
                sns.lineplot(data=kpi_data, x='Date', y=target, ax=ax, marker='o')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            elif viz_type == "Heatmap":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(kpi_data[num_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            elif viz_type == "Scatter":
                x_ax = st.selectbox("X-Axis:", num_cols, index=0)
                y_ax = st.selectbox("Y-Axis:", num_cols, index=1)
                fig, ax = plt.subplots()
                sns.scatterplot(data=kpi_data, x=x_ax, y=y_ax, ax=ax)
                st.pyplot(fig)

        with tab2:
            st.subheader("Isolation Forest Analysis")
            ml_target = st.selectbox("Target KPI for ML:", num_cols, key="ml_v")
            contam = st.slider("Sensitivity:", 0.01, 0.20, 0.05)
            if st.button("Detect Outliers"):
                results = run_isolation_forest(kpi_data, ml_target, contam)
                anoms = results[results['is_anomaly'] == True]
                fig, ax = plt.subplots()
                sns.lineplot(data=results, x='Date', y=ml_target, ax=ax)
                sns.scatterplot(data=anoms, x='Date', y=ml_target, color='red', s=100, ax=ax)
                st.pyplot(fig)
                st.warning(f"Detected {len(anoms)} anomalies.")

        with tab3:
            st.subheader("📏 RAN Intelligence Rule Engine")
            
            # 1. Date Selection & Data Filtering
            diag_date = st.selectbox("Select Date for Deep Diagnosis:", kpi_data['Date'].unique())
            day_df = kpi_data[kpi_data['Date'] == diag_date]
            
            # 2. Threshold Analysis Logic
            # These match the core categories in your LTE_Acc, Intg, Mob, and Ret files
            alerts = []
            if day_df['DL PRB Utilization'].mean() > 80: 
                alerts.append({"issue": "High Congestion", "category": "Capacity"})
            if day_df['DL BLER'].mean() > 10: 
                alerts.append({"issue": "High Interference", "category": "Radio"})
            if day_df['PS Drop Call Rate'].mean() > 0.5: 
                alerts.append({"issue": "High Drop Rate", "category": "Retainability"})

            # 3. Rule Matching Logic
            if alerts:
                st.write(f"### 🚨 Detected Issues for {diag_date}")
                
                # We will collect detailed actions to send to the AI
                detailed_ai_context = []

                for alert in alerts:
                    with st.expander(f"Analysis: {alert['issue']}", expanded=True):
                        st.error(f"**Issue:** {alert['issue']}")
                        
                        # Use the helper function to get specific recommendations from your .txt files
                        # This looks for matching 'category' in your JSON rules
                        recommendations = get_recommendations_from_rules(alert['category'])
                        
                        if recommendations:
                            st.markdown("**Recommended Actions (from Rule Engine):**")
                            for action in recommendations:
                                st.write(f"- {action}")
                            detailed_ai_context.append(f"{alert['issue']}: {', '.join(recommendations)}")
                        else:
                            st.info("No specific hardware rule match found; refer to general optimization.")

                # 4. Bridge to AI Assistant
                if st.button("Send Technical Analysis to Gemini"):
                    st.session_state.diag_context = (
                        f"On {diag_date}, the system detected: " + 
                        " | ".join(detailed_ai_context)
                    )
                    st.success("Analysis sent! Check the AI Assistant in the right column for a detailed plan.")
            
            else:
                st.success(f"✅ All network rules passed for {diag_date}. Performance is within nominal limits.")
    with col2:
        st.header("🤖 AI Assistant")
        
        # --- 1. Load Rule Engine Knowledge ---
        rule_context = load_optimization_knowledge()

        # --- 2. Initialize Chat Session State ---
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant", 
                "content": "Hello! I am your RAN Optimization Assistant. I have loaded your rule engines and I am ready to help diagnose network issues."
            }]

        # --- 3. Display Chat History ---
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # --- 4. Handle User Input ---
        if prompt := st.chat_input("Ask about network performance..."):
            # Append user message to state and display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # --- 5. Build Context-Aware Prompt ---
            # We combine the user's question with the rule engine data
            # and any diagnostic findings from the Rule Engine tab.
            full_prompt = f"""
            System Context (Rule Engines):
            {rule_context}
            
            Diagnostic Context:
            {st.session_state.get('diag_context', 'No specific diagnostics sent yet.')}
            
            User Question: {prompt}
            """

            # --- 6. Generate AI Response ---
            with st.chat_message("assistant"):
                with st.spinner("Analyzing rules and data..."):
                    # Call your LLM helper function
                    response = get_gemini_response(full_prompt)
                    st.markdown(response)
            
            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Clear diagnostic context after use to keep the next query clean
            if "diag_context" in st.session_state:
                del st.session_state.diag_context
if __name__ == "__main__":
    main()