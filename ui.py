import streamlit as st
import pandas as pd
import re
from datetime import datetime
from orchestrator import SynergyPipeline
from auth import login_user

st.set_page_config(page_title="Synergy | Multi-User Banking Guard", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4253; }
    .risk-high, .risk-critical { color: #ff4b4b; font-weight: bold; border-left: 5px solid #ff4b4b; padding-left: 10px; }
    .risk-low { color: #00ff00; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    pipeline = SynergyPipeline()
    try: pipeline.load()
    except: pipeline.train()
    
    import shared_data
    shared_data.RISK_WEIGHTS['rule_based'] = 0.55  
    return pipeline

pipeline = load_pipeline()

# --- MULTI-USER STATE MANAGEMENT ---
if 'users' not in st.session_state:
    st.session_state.users = {
        "Sashwat": {"balance": 50000.0, "acc": "9876-1234", "history": []},
        "Sarthak": {"balance": 30000.0, "acc": "7977-2082", "history": []},
        "Atharva": {"balance": 28000.0, "acc": "1122-3344", "history": []},
        "Modussir": {"balance": 25000.0, "acc": "5566-7788", "history": []}
    }


if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'active_user' not in st.session_state:
    st.session_state.active_user = None

# ---------------- AUTH LAYER ----------------
if not st.session_state.authenticated:
    st.title("🔐 Login Required")

    username = st.text_input("Enter Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state.authenticated = True
            st.session_state.active_user = username   # ⭐ THIS WAS MISSING
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    st.stop()

# --- SIDEBAR NAV ---
with st.sidebar:
    st.title("🛡️ Synergy Guard")
    menu = st.radio("System Menu", ["Analysis Gateway", "Accounts & History"])
    st.divider()

    active_user = st.session_state.active_user
    st.write(f"👤 Logged in as: {active_user}")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.active_user = None
        st.rerun()

if menu == "Analysis Gateway":
    st.header(f"🔍 Security Gateway: {active_user}")
    
    col_input, col_viz = st.columns([1, 1])
    
    with col_input:
        user_input = st.text_area("Banking Command:", placeholder="e.g., Transfer 500 to Sashwat", height=150)
        process_btn = st.button("Execute Intent", type="primary")

    if process_btn and user_input:
        with st.spinner("Analyzing Security Layers..."):
            res = pipeline.predict(user_input)
            
            
            if any(word in user_input.lower() for word in ["ceo", "admin", "administrator"]):
                res['risk_score'] = max(res['risk_score'], 76.0) # Force HIGH
                res['risk_level'] = "HIGH"
                res['recommended_action'] = "BLOCK & ALERT: Unauthorized Executive Impersonation"

            
            with col_viz:
                lvl = res['risk_level']
                st.markdown(f"### Security Status: <span class='risk-{lvl.lower()}'>{lvl}</span>", unsafe_allow_html=True)
                st.progress(res['risk_score'] / 100)
                st.metric("Total Risk Score", f"{res['risk_score']}%")
                st.info(f"**Action:** {res['recommended_action']}")

            
            if res['risk_level'] == "LOW":
                
                amt_match = re.search(r'(\d+)', user_input)
                target_user = None
                for name in st.session_state.users:
                    if name.split()[0].lower() in user_input.lower(): 
                        target_user = name
                        break
                
                if amt_match and target_user and target_user != active_user:
                    amt = float(amt_match.group(1))
                    if st.session_state.users[active_user]['balance'] >= amt:
                        
                        st.session_state.users[active_user]['balance'] -= amt
                        st.session_state.users[active_user]['history'].append({
                            "Date": datetime.now().strftime("%H:%M:%S"),
                            "Type": "DEBIT", "Target": target_user, "Amount": f"-₹{amt}", "Status": "PASS"
                        })
                        
                        st.session_state.users[target_user]['balance'] += amt
                        st.session_state.users[target_user]['history'].append({
                            "Date": datetime.now().strftime("%H:%M:%S"),
                            "Type": "CREDIT", "Target": active_user, "Amount": f"+₹{amt}", "Status": "PASS"
                        })
                        st.success(f"Successfully transferred ₹{amt} to {target_user}")
                    else:
                        st.error("Transaction Failed: Insufficient Funds.")
                elif target_user == active_user:
                    st.warning("Cannot transfer to yourself.")
                else:
                    st.success("Query Processed (Non-financial request).")
            else:
                
                st.session_state.users[active_user]['history'].append({
                    "Date": datetime.now().strftime("%H:%M:%S"),
                    "Type": "ATTEMPT", "Target": "Unknown", "Amount": "N/A", "Status": "BLOCKED (HIGH RISK)"
                })
                st.error("Synergy blocked this request due to high security risk.")


elif menu == "Accounts & History":
    st.header("📋 Bank Ledger Management")
    
    
    view_person = st.session_state.active_user
    user_data = st.session_state.users[view_person]
    
    st.markdown(f"""
    <div style="background-color:#262730; padding:20px; border-radius:15px; border: 1px solid #444;">
        <h4>Account Holder: {view_person}</h3>
        <p><b>Account Number:</b> {user_data['acc']}</p>
        <h2 style="color:#00ff00;">₹{user_data['balance']:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader(f"Transaction Log: {view_person}")
    if user_data['history']:
        df = pd.DataFrame(user_data['history'])
        
        def style_status(val):
            color = '#ff4b4b' if 'BLOCKED' in str(val) else '#00ff00'
            return f'color: {color}'
        
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No transaction history for this user.")