import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def generate_ml_data(selected_groups):
    np.random.seed(42)
    data = []
    
    # 1. Outside Wide Receivers: Very High Receiving, Near-Zero Rushing
    if 'Outside WR' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(5, 5, 40),
            'Rec': np.random.normal(1100, 100, 40),
            'True Role': 'Outside WR'
        }))
    
    # 2. Slot Receivers: High Receiving, Some Rushing (Jet Sweeps)
    if 'Slot WR' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(80, 40, 35),
            'Rec': np.random.normal(850, 120, 35),
            'True Role': 'Slot WR'
        }))
    
    # 3. Running Backs (Pure): High Rushing, Low Receiving
    if 'RB (Pure)' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(1100, 120, 30),
            'Rec': np.random.normal(150, 60, 30),
            'True Role': 'RB (Pure)'
        }))
    
    # 4. Running Backs (Hybrid): High Rushing, Med Receiving
    if 'RB (Hybrid)' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(850, 150, 30),
            'Rec': np.random.normal(450, 100, 30),
            'True Role': 'RB (Hybrid)'
        }))
    
    # 5. Tight Ends: Medium of both
    if 'Tight End' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(25, 15, 30),
            'Rec': np.random.normal(650, 120, 30),
            'True Role': 'Tight End'
        }))
    
    # 6. Traditional Quarterbacks: Low Rushing, Zero Receiving
    if 'QB (Pocket)' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(50, 30, 25),
            'Rec': np.random.normal(0, 2, 25),
            'True Role': 'QB (Pocket)'
        }))

    # 7. Dual-Threat Quarterbacks: High Rushing Outliers, Zero Receiving
    if 'QB (Dual-Threat)' in selected_groups:
        data.append(pd.DataFrame({
            'Rush': np.random.normal(800, 100, 15),
            'Rec': np.random.normal(0, 2, 15),
            'True Role': 'QB (Dual-Threat)'
        }))
    
    if not data:
        return pd.DataFrame(columns=['Rush', 'Rec', 'True Role'])
        
    df = pd.concat(data).reset_index(drop=True)
    df[['Rush', 'Rec']] = df[['Rush', 'Rec']].clip(lower=0)
    return df

def select_roles():
    st.subheader("1. Data Composition")
    all_roles = ['Outside WR', 'Slot WR', 'RB (Pure)', 'RB (Hybrid)', 'Tight End', 'QB (Pocket)', 'QB (Dual-Threat)']
    return st.multiselect("Select Roles to include", all_roles, default=['Outside WR', 'RB (Pure)', 'QB (Pocket)', 'Tight End'])