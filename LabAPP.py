import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="California Housing Data", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'housing.csv')

@st.cache_data
def load_data():
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"文件未找到: {file_path}")
        return None

df = load_data()

if df is None:
    st.stop()

st.title('California Housing Data (1990) by Wenhan Ding')


price_filter = st.slider(
    'Select Price Range:',
    min_value=0,
    max_value=int(df['median_house_value'].max()),
    value=(0, 500000),  
    step=1000
)

location_filter = st.sidebar.multiselect(
    'Choose Location Type:',
    df.ocean_proximity.unique(),  
    df.ocean_proximity.unique()   
)


form = st.sidebar.form("income_form")
income_filter = form.radio(
    'Choose Income Level:',
    ('Low', 'Medium', 'High')
)
form.form_submit_button("Apply")

filtered_df = df[
    (df.median_house_value >= price_filter[0]) &
    (df.median_house_value <= price_filter[1])
]

filtered_df = filtered_df[filtered_df.ocean_proximity.isin(location_filter)]
if income_filter == 'Low':
    filtered_df = filtered_df[filtered_df.median_income <= 2.5]
elif income_filter == 'Medium':
    filtered_df = filtered_df[
        (filtered_df.median_income > 2.5) &
        (filtered_df.median_income < 4.5)
    ]
elif income_filter == 'High':
    filtered_df = filtered_df[filtered_df.median_income >= 4.5]
st.map(filtered_df)
st.subheader('Housing Details:')
st.write(filtered_df[['longitude', 'latitude', 'median_house_value', 'median_income', 'ocean_proximity']].head(20))
st.subheader('Distribution of Median House Values (≥ $200,000)')
hist_data = filtered_df[filtered_df.median_house_value >= 200000]

if len(hist_data) > 0:
    fig, ax = plt.subplots(figsize=(20, 5))
    min_val = hist_data.median_house_value.min()
    max_val = hist_data.median_house_value.max()
    bins = np.linspace(min_val, max_val, 31)  # 31个点创建30个bins

    hist_data.median_house_value.hist(bins=bins, ax=ax)
    ax.set_xlabel('Median House Value ($)')
    ax.set_ylabel('Number of Properties')
    ax.set_title(f'Distribution of House Values (≥ $200,000) - {len(bins) - 1} bins')
    ax.set_xlim(left=200000)
    st.pyplot(fig)
