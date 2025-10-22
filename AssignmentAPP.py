import os
import subprocess
import sys

def install_from_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("Suceess")
        except subprocess.CalledProcessError as e:
            print(f"Failure: {e}")

install_from_requirements()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Titanic Data Analysis by DingWenhan",
    page_icon="🚢",
    layout="wide"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('train.csv')
        st.success("✅ Successfully loaded data from local file")
    except:
        st.warning("⚠️ Unsuccessful loading data from local file.")
        np.random.seed(42)
        n_passengers = 891
        
        data = {
            'PassengerId': range(1, n_passengers + 1),
            'Survived': np.random.choice([0, 1], n_passengers, p=[0.6, 0.4]),
            'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.25, 0.35, 0.4]),
            'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 15, n_passengers).clip(0, 80),
            'Fare': np.random.exponential(30, n_passengers).clip(0, 500),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.7, 0.2, 0.1])
        }
        df = pd.DataFrame(data)
        df['Age'] = df['Age'].round(1)
        df['Fare'] = df['Fare'].round(2)
    
    return df

df = load_data()

st.title("🚢 Titanic Data Analysis App")
st.markdown("Developed by: DingWenhan")  # 请替换为您的姓名

st.sidebar.title("🔍 Navigation")
st.sidebar.info("Select an analysis type from the dropdown below")

analysis_type = st.sidebar.selectbox(
    "Choose Analaysis Type",
    [
        "Data Overview", 
        "Embarkation Analysis", 
        "Gender Survival Analysis", 
        "Fare Analysis", 
        "Cabin Class Survival Analysis",
        "Detailed Distribution Analysis"
        "Passenger Survival Rate according to Pclass",
        "Detailed Passenger Analysis"
    ]
)

# 1. 数据概览
if analysis_type == "Data Overview":
    st.header("📊 Data Overview")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_passengers = len(df)
        st.metric("Total Passengers", total_passengers)
    
    with col2:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    
    with col3:
        avg_age = df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years old")
    
    with col4:
        avg_fare = df['Fare'].mean()
        st.metric("Average Fare", f"${avg_fare:.2f}")
    
    st.subheader("Overview of the Data")
    st.dataframe(df.head(8), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Type")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("**Columns:**", list(df.columns))
    
    with col2:
        st.subheader("Data Type")
        st.write(df.dtypes)


elif analysis_type == "Embarkation Analysis":
    st.header("🌊 Embarkation Analysis")
    
    embarkation_stats = df['Embarked'].value_counts()
    embarkation_percentages = (df['Embarked'].value_counts(normalize=True) * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Port Passenger Distribution")
        
        port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
        
        for port in ['S', 'C', 'Q']:
            count = embarkation_stats.get(port, 0)
            percentage = embarkation_percentages.get(port, 0)
            st.metric(
                label=f"{port} ({port_names.get(port, 'Unknown')})",
                value=f"{count} 人",
                delta=f"{percentage}%"
            )
    
    with col2:
        st.subheader("Port Passenger Distribution Graphs")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ports = ['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)']
        counts = [embarkation_stats.get('S', 0), embarkation_stats.get('C', 0), embarkation_stats.get('Q', 0)]
        
        bars = ax.bar(ports, counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title('Port Passenger Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Numbers of Passengers')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif analysis_type == "Gender Survival Analysis":
    st.header("🚻 Gender Survival Analysis")
    
    survival_by_gender = df.groupby('Sex')['Survived'].agg(['mean', 'count', 'sum'])
    survival_by_gender['survival_rate'] = (survival_by_gender['mean'] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Survival Rate")
        
        for gender in survival_by_gender.index:
            rate = survival_by_gender.loc[gender, 'survival_rate']
            survived = int(survival_by_gender.loc[gender, 'sum'])
            total = int(survival_by_gender.loc[gender, 'count'])
            
            gender_display = "Male" if gender == 'male' else "Female"
            st.metric(
                label=f"{gender_display} Survival Rate",
                value=f"{rate}%",
                delta=f"{survived}/{total} Passengers Survived"
            )
    
    with col2:
        st.subheader("Gender Survival Rate Graphs")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        genders = ['Male', 'Female']
        rates = [
            survival_by_gender.loc['male', 'survival_rate'],
            survival_by_gender.loc['female', 'survival_rate']
        ]
        
        bars = ax.bar(genders, rates, color=['#3498db', '#e84393'], alpha=0.8)
        ax.set_title('Gender Survival Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_ylim(0, 100)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)

elif analysis_type == "Fare Analysis":
    st.header("💰 Fare Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fare Statistics")
        
        st.metric("Highest Fare", f"${df['Fare'].max():.2f}")
        st.metric("Lowest Fare", f"${df['Fare'].min():.2f}")
        st.metric("Average Fare", f"${df['Fare'].mean():.2f}")
        st.metric("Median Fare", f"${df['Fare'].median():.2f}")
    
    with col2:
        st.subheader("Fare Distribution Control")
        
        fare_range = st.slider(
            "Select Fare Range",
            min_value=0.0,
            max_value=float(df['Fare'].max()),
            value=(0.0, 100.0),
            step=5.0
        )
        
        filtered_fares = df[(df['Fare'] >= fare_range[0]) & (df['Fare'] <= fare_range[1])]['Fare']
        
        st.write(f"** Selected:** {len(filtered_fares)} 人")
        st.write(f"** Average Fare:** ${filtered_fares.mean():.2f}")
    
    
    st.subheader("Fare Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(df['Fare'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_title('Fare Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fare ($)')
    ax.set_ylabel('Number of Passengers')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

elif analysis_type == "Cabin Class Survival Analysis":
    st.header("🎫 Cabin Class Survival Analysis")
    survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cabin Class Survival Rate")
        
        for pclass in sorted(survival_by_class.index):
            rate = survival_by_class[pclass]
            class_data = df[df['Pclass'] == pclass]
            survived = class_data['Survived'].sum()
            total = len(class_data)
            
            st.metric(
                label=f"{pclass} Class Survival Rate",
                value=f"{rate:.1f}%",
                delta=f"{survived}/{total} Passengers Survived"
            )

        best_class = survival_by_class.idxmax()
        best_rate = survival_by_class.max()
        st.success(f"🎯 **{best_class} Class** has the **highest survival rate: {best_rate:.1f}%**")
    
    with col2:
        st.subheader("Cabin Class Survival Rate Graphs")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        classes = ['1 st Class', '2 nd Class', '3 rd Class']
        rates = [survival_by_class[1], survival_by_class[2], survival_by_class[3]]
        colors = ['#f1c40f', '#95a5a6', '#e67e22']
        
        bars = ax.bar(classes, rates, color=colors, alpha=0.8)
        ax.set_title('Cabin Class Survival Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_ylim(0, 100)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)


elif analysis_type == "Passenger Survival Rate according to Pclass":
    st.header("📈 Passenger Survival Rate according to Pclass")
    
    survival_proportions = pd.crosstab(df['Pclass'], df['Survived'], normalize='index')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Survival Proportions")
        proportion_table = (survival_proportions * 100).round(1)
        proportion_table.columns = ['Died (%)', 'Survived (%)']
        proportion_table.index = ['1st Class', '2nd Class', '3rd Class']
        st.dataframe(proportion_table)
    
    with col2:
        st.subheader("Survival Rate Graphs")
        fig, ax = plt.subplots(figsize=(10, 6))
        survival_proportions.plot(kind='bar', stacked=True, ax=ax,
                                 color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title('Passenger Survival Rate according to Pclass', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pclass')
        ax.set_ylabel('Proportion')
        ax.legend(['Died', 'Survived'])
        plt.xticks(rotation=0)
        st.pyplot(fig)
    
    st.subheader("Details of Survival Rate")
    
    categories = []
    values = []
    
    for pclass in [1, 2, 3]:
        for survived in [0, 1]:
            categories.append(f'({pclass}, {survived})')
            values.append(survival_proportions.loc[pclass, survived])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c' if i % 2 == 0 else '#2ecc71' for i in range(len(categories))]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Passenger Survival Rate according to Pclass', fontsize=14, fontweight='bold')
    ax.set_xlabel('(Pclass, Survived)')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1.0)
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)


st.sidebar.markdown("---")
st.sidebar.markdown("## About the App")
st.sidebar.markdown("This app is developed by DingWenhan for the Titanic dataset analysis. It is a simple and interactive tool for data exploration and analysis. The app is built using the Streamlit library in Python. The data is loaded from a local file and cached for better performance. The app is designed to be used as a standalone app or as a part of a larger data analysis project. The app is open-source and available on GitHub. You can find the source code and instructions on how to run the app on your machine in the following links:")
st.sidebar.markdown("- [GitHub Repository](https://github.com/gooyan0000-furry/My)")

st.sidebar.markdown("---")
st.sidebar.caption("Sources: Titanic dataset")