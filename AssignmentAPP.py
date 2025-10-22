import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Titanic Data Analysis",
    page_icon="🚢",
    layout="wide"
)

# Create sample data (if file does not exist)
@st.cache_data
def load_data():
    try:
        # Try to read local file
        df = pd.read_csv('train.csv')
        st.success("✅ Successfully loaded local data file")
    except:
        st.warning("⚠️ Local data file not found, using sample data")
        # Create sample data
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

# Load data
df = load_data()

# Main title
st.title("🚢 Titanic Data Analysis App")
st.markdown("**Developed by: Wenhan Ding**")

# Display basic information of data
st.sidebar.title("🔍 Navigation")
st.sidebar.info("Select an analysis type from the dropdown below")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "Data Overview",
        "Embarkation Port Analysis",
        "Gender Survival Analysis",
        "Fare Analysis",
        "Passenger Class Survival Analysis",
        "Detailed Distribution Analysis"
    ]
)

# 1. Data Overview
if analysis_type == "Data Overview":
    st.header("📊 Data Set Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_passengers = len(df)
        st.metric("Total Passengers", total_passengers)

    with col2:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Overall Survival Rate", f"{survival_rate:.1f}%")

    with col3:
        avg_age = df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")

    with col4:
        avg_fare = df['Fare'].mean()
        st.metric("Average Fare", f"${avg_fare:.2f}")

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(8), use_container_width=True)

    # Data set information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Set Information")
        st.write(f"**Data Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("**Column Names:**", list(df.columns))

    with col2:
        st.subheader("Data Types")
        st.write(df.dtypes)

# 2. Embarkation Port Analysis
elif analysis_type == "Embarkation Port Analysis":
    st.header("🌊 Embarkation Port Analysis")

    # Calculate embarkation statistics
    embarkation_stats = df['Embarked'].value_counts()
    embarkation_percentages = (df['Embarked'].value_counts(normalize=True) * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Passenger Distribution by Port")

        port_names = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}

        for port in ['S', 'C', 'Q']:
            count = embarkation_stats.get(port, 0)
            percentage = embarkation_percentages.get(port, 0)
            st.metric(
                label=f"{port} ({port_names.get(port, 'Unknown')})",
                value=f"{count} passengers",
                delta=f"{percentage}%"
            )

    with col2:
        st.subheader("Visualization Chart")
        fig, ax = plt.subplots(figsize=(10, 6))

        ports = ['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)']
        counts = [embarkation_stats.get('S', 0), embarkation_stats.get('C', 0), embarkation_stats.get('Q', 0)]

        bars = ax.bar(ports, counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title('Passenger Count by Embarkation Port', fontsize=14, fontweight='bold')
        ax.set_ylabel('Passenger Count')

        # Add value labels to bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45)
        st.pyplot(fig)

# 3. Gender Survival Analysis
elif analysis_type == "Gender Survival Analysis":
    st.header("🚻 Gender Survival Analysis")

    # Calculate survival rate by gender
    survival_by_gender = df.groupby('Sex')['Survived'].agg(['mean', 'count', 'sum'])
    survival_by_gender['survival_rate'] = (survival_by_gender['mean'] * 100).round(1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Survival Statistics")

        for gender in survival_by_gender.index:
            rate = survival_by_gender.loc[gender, 'survival_rate']
            survived = int(survival_by_gender.loc[gender, 'sum'])
            total = int(survival_by_gender.loc[gender, 'count'])

            gender_display = "Male" if gender == 'male' else "Female"
            st.metric(
                label=f"{gender_display} Survival Rate",
                value=f"{rate}%",
                delta=f"{survived}/{total} passengers survived"
            )

    with col2:
        st.subheader("Survival Rate Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))

        genders = ['Male', 'Female']
        rates = [
            survival_by_gender.loc['male', 'survival_rate'],
            survival_by_gender.loc['female', 'survival_rate']
        ]

        bars = ax.bar(genders, rates, color=['#3498db', '#e84393'], alpha=0.8)
        ax.set_title('Survival Rate by Gender', fontsize=14, fontweight='bold')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate}%', ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig)

# 4. Fare Analysis
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

        # Filter data based on selection
        filtered_fares = df[(df['Fare'] >= fare_range[0]) & (df['Fare'] <= fare_range[1])]['Fare']
        st.write(f"**Selected Range Passenger Count:** {len(filtered_fares)} passengers")
        st.write(f"**Selected Range Average Fare:** ${filtered_fares.mean():.2f}")

    # Fare distribution histogram
    st.subheader("Fare Distribution Histogram")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(df['Fare'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_title('Fare Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fare ($)')
    ax.set_ylabel('Passenger Count')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# 5. Passenger Class Survival Analysis
elif analysis_type == "Passenger Class Survival Analysis":
    st.header("🎫 Passenger Class Survival Analysis")

    # Calculate survival rate by passenger class
    survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Survival Rate by Class")

        for pclass in sorted(survival_by_class.index):
            rate = survival_by_class[pclass]
            class_data = df[df['Pclass'] == pclass]
            survived = class_data['Survived'].sum()
            total = len(class_data)

            st.metric(
                label=f"{pclass} Class Survival Rate",
                value=f"{rate:.1f}%",
                delta=f"{survived}/{total} passengers survived"
            )

        # Find the class with the highest survival rate
        best_class = survival_by_class.idxmax()
        best_rate = survival_by_class.max()
        st.success(f"🎯 **Highest Survival Rate in {best_class} Class: {best_rate:.1f}%**")

    with col2:
        st.subheader("Survival Rate Comparison Chart")
        fig, ax = plt.subplots(figsize=(8, 6))

        classes = ['1 Class', '2 Class', '3 Class']
        rates = [survival_by_class[1], survival_by_class[2], survival_by_class[3]]
        colors = ['#f1c40f', '#95a5a6', '#e67e22']

        bars = ax.bar(classes, rates, color=colors, alpha=0.8)
        ax.set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('Survival Rate (%)')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig)

# 6. Detailed Distribution Analysis
elif analysis_type == "Detailed Distribution Analysis":
    st.header("📈 Detailed Distribution Analysis")

    # Calculate survival proportions for each (class, survival status) combination
    survival_proportions = pd.crosstab(df['Pclass'], df['Survived'], normalize='index')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Survival Distribution within Each Class")

        # Create proportion table
        proportion_table = (survival_proportions * 100).round(1)
        proportion_table.columns = ['Did Not Survive (%)', 'Survived (%)']
        proportion_table.index = ['1 Class', '2 Class', '3 Class']

        st.dataframe(proportion_table)
        st.info("💡 **Note**: Each row sums to 100%, showing the survival distribution within each class")

    with col2:
        st.subheader("Stacked Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 6))

        survival_proportions.plot(kind='bar', stacked=True, ax=ax,
                                  color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title('Survival Distribution by Passenger Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Passenger Class')
        ax.set_ylabel('Proportion')
        ax.legend(['Did Not Survive', 'Survived'])
        plt.xticks(rotation=0)

        st.pyplot(fig)

    # Detailed bar chart
    st.subheader("Detailed Distribution Bar Chart")

    # Prepare data
    categories = []
    values = []

    for pclass in [1, 2, 3]:
        for survived in [0, 1]:
            categories.append(f'({pclass}, {survived})')
            values.append(survival_proportions.loc[pclass, survived])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c' if i % 2 == 0 else '#2ecc71' for i in range(len(categories))]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('Proportion of Each (Class, Survival Status) Combination', fontsize=14, fontweight='bold')
    ax.set_xlabel('(Passenger Class, Survival Status)')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1.0)

    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "📖 **About This App**:\n"
    "This is a Titanic data analysis app, used to explore passenger demographics, "
    "survival patterns and other interesting insights."
)

# Display data source in the sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Data Source: Titanic dataset")
