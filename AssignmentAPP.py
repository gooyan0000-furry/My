import os
import subprocess
import sys

def install_from_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"安装依赖失败: {e}")

install_from_requirements()

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

# 创建示例数据（如果文件不存在）
@st.cache_data
def load_data():
    try:
        # 尝试读取本地文件
        df = pd.read_csv('train.csv')
        st.success("✅ 成功加载本地数据文件")
    except:
        st.warning("⚠️ 未找到本地数据文件，使用示例数据")
        # 创建示例数据
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

# 加载数据
df = load_data()

# 主标题
st.title("🚢 Titanic Data Analysis App")
st.markdown("**Developed by: Zhang San**")  # 请替换为您的姓名

# 显示数据基本信息
st.sidebar.title("🔍 Navigation")
st.sidebar.info("Select an analysis type from the dropdown below")

analysis_type = st.sidebar.selectbox(
    "选择分析类型",
    [
        "数据概览", 
        "登船港口分析", 
        "性别生存分析", 
        "票价分析", 
        "客舱等级生存分析",
        "详细分布分析"
    ]
)

# 1. 数据概览
if analysis_type == "数据概览":
    st.header("📊 数据集概览")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_passengers = len(df)
        st.metric("总乘客数", total_passengers)
    
    with col2:
        survival_rate = df['Survived'].mean() * 100
        st.metric("总体生存率", f"{survival_rate:.1f}%")
    
    with col3:
        avg_age = df['Age'].mean()
        st.metric("平均年龄", f"{avg_age:.1f}岁")
    
    with col4:
        avg_fare = df['Fare'].mean()
        st.metric("平均票价", f"${avg_fare:.2f}")
    
    # 数据预览
    st.subheader("数据预览")
    st.dataframe(df.head(8), use_container_width=True)
    
    # 数据集信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("数据集信息")
        st.write(f"**数据形状:** {df.shape[0]} 行 × {df.shape[1]} 列")
        st.write("**列名:**", list(df.columns))
    
    with col2:
        st.subheader("数据类型")
        st.write(df.dtypes)

# 2. 登船港口分析
elif analysis_type == "登船港口分析":
    st.header("🌊 登船港口分析")
    
    # 计算港口统计
    embarkation_stats = df['Embarked'].value_counts()
    embarkation_percentages = (df['Embarked'].value_counts(normalize=True) * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("港口乘客分布")
        
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
        st.subheader("可视化图表")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ports = ['Southampton (S)', 'Cherbourg (C)', 'Queenstown (Q)']
        counts = [embarkation_stats.get('S', 0), embarkation_stats.get('C', 0), embarkation_stats.get('Q', 0)]
        
        bars = ax.bar(ports, counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title('各登船港口乘客数量', fontsize=14, fontweight='bold')
        ax.set_ylabel('乘客数量')
        
        # 在柱子上添加数值
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        st.pyplot(fig)

# 3. 性别生存分析
elif analysis_type == "性别生存分析":
    st.header("🚻 性别生存分析")
    
    # 计算性别生存率
    survival_by_gender = df.groupby('Sex')['Survived'].agg(['mean', 'count', 'sum'])
    survival_by_gender['survival_rate'] = (survival_by_gender['mean'] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("生存统计")
        
        for gender in survival_by_gender.index:
            rate = survival_by_gender.loc[gender, 'survival_rate']
            survived = int(survival_by_gender.loc[gender, 'sum'])
            total = int(survival_by_gender.loc[gender, 'count'])
            
            gender_display = "男性" if gender == 'male' else "女性"
            st.metric(
                label=f"{gender_display}生存率",
                value=f"{rate}%",
                delta=f"{survived}/{total} 人存活"
            )
    
    with col2:
        st.subheader("生存率对比")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        genders = ['男性', '女性']
        rates = [
            survival_by_gender.loc['male', 'survival_rate'],
            survival_by_gender.loc['female', 'survival_rate']
        ]
        
        bars = ax.bar(genders, rates, color=['#3498db', '#e84393'], alpha=0.8)
        ax.set_title('性别生存率对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('生存率 (%)')
        ax.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)

# 4. 票价分析
elif analysis_type == "票价分析":
    st.header("💰 票价分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("票价统计")
        
        st.metric("最高票价", f"${df['Fare'].max():.2f}")
        st.metric("最低票价", f"${df['Fare'].min():.2f}")
        st.metric("平均票价", f"${df['Fare'].mean():.2f}")
        st.metric("票价中位数", f"${df['Fare'].median():.2f}")
    
    with col2:
        st.subheader("票价分布控制")
        
        fare_range = st.slider(
            "选择票价范围",
            min_value=0.0,
            max_value=float(df['Fare'].max()),
            value=(0.0, 100.0),
            step=5.0
        )
        
        # 根据选择过滤数据
        filtered_fares = df[(df['Fare'] >= fare_range[0]) & (df['Fare'] <= fare_range[1])]['Fare']
        
        st.write(f"**选中范围乘客数:** {len(filtered_fares)} 人")
        st.write(f"**选中范围平均票价:** ${filtered_fares.mean():.2f}")
    
    # 票价分布直方图
    st.subheader("票价分布直方图")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(df['Fare'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_title('票价分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('票价 ($)')
    ax.set_ylabel('乘客数量')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# 5. 客舱等级生存分析
elif analysis_type == "客舱等级生存分析":
    st.header("🎫 客舱等级生存分析")
    
    # 按客舱等级计算生存率
    survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("各等级生存率")
        
        for pclass in sorted(survival_by_class.index):
            rate = survival_by_class[pclass]
            class_data = df[df['Pclass'] == pclass]
            survived = class_data['Survived'].sum()
            total = len(class_data)
            
            st.metric(
                label=f"{pclass} 等舱生存率",
                value=f"{rate:.1f}%",
                delta=f"{survived}/{total} 人存活"
            )
        
        # 找出生存率最高的等级
        best_class = survival_by_class.idxmax()
        best_rate = survival_by_class.max()
        st.success(f"🎯 **{best_class} 等舱生存率最高: {best_rate:.1f}%**")
    
    with col2:
        st.subheader("生存率对比图")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        classes = ['1 等舱', '2 等舱', '3 等舱']
        rates = [survival_by_class[1], survival_by_class[2], survival_by_class[3]]
        colors = ['#f1c40f', '#95a5a6', '#e67e22']
        
        bars = ax.bar(classes, rates, color=colors, alpha=0.8)
        ax.set_title('各客舱等级生存率', fontsize=14, fontweight='bold')
        ax.set_ylabel('生存率 (%)')
        ax.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)

# 6. 详细分布分析
elif analysis_type == "详细分布分析":
    st.header("📈 详细分布分析")
    
    # 计算每个(等级, 生存状态)组合的比例
    survival_proportions = pd.crosstab(df['Pclass'], df['Survived'], normalize='index')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("各等级内部生存分布")
        
        # 创建比例表格
        proportion_table = (survival_proportions * 100).round(1)
        proportion_table.columns = ['未生存 (%)', '已生存 (%)']
        proportion_table.index = ['1 等舱', '2 等舱', '3 等舱']
        
        st.dataframe(proportion_table)
        
        st.info("💡 **说明**: 每行总和为100%，显示各等级内部的生存分布")
    
    with col2:
        st.subheader("堆叠条形图")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        survival_proportions.plot(kind='bar', stacked=True, ax=ax,
                                 color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title('各等级生存分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('客舱等级')
        ax.set_ylabel('比例')
        ax.legend(['未生存', '已生存'])
        plt.xticks(rotation=0)
        
        st.pyplot(fig)
    
    # 详细柱状图
    st.subheader("详细分布柱状图")
    
    # 准备数据
    categories = []
    values = []
    
    for pclass in [1, 2, 3]:
        for survived in [0, 1]:
            categories.append(f'({pclass}, {survived})')
            values.append(survival_proportions.loc[pclass, survived])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c' if i % 2 == 0 else '#2ecc71' for i in range(len(categories))]
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_title('各(等级, 生存状态)组合的比例', fontsize=14, fontweight='bold')
    ax.set_xlabel('(客舱等级, 生存状态)')
    ax.set_ylabel('比例')
    ax.set_ylim(0, 1.0)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)

# 页脚
st.sidebar.markdown("---")
st.sidebar.info(
    "📖 **关于此应用**:\n"
    "这是一个泰坦尼克号数据分析应用，用于探索乘客 demographics、"
    "生存模式和其他有趣的洞察。"
)

# 在侧边栏显示数据来源
st.sidebar.markdown("---")
st.sidebar.caption("数据来源: Titanic dataset")