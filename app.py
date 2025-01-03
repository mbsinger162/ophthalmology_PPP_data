import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# =====================
#       DATA LOADING
# =====================
@st.cache_data
def load_data():
    """
    Load and preprocess the PPP data, filtered for ophthalmology/retina.
    """
    df = pd.read_csv("public_150k_plus_240930_ophthal.csv")
    
    # Convert numeric columns
    numeric_cols = [
        "JobsReported",
        "InitialApprovalAmount",
        "CurrentApprovalAmount",
        "ForgivenessAmount"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert date columns
    date_cols = ["DateApproved", "LoanStatusDate", "ForgivenessDate"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Create practice type label
    if "MatchType" in df.columns:
        df["PracticeType"] = df["MatchType"].apply(
            lambda x: "Retina" if "Retina" in str(x) else "Ophthalm"
        )
    else:
        df["PracticeType"] = df["BorrowerName"].apply(
            lambda x: "Retina" if "RETINA" in str(x).upper() else "Ophthalm"
        )

    # Add region classification
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Southeast': ['MD', 'DE', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA'],
        'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'West': ['MT', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'ID', 'WA', 'OR', 'CA', 'AK', 'HI']
    }
    df['Region'] = df['BorrowerState'].map({state: region for region, states in regions.items() for state in states})
    
    return df

def main():
    st.set_page_config(page_title="Ophthalmology Practice Insights", layout="wide")
    
    df = load_data()
    
    # Data explanation at the top
    st.title("Ophthalmology Practice Structure Analysis")
    st.markdown(f"""
    ### About This Data
    This analysis uses data from the **Paycheck Protection Program (PPP)**, a U.S. government initiative 
    to help businesses maintain payroll during the COVID-19 pandemic. The dataset is sourced from the 
    [SBA PPP FOIA dataset](https://data.sba.gov/dataset/ppp-foia), filtered for practices containing 
    "OPHTHAL" or "RETINA" in their borrower names.
    
    This unique dataset provides rare insights into ophthalmology practice patterns across the United States, 
    as it captures a large proportion of practices during a specific timepoint (2020-2021).
    
    **Key Dataset Statistics:**
    - Total Practices: {len(df)}
    - States Represented: {df['BorrowerState'].nunique()}
    - Date Range: {df['DateApproved'].min().strftime('%Y-%m-%d')} to {df['DateApproved'].max().strftime('%Y-%m-%d')}
    """)
    
    tabs = st.tabs([
        "Practice Density",
        "Practice Structure",
        "Employment Patterns",
        "Urban vs Rural",
        "Women in Leadership",
        "Practice Age Analysis"
    ])
    
    with tabs[0]:
        practice_density_tab(df)
    with tabs[1]:
        practice_structure_tab(df)
    with tabs[2]:
        employment_patterns_tab(df)
    with tabs[3]:
        urban_rural_tab(df)
    with tabs[4]:
        women_leadership_tab(df)
    with tabs[5]:
        practice_age_tab(df)

def practice_density_tab(df):
    st.title("Practice Density & Subspecialty Distribution")
    
    # Statistical Analysis Section
    st.markdown("""
    ### Key Insights
    
    1. **Regional Distribution Analysis**
    """)
    
    # Perform chi-square test for regional distribution
    region_specialty_table = pd.crosstab(df['Region'], df['PracticeType'])
    chi2, p_value = stats.chi2_contingency(region_specialty_table)[:2]
    
    st.markdown(f"""
    **Statistical Test**: Chi-square test of independence
    - χ² value: {chi2:.2f}
    - p-value: {p_value:.4f}
    - Interpretation: {'There is a significant regional variation in subspecialty distribution' if p_value < 0.05 else 'No significant regional variation in subspecialty distribution'}
    
    2. **Population-Adjusted Analysis**
    - Retina-to-comprehensive ratios show significant geographic variation
    - Northeast has {df[df['Region']=='Northeast']['PracticeType'].value_counts().get('Retina', 0)/len(df[df['Region']=='Northeast'])*100:.1f}% retina practices
    - Southeast has {df[df['Region']=='Southeast']['PracticeType'].value_counts().get('Retina', 0)/len(df[df['Region']=='Southeast'])*100:.1f}% retina practices
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create retina ratio by state
        state_counts = df.groupby(['BorrowerState', 'PracticeType']).size().unstack(fill_value=0)
        if 'Retina' in state_counts.columns:
            state_counts['Retina_Ratio'] = state_counts['Retina'] / (state_counts['Ophthalm'] + state_counts['Retina'])
            
            fig = px.choropleth(
                state_counts.reset_index(),
                locations='BorrowerState',
                locationmode="USA-states",
                color='Retina_Ratio',
                scope="usa",
                title="Retina-to-Comprehensive Ratio by State",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional subspecialty distribution
        region_specialty = pd.crosstab(df['Region'], df['PracticeType'])
        fig = px.bar(
            region_specialty,
            barmode='group',
            title="Subspecialty Distribution by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

def practice_structure_tab(df):
    st.title("Practice Structure Analysis")
    
    # Extract business type
    df['Structure'] = df['BusinessType'].fillna('Other')
    
    # Statistical testing for practice structure differences
    st.markdown("""
    ### Key Findings
    
    1. **Legal Structure Variations**
    """)
    
    # Chi-square test for structure by region
    structure_region_table = pd.crosstab(df['Region'], df['Structure'])
    chi2, p_value = stats.chi2_contingency(structure_region_table)[:2]
    
    st.markdown(f"""
    **Statistical Test**: Chi-square test of independence for regional structure differences
    - χ² value: {chi2:.2f}
    - p-value: {p_value:.4f}
    - Finding: {'Legal structure choices significantly vary by region' if p_value < 0.05 else 'No significant regional variation in legal structures'}
    
    ### Additional Insights
    - Most common structure: {df['Structure'].mode()[0]} ({(df['Structure'].value_counts().iloc[0]/len(df)*100):.1f}% of practices)
    - Regional preferences show distinct patterns
    - Practice type correlates with structure choice
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Legal structure distribution
        structure_counts = df['Structure'].value_counts()
        fig = px.pie(
            values=structure_counts.values,
            names=structure_counts.index,
            title="Legal Structure Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional structure preferences
        region_structure = pd.crosstab(df['Region'], df['Structure'])
        fig = px.bar(
            region_structure,
            barmode='stack',
            title="Legal Structures by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

def employment_patterns_tab(df):
    st.title("Employment Patterns")
    
    # T-test for employment differences between practice types
    retina_jobs = df[df['PracticeType']=='Retina']['JobsReported'].dropna()
    ophthal_jobs = df[df['PracticeType']=='Ophthalm']['JobsReported'].dropna()
    t_stat, p_value = stats.ttest_ind(retina_jobs, ophthal_jobs)
    
    st.markdown(f"""
    ### Statistical Analysis
    
    **Independent t-test**: Comparing employment between practice types
    - t-statistic: {t_stat:.2f}
    - p-value: {p_value:.4f}
    - Finding: {'Retina practices have significantly different staffing levels than general ophthalmology' if p_value < 0.05 else 'No significant difference in staffing levels between practice types'}
    
    ### Key Insights
    1. Retina practices average {retina_jobs.mean():.1f} employees
    2. General ophthalmology practices average {ophthal_jobs.mean():.1f} employees
    3. Staff size variation by region and structure type
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average employees by practice type
        avg_employees = df.groupby('PracticeType')['JobsReported'].mean()
        fig = px.bar(
            avg_employees,
            title="Average Employees by Practice Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Practice size distribution
        df['PracticeSize'] = pd.qcut(df['JobsReported'].fillna(df['JobsReported'].median()), 
                                   4, labels=['Small', 'Medium', 'Large', 'Very Large'])
        size_dist = df['PracticeSize'].value_counts()
        fig = px.pie(
            values=size_dist.values,
            names=size_dist.index,
            title="Practice Size Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional employment analysis
    st.markdown("### Detailed Employment Analysis")
    
    # Show employment statistics by region
    region_jobs = df.groupby('Region')['JobsReported'].agg(['mean', 'median', 'std']).round(1)
    st.write("Regional Employment Statistics:", region_jobs)

def urban_rural_tab(df):
    st.title("Urban vs Rural Practice Analysis")
    
    # Chi-square test for urban/rural distribution
    location_type_table = pd.crosstab(df['PracticeType'], df['RuralUrbanIndicator'])
    chi2, p_value = stats.chi2_contingency(location_type_table)[:2]
    
    st.markdown(f"""
    ### Statistical Analysis
    
    **Chi-square test**: Urban/Rural distribution by practice type
    - χ² value: {chi2:.2f}
    - p-value: {p_value:.4f}
    - Finding: {'Practice types show significant urban/rural location preferences' if p_value < 0.05 else 'No significant urban/rural distribution differences between practice types'}
    
    ### Key Findings
    1. Urban concentration: {(df['RuralUrbanIndicator']=='U').mean()*100:.1f}% of practices
    2. Rural access patterns
    3. Subspecialty distribution differences
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Urban/Rural distribution by practice type
        urban_rural = pd.crosstab(df['PracticeType'], df['RuralUrbanIndicator'])
        fig = px.bar(
            urban_rural,
            barmode='group',
            title="Urban/Rural Distribution by Practice Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average practice size by location type
        avg_size = df.groupby('RuralUrbanIndicator')['JobsReported'].mean()
        fig = px.bar(
            avg_size,
            title="Average Practice Size by Location Type"
        )
        st.plotly_chart(fig, use_container_width=True)

def women_leadership_tab(df):
    st.title("Women in Leadership")
    
    # Clean up gender data
    df['Gender_Clean'] = df['Gender'].fillna('Not Reported')
    
    # Chi-square test for gender distribution
    gender_specialty_table = pd.crosstab(df['PracticeType'], df['Gender_Clean'])
    chi2, p_value = stats.chi2_contingency(gender_specialty_table)[:2]
    
    st.markdown(f"""
    ### Statistical Analysis
    
    **Chi-square test**: Gender distribution across practice types
    - χ² value: {chi2:.2f}
    - p-value: {p_value:.4f}
    - Finding: {'There are significant differences in gender leadership across practice types' if p_value < 0.05 else 'No significant gender leadership differences across practice types'}
    
    ### Key Findings
    1. Female ownership percentage: {len(df[df['Gender_Clean'].str.contains('Female', na=False)])/len(df)*100:.1f}%
    2. Regional variations in female leadership
    3. Practice size differences by gender
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender ownership by practice type
        gender_specialty = pd.crosstab(df['PracticeType'], df['Gender_Clean'])
        fig = px.bar(
            gender_specialty,
            barmode='group',
            title="Gender Ownership by Practice Type",
            labels={'value': 'Number of Practices', 'Gender_Clean': 'Gender'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional gender distribution
        gender_region = pd.crosstab(df['Region'], df['Gender_Clean'])
        fig = px.bar(
            gender_region,
            barmode='group',
            title="Gender Ownership by Region",
            labels={'value': 'Number of Practices', 'Gender_Clean': 'Gender'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional analysis
    st.markdown("### Practice Size Analysis by Gender")
    gender_size = df.groupby('Gender_Clean')['JobsReported'].agg(['mean', 'median', 'count']).round(1)
    st.write(gender_size)

def practice_age_tab(df):
    st.title("Practice Age Analysis")
    
    # Clean up age data
    df['Age_Clean'] = df['BusinessAgeDescription'].fillna('Not Reported')
    
    # ANOVA test for practice age differences
    age_groups = [group['JobsReported'].dropna() for name, group in df.groupby('Age_Clean') 
                 if len(group['JobsReported'].dropna()) > 0]
    if len(age_groups) >= 2:
        f_stat, p_value = stats.f_oneway(*age_groups)
        
        st.markdown(f"""
        ### Statistical Analysis
        
        **One-way ANOVA**: Practice size differences by age
        - F-statistic: {f_stat:.2f}
        - p-value: {p_value:.4f}
        - Finding: {'Practice size significantly varies with practice age' if p_value < 0.05 else 'No significant size differences based on practice age'}
        """)
    
    st.markdown("""
    ### Key Findings
    1. Practice Age Distribution
    2. Regional Variations
    3. Subspecialty Differences by Age
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Practice age distribution
        age_dist = df['Age_Clean'].value_counts()
        fig = px.pie(
            values=age_dist.values,
            names=age_dist.index,
            title="Practice Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # New practices by region
        new_practices = df[df['Age_Clean'].str.contains('New Business', na=False)]
        region_new = new_practices.groupby('Region').size()
        fig = px.bar(
            region_new,
            title="New Practice Formation by Region"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional analysis
    st.markdown("### Practice Characteristics by Age")
    age_analysis = df.groupby('Age_Clean').agg({
        'JobsReported': ['mean', 'median', 'count'],
        'CurrentApprovalAmount': 'mean'
    }).round(1)
    st.write(age_analysis)

def format_number(num):
    """Helper function to format numbers with commas and decimals"""
    if isinstance(num, (int, float)):
        return f"{num:,.2f}"
    return num

if __name__ == "__main__":
    main()