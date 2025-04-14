import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(layout="wide", page_title="Indian Crime Analysis", page_icon="📊")


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    for col in ['Date Reported', 'Date of Occurrence', 'Date Case Closed']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', format='%d-%m-%Y %H:%M')
            except ValueError:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Try default parsing
    return df


if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

def toggle_theme():
    st.session_state['theme'] = 'dark' if st.session_state['theme'] == 'light' else 'light'

st.sidebar.button("Toggle Theme", on_click=toggle_theme)


if st.session_state['theme'] == 'dark':
    st.markdown(
        """
        <style>
        body {
            background-color: #262730;
            color: white;
        }
        .streamlit-expanderHeader {
            color: white !important;
        }
        .css-1adrfps {
            color: white !important;
        }
        .css-qrbk6 {
            color: white !important;
        }
        .css-qrbk6 a {
            color: white !important;
        }
        div.stRadio > label {
             color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    plotly_template = "plotly_dark"
else:
    plotly_template = "plotly_white"  # Or default

# --- SIDEBAR ---
st.sidebar.header("Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV content
    csv_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = load_data(csv_data)

    # --- ADVANCED FILTERS ---
    st.sidebar.header("Advanced Filters")

    # Multi-select for Cities
    available_cities = sorted(df['City'].unique().tolist())
    selected_cities = st.sidebar.multiselect("Select Cities", available_cities)

    # Multi-select for Crime Domains
    available_crime_domains = sorted(df['Crime Domain'].unique().tolist())
    selected_crime_domains = st.sidebar.multiselect("Select Crime Domains", available_crime_domains)

    # Slider for Victim Age
    min_age = int(df['Victim Age'].min())
    max_age = int(df['Victim Age'].max())
    selected_age_range = st.sidebar.slider("Select Victim Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Date Range for Date of Occurrence
    min_date = df['Date of Occurrence'].min()
    max_date = df['Date of Occurrence'].max()
    selected_date_range = st.sidebar.date_input("Select Date of Occurrence Range",
                                                  value=[min_date, max_date])

    # --- FILTERING LOGIC ---
    df_filtered = df.copy()

    # Filter by Cities
    if selected_cities:
        df_filtered = df_filtered[df_filtered['City'].isin(selected_cities)]

    # Filter by Crime Domains
    if selected_crime_domains:
        df_filtered = df_filtered[df_filtered['Crime Domain'].isin(selected_crime_domains)]

    # Filter by Age Range
    df_filtered = df_filtered[(df_filtered['Victim Age'] >= selected_age_range[0]) & (df_filtered['Victim Age'] <= selected_age_range[1])]

   # Convert selected_date_range to datetime64[ns]
    start_date, end_date = pd.to_datetime(selected_date_range)

    # Apply date filter
    df_filtered = df_filtered[(df_filtered['Date of Occurrence'] >= start_date) & (df_filtered['Date of Occurrence'] <= end_date)]

    # --- DISPLAY FILTERED DATA ---
    if st.sidebar.checkbox("Show Filtered Data"):
        st.sidebar.write(df_filtered)

    # --- MAIN DASHBOARD ---
    st.title("Crime Analysis Dashboard")

    # --- KPIs ---
    total_cases = df_filtered.shape[0]
    cities_involved = len(df_filtered['City'].unique())
    avg_age = df_filtered['Victim Age'].mean()

    kpi1, kpi2, kpi3 = st.columns(3)

    kpi1.metric(label="Total Cases", value=total_cases)
    kpi2.metric(label="Cities Involved", value=cities_involved)
    kpi3.metric(label="Average Victim Age", value=f"{avg_age:.2f}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # --- VISUALIZATIONS ---
    tab1, tab2, tab3, tab4 = st.tabs(["City & Crime", "Victim Analysis", "Case Details", "Comparative Analysis"])

    with tab1:
        st.subheader("City and Crime Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Configurable Chart Selection
            chart_options_city = st.multiselect("Select charts to display for City Analysis:",
                                                ["Crimes by City (Bar)", "Crimes by City (Pie)", "Top Crime Types by City", "Crime Rate by City (Bar)"])

            if "Crimes by City (Bar)" in chart_options_city:
                st.subheader("Crimes by City")
                city_crime_counts = df_filtered['City'].value_counts().reset_index()
                city_crime_counts.columns = ['City', 'Number of Crimes']
                fig_city = px.bar(city_crime_counts, x='City', y='Number of Crimes', color='Number of Crimes', title="Crimes by City", template=plotly_template)
                st.plotly_chart(fig_city, use_container_width=True)

            if "Crimes by City (Pie)" in chart_options_city:
                st.subheader("Crimes by City (Pie Chart)")
                city_crime_counts = df_filtered['City'].value_counts().reset_index()
                city_crime_counts.columns = ['City', 'Number of Crimes']
                fig_city_pie = px.pie(city_crime_counts, names='City', values='Number of Crimes', title='Crimes Distribution Across Cities', template=plotly_template)
                st.plotly_chart(fig_city_pie, use_container_width=True)

            if "Top Crime Types by City" in chart_options_city:
                st.subheader("Top 5 Crime Types by City")
                top_crime_types = df_filtered.groupby(['City', 'Crime Description']).size().reset_index(name='count')
                top_crime_types = top_crime_types.groupby('City').apply(lambda x: x.nlargest(5, 'count')).reset_index(drop=True)
                fig_top_crimes = px.bar(top_crime_types, x='City', y='count', color='Crime Description', title='Top 5 Crime Types by City', template=plotly_template)
                st.plotly_chart(fig_top_crimes, use_container_width=True)
            
            if "Crime Rate by City (Bar)" in chart_options_city:
                st.subheader("Crime Rate by City")
                city_crime_counts = df_filtered['City'].value_counts().reset_index()
                city_crime_counts.columns = ['City', 'Number of Crimes']
                # Assuming a constant population for each city for simplicity
                crime_rate_city = city_crime_counts.copy()
                crime_rate_city['Population'] = 100000  # Placeholder population
                crime_rate_city['Crime Rate'] = (crime_rate_city['Number of Crimes'] / crime_rate_city['Population']) * 100000
                fig_crime_rate_city = px.bar(crime_rate_city, x='City', y='Crime Rate', color='Crime Rate', title='Crime Rate by City (per 100,000 population)', template=plotly_template)
                st.plotly_chart(fig_crime_rate_city, use_container_width=True)

        with col2:
            # Configurable Chart Selection
            chart_options_domain = st.multiselect("Select charts to display for Crime Domain Analysis:",
                                                    ["Crime Domains Distribution (Pie)", "Crime Domains (Bar)", "Crime Domain vs. City (Sunburst)", "Crime Domain Over Time (Line)"])

            if "Crime Domains Distribution (Pie)" in chart_options_domain:
                st.subheader("Crime Domains Distribution")
                crime_domain_counts = df_filtered['Crime Domain'].value_counts().reset_index()
                crime_domain_counts.columns = ['Crime Domain', 'Number of Crimes']
                fig_domain = px.pie(crime_domain_counts, names='Crime Domain', values='Number of Crimes', title="Crime Domain Distribution", template=plotly_template)
                st.plotly_chart(fig_domain, use_container_width=True)

            if "Crime Domains (Bar)" in chart_options_domain:
                st.subheader("Crime Domains (Bar Chart)")
                crime_domain_counts = df_filtered['Crime Domain'].value_counts().reset_index()
                crime_domain_counts.columns = ['Crime Domain', 'Number of Crimes']
                fig_domain_bar = px.bar(crime_domain_counts, x='Crime Domain', y='Number of Crimes', color='Crime Domain', title='Number of Crimes by Domain', template=plotly_template)
                st.plotly_chart(fig_domain_bar, use_container_width=True)

            if "Crime Domain vs. City (Sunburst)" in chart_options_domain:
                st.subheader("Crime Domain vs. City (Sunburst Chart)")
                fig_sunburst = px.sunburst(df_filtered, path=['City', 'Crime Domain'], title='Crime Domain Distribution by City', template=plotly_template)
                st.plotly_chart(fig_sunburst, use_container_width=True)
            
            if "Crime Domain Over Time (Line)" in chart_options_domain:
                st.subheader("Crime Domain Over Time (Line Chart)")
                crime_domain_time = df_filtered.groupby(['Date of Occurrence', 'Crime Domain']).size().reset_index(name='Crimes')
                crime_domain_time['Date of Occurrence'] = pd.to_datetime(crime_domain_time['Date of Occurrence'])
                crime_domain_time = crime_domain_time.sort_values('Date of Occurrence')
                fig_domain_time = px.line(crime_domain_time, x='Date of Occurrence', y='Crimes', color='Crime Domain', title='Crime Domain Trend Over Time', template=plotly_template)
                st.plotly_chart(fig_domain_time, use_container_width=True)

        st.subheader("Crimes Over Time (Line Chart)")
        date_counts = df_filtered.dropna(subset=['Date of Occurrence']).set_index('Date of Occurrence').resample('M').size().reset_index(name='Crimes')
        if not date_counts.empty:
            fig_time = px.line(date_counts, x='Date of Occurrence', y='Crimes', title='Monthly Trend of Crimes', template=plotly_template)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.write("No date-related data to display this chart.")

    with tab2:
        st.subheader("Victim Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Configurable Chart Selection
            chart_options_victim_age = st.multiselect("Select charts to display for Victim Age Analysis:",
                                                    ["Victim Age Distribution (Histogram)", "Victim Age (Box Plot)", "Victim Age vs. Crime Domain (Violin Plot)", "Victim Age Density (KDE)"])

            if "Victim Age Distribution (Histogram)" in chart_options_victim_age:
                st.subheader("Victim Age Distribution (Histogram)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df_filtered['Victim Age'], kde=True, bins=20, color="skyblue", ax=ax)
                ax.set_title("Victim Age Distribution")
                ax.set_xlabel("Age")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

            if "Victim Age (Box Plot)" in chart_options_victim_age:
                st.subheader("Victim Age (Box Plot)")
                fig_age_box = px.box(df_filtered, y='Victim Age', title='Distribution of Victim Ages', template=plotly_template)
                st.plotly_chart(fig_age_box, use_container_width=True)

            if "Victim Age vs. Crime Domain (Violin Plot)" in chart_options_victim_age:
                st.subheader("Victim Age vs. Crime Domain (Violin Plot)")
                fig_age_violin = px.violin(df_filtered, x='Crime Domain', y='Victim Age', color='Crime Domain', title='Victim Age Distribution by Crime Domain', template=plotly_template)
                st.plotly_chart(fig_age_violin, use_container_width=True)
            
            if "Victim Age Density (KDE)" in chart_options_victim_age:
                st.subheader("Victim Age Density (KDE)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(df_filtered['Victim Age'], fill=True, color="skyblue", ax=ax)
                ax.set_title("Victim Age Density")
                ax.set_xlabel("Age")
                ax.set_ylabel("Density")
                st.pyplot(fig)

        with col2:
            # Configurable Chart Selection
            chart_options_victim_gender = st.multiselect("Select charts to display for Victim Gender Analysis:",
                                                        ["Victim Gender Distribution (Bar Chart)", "Victim Gender (Pie Chart)", "Gender vs. Crime Domain (Stacked Bar Chart)", "Gender Distribution Over Time (Line)"])

            if "Victim Gender Distribution (Bar Chart)" in chart_options_victim_gender:
                st.subheader("Victim Gender Distribution (Bar Chart)")
                gender_counts = df_filtered['Victim Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Number of Cases']
                fig_gender = px.bar(gender_counts, x='Gender', y='Number of Cases', color='Gender', title="Distribution of Victims by Gender", template=plotly_template)
                st.plotly_chart(fig_gender, use_container_width=True)

            if "Victim Gender (Pie Chart)" in chart_options_victim_gender:
                st.subheader("Victim Gender (Pie Chart)")
                gender_counts = df_filtered['Victim Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Number of Cases']
                fig_gender_pie = px.pie(gender_counts, names='Gender', values='Number of Cases', title='Proportion of Victims by Gender', template=plotly_template)
                st.plotly_chart(fig_gender_pie, use_container_width=True)

            if "Gender vs. Crime Domain (Stacked Bar Chart)" in chart_options_victim_gender:
                st.subheader("Gender vs. Crime Domain (Stacked Bar Chart)")
                gender_crime = pd.crosstab(df_filtered['Victim Gender'], df_filtered['Crime Domain'])
                fig_gender_crime = px.bar(gender_crime, title='Crime Domains by Gender', labels={'value': 'Number of Cases'}, template=plotly_template)
                st.plotly_chart(fig_gender_crime, use_container_width=True)
            
            if "Gender Distribution Over Time (Line)" in chart_options_victim_gender:
                st.subheader("Gender Distribution Over Time (Line)")
                gender_time = df_filtered.groupby(['Date of Occurrence', 'Victim Gender']).size().reset_index(name='Cases')
                gender_time['Date of Occurrence'] = pd.to_datetime(gender_time['Date of Occurrence'])
                gender_time = gender_time.sort_values('Date of Occurrence')
                fig_gender_time = px.line(gender_time, x='Date of Occurrence', y='Cases', color='Victim Gender', title='Gender Distribution Trend Over Time', template=plotly_template)
                st.plotly_chart(fig_gender_time, use_container_width=True)

        st.subheader("Victim Age vs. Gender (Scatter Plot)")
        fig_age_gender = px.scatter(df_filtered, x='Victim Age', y='Victim Gender', color='Crime Domain', title='Victim Age vs Gender', template=plotly_template)
        st.plotly_chart(fig_age_gender, use_container_width=True)

    with tab3:
        st.subheader("Case Details")
        col1, col2 = st.columns(2)
        with col1:
            # Configurable Chart Selection
            chart_options_weapon = st.multiselect("Select charts to display for Weapon Analysis:",
                                                    ["Weapon Used Distribution (Bar Chart)", "Case Closed Status (Pie Chart)", "Police Deployed Distribution", "Weapon Used vs. Crime Domain (Stacked Bar)"])

            if "Weapon Used Distribution (Bar Chart)" in chart_options_weapon:
                st.subheader("Weapon Used Distribution (Bar Chart)")
                weapon_counts = df_filtered['Weapon Used'].value_counts().reset_index()
                weapon_counts.columns = ['Weapon', 'Number of Crimes']
                fig_weapon = px.bar(weapon_counts, x='Weapon', y='Number of Crimes', color='Weapon', title="Distribution of Weapons Used", template=plotly_template)
                st.plotly_chart(fig_weapon, use_container_width=True)

            if "Case Closed Status (Pie Chart)" in chart_options_weapon:
                st.subheader("Case Closed Status (Pie Chart)")
                case_closed_counts = df_filtered['Case Closed'].value_counts().reset_index()
                case_closed_counts.columns = ['Case Closed', 'Number of Cases']
                fig_case_closed = px.pie(case_closed_counts, names='Case Closed', values='Number of Cases', title='Proportion of Cases Closed', template=plotly_template)
                st.plotly_chart(fig_case_closed, use_container_width=True)

            if "Police Deployed Distribution" in chart_options_weapon:
                st.subheader("Police Deployed Distribution")
                fig_police_hist = px.histogram(df_filtered, x='Police Deployed', title='Distribution of Police Deployed', template=plotly_template)
                st.plotly_chart(fig_police_hist, use_container_width=True)
            
            if "Weapon Used vs. Crime Domain (Stacked Bar)" in chart_options_weapon:
                st.subheader("Weapon Used vs. Crime Domain (Stacked Bar)")
                weapon_crime = pd.crosstab(df_filtered['Weapon Used'], df_filtered['Crime Domain'])
                fig_weapon_crime = px.bar(weapon_crime, title='Crime Domains by Weapon Used', labels={'value': 'Number of Cases'}, template=plotly_template)
                st.plotly_chart(fig_weapon_crime, use_container_width=True)

        with col2:
            # Configurable Chart Selection
            chart_options_police = st.multiselect("Select charts to display for Police Analysis:",
                                                    ["Police Deployed vs Case Closed (Stacked Bar Chart)", "Police Deployed (Box Plot)", "Police Deployed Over Time (Line)"])

            if "Police Deployed vs Case Closed (Stacked Bar Chart)" in chart_options_police:
                st.subheader("Police Deployed vs Case Closed (Stacked Bar Chart)")
                cross_tab = pd.crosstab(df_filtered['Police Deployed'], df_filtered['Case Closed'])
                fig, ax = plt.subplots()
                cross_tab.plot(kind='bar', stacked=False, ax=ax, colormap='viridis')
                plt.title('Police Deployed vs Case Closed')
                plt.xlabel('Police Deployed')
                plt.ylabel('Number of Cases')
                plt.xticks(rotation=0)
                st.pyplot(fig)

            if "Police Deployed (Box Plot)" in chart_options_police:
                st.subheader("Police Deployed (Box Plot)")
                fig_police_box = px.box(df_filtered, y='Police Deployed', title='Distribution of Police Deployed', template=plotly_template)
                st.plotly_chart(fig_police_box, use_container_width=True)
            
            if "Police Deployed Over Time (Line)" in chart_options_police:
                st.subheader("Police Deployed Over Time (Line)")
                police_time = df_filtered.groupby(['Date of Occurrence', 'Police Deployed']).size().reset_index(name='Cases')
                police_time['Date of Occurrence'] = pd.to_datetime(police_time['Date of Occurrence'])
                police_time = police_time.sort_values('Date of Occurrence')
                fig_police_time = px.line(police_time, x='Date of Occurrence', y='Cases', color='Police Deployed', title='Police Deployment Trend Over Time', template=plotly_template)
                st.plotly_chart(fig_police_time, use_container_width=True)

        st.subheader("Case Closed Over Time (Line Chart)")
        closed_over_time = df_filtered.dropna(subset=['Date Case Closed']).set_index('Date Case Closed').sort_index()
        if not closed_over_time.empty:
            monthly_cases = closed_over_time.resample('M').size()
            monthly_cases = monthly_cases.reset_index(name='Cases Closed')
            fig_time = px.line(monthly_cases, x='Date Case Closed', y='Cases Closed', title='Monthly Trend of Closed Cases', template=plotly_template)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.write("No date-related data to display this chart.")

    with tab4:
        st.subheader("Comparative Analysis")

        # Configurable Chart Selection
        chart_options_comparative = st.multiselect("Select charts to display for Comparative Analysis:",
                                                    ["Crime Domain vs Victim Age (Box Plot)", "Victim Age vs. Police Deployed (Scatter Chart)",
                                                     "Tree Chart of Crime Domains", "Parallel Categories Diagram", "Crime Domain vs Weapon Used (Heatmap)", "City vs. Case Closed (Bar)"])

        if "Crime Domain vs Victim Age (Box Plot)" in chart_options_comparative:
            st.subheader("Crime Domain vs Victim Age (Box Plot)")
            fig_box = px.box(df_filtered, x='Crime Domain', y='Victim Age', color='Crime Domain', title='Crime Domain vs Victim Age', template=plotly_template)
            st.plotly_chart(fig_box, use_container_width=True)

        if "Victim Age vs. Police Deployed (Scatter Chart)" in chart_options_comparative:
            st.subheader("Scatter Chart: Victim Age vs. Police Deployed")
            fig_scatter = px.scatter(df_filtered, x='Victim Age', y='Police Deployed', color='Crime Domain',
                                     hover_data=['City'], title='Victim Age vs Police Deployed', template=plotly_template)
            st.plotly_chart(fig_scatter, use_container_width=True)

        if "Tree Chart of Crime Domains" in chart_options_comparative:
            st.subheader("Tree Chart of Crime Domains")
            fig_tree = px.treemap(df_filtered, path=['City', 'Crime Domain'], title='Tree Chart of Crime Domains by City', template=plotly_template)
            st.plotly_chart(fig_tree, use_container_width=True)

        if "Parallel Categories Diagram" in chart_options_comparative:
            st.subheader("Parallel Categories Diagram: Crime Analysis")
            fig_parallel = px.parallel_categories(df_filtered,
                                                 dimensions=['City', 'Crime Domain', 'Victim Gender'],
                                                 title='Parallel Categories Diagram of Crime Factors', template=plotly_template)
            st.plotly_chart(fig_parallel, use_container_width=True)
        
        if "Crime Domain vs Weapon Used (Heatmap)" in chart_options_comparative:
            st.subheader("Crime Domain vs Weapon Used (Heatmap)")
            cross_tab = pd.crosstab(df_filtered['Crime Domain'], df_filtered['Weapon Used'])
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt='d', ax=ax)
            plt.title('Heatmap of Crime Domain vs Weapon Used')
            plt.xlabel('Weapon Used')
            plt.ylabel('Crime Domain')
            st.pyplot(fig)
        
        if "City vs. Case Closed (Bar)" in chart_options_comparative:
            st.subheader("City vs. Case Closed (Bar)")
            city_case_closed = pd.crosstab(df_filtered['City'], df_filtered['Case Closed'])
            fig_city_case_closed = px.bar(city_case_closed, title='Case Closed Status by City', labels={'value': 'Number of Cases'}, template=plotly_template)
            st.plotly_chart(fig_city_case_closed, use_container_width=True)
else:
    st.warning("Upload data to continue!")
