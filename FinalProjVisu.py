import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv(r'data.csv')

st.title("My Streamlit Dashboard")

# Calculate average severity per state
average_severity_data = df.groupby('State')['Severity'].mean().reset_index()
average_severity_data.columns = ['State', 'average_severity']
# Round the average severity values
average_severity_data['average_severity'] = average_severity_data['average_severity'].round().astype(int)

# Box plot
st.subheader("Box Plot")
bar_plot = sns.boxplot(x='Severity', y=df['Temperature(F)'], data=df)
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Severity vs Temperature')
st.pyplot(bar_plot.figure)

# Correlation matrix
st.subheader("Correlation Matrix")
selected_columns = ['Severity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                    'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
new_df = df[selected_columns].copy()
# Compute correlation matrix
correlation_matrix = new_df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
# Create a heatmap plot of the correlation matrix
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.xlabel('Variables')
plt.ylabel('Variables')
plt.title('Correlation Matrix')
st.pyplot(heatmap.figure)


# Bar plot
top_10_conditions = df['Weather_Condition'].value_counts().nlargest(10).index
# Filter the DataFrame to include only the top 10 weather conditions
filtered_df = df[df['Weather_Condition'].isin(top_10_conditions)]
# Group the filtered data by weather condition and severity, and count the occurrences
grouped_data = filtered_df.groupby(['Weather_Condition', 'Severity']).size().unstack()

st.subheader("Bar Plot")
sns.set_palette("Set2")
bar_plot = grouped_data.plot(kind='bar', stacked=True)
plt.xlabel('Weather Condition')
plt.ylabel('Count')
plt.title('Top 10 Weather Condition Severity Distribution')
st.pyplot(bar_plot.figure)


# Map
st.subheader("USA Map - Average Severity")
fig = px.choropleth(average_severity_data,
                    locations='State',
                    locationmode="USA-states",
                    color='average_severity',
                    color_continuous_scale='Reds',
                    scope="usa",
                    labels={'average_severity': 'Average Severity'}
                    )
fig.update_layout(title_text='Average Severity by State')
st.plotly_chart(fig)


st.subheader("Data")
st.write(df)