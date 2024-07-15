#%%
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import numpy as np

#%% ANALYSIS OVERALL
df = pd.read_excel('/Users/vidhyasainath/Desktop/khooj/CIRCULATE/AKHIRI_BAR_FINAL/AKHIRI_1555_FINAL_CIRCULATE.xlsx')
#data = df[['PATIENT', 'DFS_months', 'DFS_Event', 'ctDNA_MRD', 'group', 'Age', 'Gender', 'ECOG', 'pT', 'pN', 'MSI', 'RAS', 'BRAF_V600E','ACT']]
data = df[['PATIENT', 'DFS_months', 'DFS_Event', 'ctDNA_MRD', 'group', 'Age', 'Gender', 'pT', 'pN']]
data['age_binary'] = data['Age'].apply(lambda x: 'above_70' if x > 70 else 'less_70')
data['group'] = data['group'].apply(lambda x: 1 if x == "High" else 0)
table = pd.pivot_table(data, values='PATIENT', index=['ctDNA_MRD','group'], aggfunc='count')
print(table)
del data['Age']
del data['PATIENT'] # REMOVING PATIENT AS WE DONT WANT TO INCLUDE THEM IN THE MULTIVARIATE ANALYSIS
data_cleaned = data.dropna() # REMOVING ROWS THAT DONT HAVE VALUES
#data_encoded = pd.get_dummies(data_cleaned, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group', 'RAS', 'MSI','pN','pT','ECOG','ACT','BRAF_V600E'], drop_first=True)
data_encoded = pd.get_dummies(data_cleaned, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group','pN','pT'], drop_first=True)
data_encoded.rename(columns={'group_1': 'DLRiskScore_High'},inplace=True)
data_encoded.rename(columns={'age_binary_less_70': 'Age_less_70'},inplace=True)
cph = CoxPHFitter()
cph.fit(data_encoded, duration_col='DFS_months', event_col='DFS_Event', show_progress=True, fit_options= {'step_size':0.5})
cph.print_summary()
cph.plot()
#%% AKIRI FOREST PLOT WITH TABLE

def create_forest_plot_with_table(model, ax):
    summary = model.summary
    
    # Determine significance
    significant_mask = summary.apply(lambda x: np.isfinite(x['exp(coef) lower 95%']) and np.isfinite(x['exp(coef) upper 95%']) and (x['exp(coef) lower 95%'] > 1 or x['exp(coef) upper 95%'] < 1), axis=1)
    
    # Sort by hazard ratio (HR)
    summary = summary.sort_values(by='exp(coef)', ascending=True)
    summary['y'] = range(len(summary), 0, -1)
    
    # Add asterisk for significant variables
    labels = summary.index.to_series()
    labels[significant_mask] = labels[significant_mask] + '*'
    
    # Regular points with finite confidence intervals
    finite_mask = summary.apply(lambda x: np.isfinite(x['exp(coef) lower 95%']) and np.isfinite(x['exp(coef) upper 95%']), axis=1)
    regular_data = summary[finite_mask]
    
    ax.errorbar(regular_data['exp(coef)'], regular_data['y'], 
                xerr=[regular_data['exp(coef)'] - regular_data['exp(coef) lower 95%'],
                      regular_data['exp(coef) upper 95%'] - regular_data['exp(coef)']],
                fmt='o', color='black', ecolor='black', capsize=3, label='Significant if not crossing 1')

    # Points where CI is NaN or Inf
    infinite_mask = ~finite_mask
    infinite_data = summary[infinite_mask]
    
    ax.scatter(infinite_data['exp(coef)'].apply(lambda x: 0 if not np.isfinite(x) else x), infinite_data['y'], 
               marker='^', color='red')
    
    ax.set_yticks(summary['y'])
    ax.set_yticklabels(labels)
    ax.set_xlabel('Hazard Ratio')
    ax.set_xscale('log')  # Set X-axis to log scale
    ax.axvline(x=1, color='grey', linestyle='--')

    # Ensure limits are appropriate for log scale
    min_xlim = 0.1 if regular_data['exp(coef)'].min() <= 0 else regular_data['exp(coef)'].min() * 0.9
    max_xlim = regular_data['exp(coef) upper 95%'].max() * 1.1 if not regular_data.empty else 2
    ax.set_xlim(left=min_xlim, right=max_xlim)
    
    # Custom tick marks for better visualization
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.legend()
    
    return summary

def save_table(summary):
    # Create the table with regression coefficient, p-value, and CI
    table_data = summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']]
    table_data.columns = ['HR', 'p-value', 'Lower 95% CI', 'Upper 95% CI']
    
    # Save the table to a SVG format
    fig, ax_table = plt.subplots(figsize=(10, len(table_data) * 0.5))
    ax_table.axis('tight')
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    plt.savefig('Forest_plot_table_NegativeOnly.svg')
    plt.show()

#%% MRD NEGATIVE ONLY OVERALL
negative_data = data_cleaned[data_cleaned['ctDNA_MRD'] == "NEGATIVE"]
negative_data_encoded = pd.get_dummies(negative_data, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group','pN','pT'], drop_first=True)
negative_data_encoded.rename(columns={'group_1': 'DLRiskScore_High'},inplace=True)
negative_data_encoded.rename(columns={'age_binary_less_70': 'Age_less_70'},inplace=True)
cph_negative = CoxPHFitter()
cph_negative.fit(negative_data_encoded, duration_col='DFS_months', event_col='DFS_Event', show_progress=True, fit_options= {'step_size':0.5})
cph_negative.plot()
fig, ax = plt.subplots(figsize=(10, 5))
sorted_summary = create_forest_plot_with_table(cph_negative, ax)
plt.savefig('Forest_plot_NegativeOnly.svg')
plt.show()
save_table(sorted_summary)
#%% MRD POSITIVE ONLY OVERALL
positive_data = data_cleaned[data_cleaned['ctDNA_MRD'] == "POSITIVE"]
positive_data_encoded = pd.get_dummies(positive_data, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group','pN','pT'], drop_first=True)
positive_data_encoded.rename(columns={'group_1': 'DLRiskScore_High'},inplace=True)
positive_data_encoded.rename(columns={'age_binary_less_70': 'Age_less_70'},inplace=True)
cph_positive = CoxPHFitter()
cph_positive.fit(positive_data_encoded, duration_col='DFS_months', event_col='DFS_Event', show_progress=True, fit_options= {'step_size':0.5})
cph_positive.plot()
fig, ax = plt.subplots(figsize=(10, 5))
sorted_summary = create_forest_plot_with_table(cph_positive, ax)
plt.savefig('Forest_plot_PositiveOnly.svg')
plt.show()
save_table(sorted_summary)
#%%
dataACT = df[['PATIENT', 'DFS_months', 'DFS_Event', 'ctDNA_MRD', 'group', 'Age', 'Gender', 'pT', 'pN', 'ACT']]
dataACT['age_binary'] = dataACT['Age'].apply(lambda x: 'above_70' if x > 70 else 'less_70')
dataACT['group'] = dataACT['group'].apply(lambda x: 1 if x == "High" else 0)
#%%
del dataACT['Age']
del dataACT['PATIENT'] # REMOVING PATIENT AS WE DONT WANT TO INCLUDE THEM IN THE MULTIVARIATE ANALYSIS

data_cleaned = dataACT.dropna() # REMOVING ROWS THAT DONT HAVE VALUES
#%%
negative_data = data_cleaned[data_cleaned['ctDNA_MRD'] == "NEGATIVE"]
#%%
negative_with_ACT = negative_data[negative_data['ACT'] == True]
negative_without_ACT = negative_data[negative_data['ACT'] == False]
# %%
#data_encoded = pd.get_dummies(data_cleaned, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group', 'RAS', 'MSI','pN','pT','ECOG','ACT','BRAF_V600E'], drop_first=True)
data_encoded_with_ACT = pd.get_dummies(negative_with_ACT, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group','pN','pT'], drop_first=True)
del data_encoded_with_ACT['ACT']
#%%
data_encoded_with_ACT.rename(columns={'group_1': 'DLRiskScore_High'},inplace=True)
data_encoded_with_ACT.rename(columns={'age_binary_less_70': 'Age_less_70'},inplace=True)
#%%
cph_with_ACT = CoxPHFitter()
cph_with_ACT.fit(data_encoded_with_ACT, duration_col='DFS_months', event_col='DFS_Event', show_progress=True, fit_options= {'step_size':0.5})

#%%
cph_with_ACT.print_summary()
cph_with_ACT.plot()
#%%
# Creating the forest plot
fig, ax = plt.subplots(figsize=(10, 5))
sorted_summary = create_forest_plot_with_table(cph_with_ACT, ax)
plt.savefig('Forest_plot_Negative_with_ACT.svg')
plt.show()

# Saving the table separately with the sorted summary
save_table(sorted_summary)
#%%
data_encoded_without_ACT = pd.get_dummies(negative_without_ACT, columns=['age_binary', 'Gender', 'ctDNA_MRD', 'group','pN','pT'], drop_first=True)
del data_encoded_without_ACT['ACT']
#%%
data_encoded_without_ACT.rename(columns={'group_1': 'DLRiskScore_High'},inplace=True)
data_encoded_without_ACT.rename(columns={'age_binary_less_70': 'Age_less_70'},inplace=True)
#%%
cph_without_ACT = CoxPHFitter()
cph_without_ACT.fit(data_encoded_without_ACT, duration_col='DFS_months', event_col='DFS_Event', show_progress=True, fit_options= {'step_size':0.5})

#%%
cph_without_ACT.print_summary()
cph_without_ACT.plot()
#%%
# Creating the forest plot
fig, ax = plt.subplots(figsize=(10, 5))
sorted_summary = create_forest_plot_with_table(cph_with_ACT, ax)
plt.savefig('Forest_plot_Negative_with_ACT.svg')
plt.show()
