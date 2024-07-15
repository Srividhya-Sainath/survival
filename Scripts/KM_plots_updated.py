#%%
import pandas as pd
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
import numpy as np
#%%
df = pd.read_excel('/Users/vidhyasainath/Desktop/khooj/CIRCULATE/AKHIRI_BAR_FINAL/AKHIRI_1555_FINAL_CIRCULATE.xlsx')
#%%
data = df[['PATIENT', 'DFS_months', 'DFS_Event', 'ctDNA_MRD', 'group']]
data['group'] = data['group'].apply(lambda x: 1 if x == "High" else 0)
data.rename(columns={'group': 'SCORE_BINARY'},inplace=True)
#%%
data.rename(columns={'DFS_months': 'DFS_MONTHS'},inplace=True)
data.rename(columns={'DFS_Event': 'DFS_E'},inplace=True)
data.rename(columns={'ctDNA_MRD': 'MRD-STATUS'},inplace=True)
#%% KM PLOTS OVERALL
kmf_overall_lower_than_median = KaplanMeierFitter()
kmf_overall_higher_equal_median = KaplanMeierFitter()

overall_lower_than_median = data[data['SCORE_BINARY'] == 0]
overall_higher_equal_median = data[data['SCORE_BINARY'] == 1]

kmf_overall_lower_than_median.fit(overall_lower_than_median['DFS_MONTHS'], event_observed=overall_lower_than_median['DFS_E'], label=f'Low Risk')
kmf_overall_higher_equal_median.fit(overall_higher_equal_median['DFS_MONTHS'], event_observed=overall_higher_equal_median['DFS_E'], label=f'High Risk')

results = logrank_test(overall_lower_than_median['DFS_MONTHS'], overall_higher_equal_median['DFS_MONTHS'], overall_lower_than_median['DFS_E'], overall_higher_equal_median['DFS_E'])
logrank_p_value = results.p_value

coxph_model = CoxPHFitter()
coxph_model.fit(data, duration_col='DFS_MONTHS', event_col='DFS_E', formula="SCORE_BINARY")
print(coxph_model.summary)
print(logrank_p_value)

plt.figure(figsize=(10, 6))
kmf_overall_lower_than_median.plot(ci_show=True)
kmf_overall_higher_equal_median.plot(ci_show=True)
plt.xlabel('Time (Months)')

add_at_risk_counts(kmf_overall_higher_equal_median,kmf_overall_lower_than_median)
plt.tight_layout()

plt.text(0, 0.55, f'High Risk: HR 2.57 [2.082-3.182], p < 0.005', fontsize=10)
plt.text(0,0.5,f'Logrank Test p < 0.001', fontsize=10)

#plt.title('Kaplan-Meier Disease-free Survival Curves (Overall - 4W)')
plt.ylabel('Disease-free Survival Probability')
plt.legend()

#plt.savefig("KM_plot_overall_uncorrected.svg", format="svg")
plt.show()

#%% MRD-POSITIVE DL BASED HIGH RISK VERSUS DL BASED LOW RISK CORRECTED HR ##

positive_data = data[data['MRD-STATUS'] == "POSITIVE"]

positive_lower_than_median = positive_data[positive_data['SCORE_BINARY'] == 0]
positive_higher_equal_median = positive_data[positive_data['SCORE_BINARY'] == 1]
#positive_encoded = pd.get_dummies(positive_data, columns=['sex', 'cohort', 'MRD-STATUS', 'SCORE_BINARY'], drop_first=True)

kmf_positive_lower_than_median = KaplanMeierFitter()
kmf_positive_higher_equal_median = KaplanMeierFitter()

kmf_positive_lower_than_median.fit(positive_lower_than_median['DFS_MONTHS'], event_observed=positive_lower_than_median['DFS_E'], label=f'Low Risk')
kmf_positive_higher_equal_median.fit(positive_higher_equal_median['DFS_MONTHS'], event_observed=positive_higher_equal_median['DFS_E'], label=f'High Risk')

positive_coxph_model = CoxPHFitter()
positive_coxph_model.fit(positive_data, duration_col='DFS_MONTHS', event_col='DFS_E', formula="SCORE_BINARY")

results = logrank_test(positive_lower_than_median['DFS_MONTHS'], positive_higher_equal_median['DFS_MONTHS'], positive_lower_than_median['DFS_E'], positive_higher_equal_median['DFS_E'])
logrank_p_value = results.p_value

print(positive_coxph_model.summary)
print(logrank_p_value)

plt.figure(figsize=(10, 6))
kmf_positive_lower_than_median.plot(ci_show=True)
kmf_positive_higher_equal_median.plot(ci_show=True)
plt.xlabel('Time (Months)')

add_at_risk_counts(kmf_positive_higher_equal_median,kmf_positive_lower_than_median)
plt.tight_layout()

plt.text(0, 0.13, f'High Risk: HR 1.57 [1.175-2.116], p=0.002', fontsize=10)
plt.text(0,0.07,f'Logrank Test p = 0.002', fontsize=10)

#plt.title('Kaplan-Meier Disease-free Survival Curves (Overall - 4W)')
plt.ylabel('Disease-free Survival Probability')
plt.legend()
plt.savefig("KM_plot_MRDPositive_uncorrected.svg", format="svg")
plt.show()

#%% MRD-NEGATIVE DL BASED HIGH RISK VERSUS DL BASED LOW RISK CORRECTED HR ##

negative_data = data[data['MRD-STATUS'] == "NEGATIVE"]

negative_lower_than_median = negative_data[negative_data['SCORE_BINARY'] == 0]
negative_higher_equal_median = negative_data[negative_data['SCORE_BINARY'] == 1]

kmf_negative_lower_than_median = KaplanMeierFitter()
kmf_negative_higher_equal_median = KaplanMeierFitter()

#negative_encoded = pd.get_dummies(negative_data, columns=['sex', 'cohort', 'MRD-STATUS', 'SCORE_BINARY'], drop_first=True)

negative_coxph_model = CoxPHFitter()
negative_coxph_model.fit(negative_data, duration_col='DFS_MONTHS', event_col='DFS_E',formula="SCORE_BINARY")

kmf_negative_lower_than_median.fit(negative_lower_than_median['DFS_MONTHS'], event_observed=negative_lower_than_median['DFS_E'], label=f'Low Risk')
kmf_negative_higher_equal_median.fit(negative_higher_equal_median['DFS_MONTHS'], event_observed=negative_higher_equal_median['DFS_E'], label=f'High Risk')

results = logrank_test(negative_lower_than_median['DFS_MONTHS'], negative_higher_equal_median['DFS_MONTHS'], negative_lower_than_median['DFS_E'], negative_higher_equal_median['DFS_E'])
logrank_p_value = results.p_value

print(negative_coxph_model.summary)
print(logrank_p_value)

plt.figure(figsize=(10, 6))
kmf_negative_lower_than_median.plot(ci_show=True)
kmf_negative_higher_equal_median.plot(ci_show=True)
plt.xlabel('Time (Months)')

add_at_risk_counts(kmf_negative_higher_equal_median,kmf_negative_lower_than_median)
plt.tight_layout()

plt.text(0, 0.70, f'High Risk: HR 2.36 [1.733-3.234], p<0.001', fontsize=10)
plt.text(0,0.68,f'Logrank Test p<0.001', fontsize=10)

#plt.title('Kaplan-Meier Disease-free Survival Curves (Overall - 4W)')
plt.ylabel('Disease-free Survival Probability')
plt.legend()
plt.savefig("KM_plot_MRDNegative_uncorrected.svg", format="svg")
plt.show()
#%% MRD NEGATIVE GROUPED FOR ACT ALONE
data2 = df[['PATIENT', 'DFS_months', 'DFS_Event', 'ctDNA_MRD', 'group', 'ACT']]
data2['group'] = data2['group'].apply(lambda x: 1 if x == "High" else 0)
data2.rename(columns={'group': 'SCORE_BINARY'},inplace=True)
data2.rename(columns={'DFS_months': 'DFS_MONTHS'},inplace=True)
data2.rename(columns={'DFS_Event': 'DFS_E'},inplace=True)
data2.rename(columns={'ctDNA_MRD': 'MRD-STATUS'},inplace=True)
# %%
negative_data2 = data2[data2['MRD-STATUS'] == "NEGATIVE"]
negative_lower_than_median2 = negative_data2[negative_data2['SCORE_BINARY'] == 0]
negative_higher_equal_median2 = negative_data2[negative_data2['SCORE_BINARY'] == 1]

negative_lower_than_median2_with_ACT = negative_lower_than_median2[negative_lower_than_median2['ACT'] == True]
negative_lower_than_median2_without_ACT = negative_lower_than_median2[negative_lower_than_median2['ACT'] == False]

negative_higher_equal_median2_with_ACT = negative_higher_equal_median2[negative_higher_equal_median2['ACT'] == True]
negative_higher_equal_median2_without_ACT = negative_higher_equal_median2[negative_higher_equal_median2['ACT'] == False]

kmf_negative_lower_than_median2_with_ACT = KaplanMeierFitter()
kmf_negative_lower_than_median2_without_ACT = KaplanMeierFitter()

kmf_negative_higher_equal_median2_with_ACT = KaplanMeierFitter()
kmf_negative_higher_equal_median2_without_ACT = KaplanMeierFitter()

negative_lower_than_medium_coxph_model2 = CoxPHFitter()
negative_lower_than_medium_coxph_model2.fit(negative_lower_than_median2, duration_col='DFS_MONTHS', event_col='DFS_E',formula="ACT",show_progress=True, fit_options= {'step_size':0.9})

kmf_negative_lower_than_median2_with_ACT.fit(negative_lower_than_median2_with_ACT['DFS_MONTHS'], event_observed=negative_lower_than_median2_with_ACT['DFS_E'], label=f'Low Risk With ACT')
kmf_negative_lower_than_median2_without_ACT.fit(negative_lower_than_median2_without_ACT['DFS_MONTHS'], event_observed=negative_lower_than_median2_without_ACT['DFS_E'], label=f'Low Risk Without ACT')

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
kmf_negative_lower_than_median2_with_ACT.plot(ax=ax,ci_show=False)
kmf_negative_lower_than_median2_without_ACT.plot(ax=ax,ci_show=False)

plt.text(0, 0.85, f'High Risk: HR 1.14 [0.80-1.63], p=0.45', fontsize=10)
plt.text(0,0.83,f'Logrank Test p=0.45', fontsize=10)

plt.xlabel('Time (Months)')
add_at_risk_counts(kmf_negative_lower_than_median2_with_ACT,kmf_negative_lower_than_median2_without_ACT)
plt.tight_layout()
plt.savefig("KM_plot_MRDNegative_LowRisk_ACT.svg", format="svg")

results = logrank_test(negative_lower_than_median2_with_ACT['DFS_MONTHS'], negative_lower_than_median2_without_ACT['DFS_MONTHS'], negative_lower_than_median2_with_ACT['DFS_E'], negative_lower_than_median2_without_ACT['DFS_E'])
logrank_p_value = results.p_value
print(logrank_p_value)

negative_higher_equal_medium_coxph_model2 = CoxPHFitter()
negative_higher_equal_medium_coxph_model2.fit(negative_higher_equal_median2, duration_col='DFS_MONTHS', event_col='DFS_E',formula="ACT",show_progress=True, fit_options= {'step_size':0.9})

kmf_negative_higher_equal_median2_with_ACT.fit(negative_higher_equal_median2_with_ACT['DFS_MONTHS'], event_observed=negative_higher_equal_median2_with_ACT['DFS_E'], label=f'High Risk With ACT')
kmf_negative_higher_equal_median2_without_ACT.fit(negative_higher_equal_median2_without_ACT['DFS_MONTHS'], event_observed=negative_higher_equal_median2_without_ACT['DFS_E'], label=f'High Risk Without ACT')
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
kmf_negative_higher_equal_median2_with_ACT.plot(ax=ax,ci_show=False)
kmf_negative_higher_equal_median2_without_ACT.plot(ax=ax,ci_show=False)

plt.text(0, 0.72, f'High Risk: HR 0.48 [0.27-0.86], p=0.01', fontsize=10)
plt.text(0,0.70,f'Logrank Test p=0.01', fontsize=10)

plt.xlabel('Time (Months)')
add_at_risk_counts(kmf_negative_higher_equal_median2_with_ACT,kmf_negative_higher_equal_median2_without_ACT)
plt.tight_layout()
plt.savefig("KM_plot_MRDNegative_HighRisk_ACT.svg", format="svg")
results = logrank_test(negative_higher_equal_median2_with_ACT['DFS_MONTHS'], negative_higher_equal_median2_without_ACT['DFS_MONTHS'], negative_higher_equal_median2_with_ACT['DFS_E'], negative_higher_equal_median2_without_ACT['DFS_E'])
logrank_p_value = results.p_value
print(logrank_p_value)

kmf_negative_higher_equal_median2_with_ACT.fit(negative_higher_equal_median2_with_ACT['DFS_MONTHS'], event_observed=negative_higher_equal_median2_with_ACT['DFS_E'], label=f'High Risk with ACT')
kmf_negative_higher_equal_median2_without_ACT.fit(negative_higher_equal_median2_without_ACT['DFS_MONTHS'], event_observed=negative_higher_equal_median2_without_ACT['DFS_E'], label=f'High Risk without ACT')

results = logrank_test(negative_higher_equal_median2_with_ACT['DFS_MONTHS'], negative_higher_equal_median2_without_ACT['DFS_MONTHS'], negative_higher_equal_median2_with_ACT['DFS_E'], negative_higher_equal_median2_without_ACT['DFS_E'])
logrank_p_value = results.p_value

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
kmf_negative_higher_equal_median2_with_ACT.plot(ax=ax,ci_show=True)
kmf_negative_higher_equal_median2_without_ACT.plot(ax=ax,ci_show=True)
plt.xlabel('Time (Months)')
add_at_risk_counts(kmf_negative_higher_equal_median2_with_ACT,kmf_negative_higher_equal_median2_without_ACT)
plt.tight_layout()

plt.legend()

plt.savefig("KM_plot_MRD_Negative_ACT_uncorrected.svg", format="svg")
plt.show()
#%% MRD POSITIVE GROUPED FOR ACT ALONE
positive_data2 = data2[data2['MRD-STATUS'] == "POSITIVE"]
#%%
positive_lower_than_median2 = positive_data2[positive_data2['SCORE_BINARY'] == 0]
positive_higher_equal_median2 = positive_data2[positive_data2['SCORE_BINARY'] == 1]
#%%
positive_lower_than_median2_with_ACT = positive_lower_than_median2[positive_lower_than_median2['ACT'] == True]
positive_lower_than_median2_without_ACT = positive_lower_than_median2[positive_lower_than_median2['ACT'] == False]
positive_higher_equal_median2_with_ACT = positive_higher_equal_median2[positive_higher_equal_median2['ACT'] == True]
positive_higher_equal_median2_without_ACT = positive_higher_equal_median2[positive_higher_equal_median2['ACT'] == False]
#%%
kmf_positive_lower_than_median2_with_ACT = KaplanMeierFitter()
kmf_positive_lower_than_median2_without_ACT = KaplanMeierFitter()

kmf_positive_higher_equal_median2_with_ACT = KaplanMeierFitter()
kmf_positive_higher_equal_median2_without_ACT = KaplanMeierFitter()

#%%
positive_lower_than_medium_coxph_model2 = CoxPHFitter()
positive_lower_than_medium_coxph_model2.fit(positive_lower_than_median2, duration_col='DFS_MONTHS', event_col='DFS_E',formula="ACT",show_progress=True, fit_options= {'step_size':0.9})

kmf_positive_lower_than_median2_with_ACT.fit(positive_lower_than_median2_with_ACT['DFS_MONTHS'], event_observed=positive_lower_than_median2_with_ACT['DFS_E'], label=f'Low Risk With ACT')
kmf_positive_lower_than_median2_without_ACT.fit(positive_lower_than_median2_without_ACT['DFS_MONTHS'], event_observed=positive_lower_than_median2_without_ACT['DFS_E'], label=f'Low Risk Without ACT')

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
kmf_positive_lower_than_median2_with_ACT.plot(ax=ax,ci_show=False)
kmf_positive_lower_than_median2_without_ACT.plot(ax=ax,ci_show=False)

plt.text(0, 0.2, f'High Risk: HR 0.20 [0.14-0.30], p<0.005', fontsize=10)
plt.text(0,0.15,f'Logrank Test p<0.001', fontsize=10)

plt.xlabel('Time (Months)')
add_at_risk_counts(kmf_positive_lower_than_median2_with_ACT,kmf_positive_lower_than_median2_without_ACT)
plt.tight_layout()
plt.savefig("KM_plot_MRDPositive_LowRisk_ACT.svg", format="svg")

results = logrank_test(positive_lower_than_median2_with_ACT['DFS_MONTHS'], positive_lower_than_median2_without_ACT['DFS_MONTHS'], positive_lower_than_median2_with_ACT['DFS_E'], positive_lower_than_median2_without_ACT['DFS_E'])
logrank_p_value = results.p_value
print(logrank_p_value)
#%%
positive_higher_equal_medium_coxph_model2 = CoxPHFitter()
positive_higher_equal_medium_coxph_model2.fit(positive_higher_equal_median2, duration_col='DFS_MONTHS', event_col='DFS_E',formula="ACT",show_progress=True, fit_options= {'step_size':0.9})

kmf_positive_higher_equal_median2_with_ACT.fit(positive_higher_equal_median2_with_ACT['DFS_MONTHS'], event_observed=positive_higher_equal_median2_with_ACT['DFS_E'], label=f'High Risk With ACT')
kmf_positive_higher_equal_median2_without_ACT.fit(positive_higher_equal_median2_without_ACT['DFS_MONTHS'], event_observed=positive_higher_equal_median2_without_ACT['DFS_E'], label=f'High Risk Without ACT')
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)
kmf_positive_higher_equal_median2_with_ACT.plot(ax=ax,ci_show=False)
kmf_positive_higher_equal_median2_without_ACT.plot(ax=ax,ci_show=False)

plt.text(0, 0.2, f'High Risk: HR 0.25 [0.16-0.42], p<0.005', fontsize=10)
plt.text(0,0.15,f'Logrank Test p<0.001', fontsize=10)

plt.xlabel('Time (Months)')
add_at_risk_counts(kmf_positive_higher_equal_median2_with_ACT,kmf_positive_higher_equal_median2_without_ACT)
plt.tight_layout()
plt.savefig("KM_plot_MRDPositive_HighRisk_ACT.svg", format="svg")
results = logrank_test(positive_higher_equal_median2_with_ACT['DFS_MONTHS'], positive_higher_equal_median2_without_ACT['DFS_MONTHS'], positive_higher_equal_median2_with_ACT['DFS_E'], positive_higher_equal_median2_without_ACT['DFS_E'])
logrank_p_value = results.p_value
print(logrank_p_value)

# %%
