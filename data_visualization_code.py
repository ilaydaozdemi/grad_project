# -*- coding: utf-8 -*-
"""
Created on Tue May 27 21:43:15 2025

@author: asus
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import numpy as np

file_path = r"DDGValueAnalysis.xlsx"
df = pd.read_excel(file_path)

df.columns = df.columns.str.strip()

df = df.dropna(subset=['DDG_EXP', 'DDG_FoldXArticle', 'Av_Cabs_DDG'])

df['AbsError_Article'] = abs(df['DDG_FoldXArticle'] - df['DDG_EXP'])
df['AbsError_AvCabs'] = abs(df['Av_Cabs_DDG'] - df['DDG_EXP'])

df['SqError_Article'] = (df['DDG_FoldXArticle'] - df['DDG_EXP'])**2
df['SqError_AvCabs'] = (df['Av_Cabs_DDG'] - df['DDG_EXP'])**2

rmse_article = np.sqrt(mean_squared_error(df['DDG_EXP'], df['DDG_FoldXArticle']))
rmse_cabs = np.sqrt(mean_squared_error(df['DDG_EXP'], df['Av_Cabs_DDG']))

r2_article = r2_score(df['DDG_EXP'], df['DDG_FoldXArticle'])
pearsonr_article, _ = stats.pearsonr(df['DDG_EXP'], df['DDG_FoldXArticle'])
r2_cabs = r2_score(df['DDG_EXP'], df['Av_Cabs_DDG'])
pearsonr_cabs, _ = stats.pearsonr(df['DDG_EXP'], df['Av_Cabs_DDG'])
print(f"Pearson correlation FoldX Article: {pearsonr_article:.4f}")
print(f"Pearson correlation Av_Cabs_DDG: {pearsonr_cabs:.4f}")


sns.set(style="whitegrid")

plt.figure(figsize=(8,6))
sns.regplot(x='DDG_EXP', y='DDG_FoldXArticle', data=df, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title(f'FoldX Article vs Experimental\nRMSE: {rmse_article:.3f} | R²: {r2_article:.3f} | Pearson: {pearsonr_article:.4f}')
plt.xlabel('Experimental ddG')
plt.ylabel('FoldX Article ddG')
plt.tight_layout()
plt.savefig("FoldXArticle_vs_Experimental.png")
plt.show()

plt.figure(figsize=(8,6))
sns.regplot(x='DDG_EXP', y='Av_Cabs_DDG', data=df, scatter_kws={'s': 50}, line_kws={'color': 'green'})
plt.title(f'Av_Cabs_DDG vs Experimental\nRMSE: {rmse_cabs:.3f} | R²: {r2_cabs:.3f} | Pearson: {pearsonr_cabs:.4f}')
plt.xlabel('Experimental ddG')
plt.ylabel('Average CABS ddG')
plt.tight_layout()
plt.savefig("AvCabs_vs_Experimental.png")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=df[['AbsError_Article', 'AbsError_AvCabs']])
plt.title('Comparison of Absolute Errors')
plt.ylabel('Absolute Error')
plt.xticks([0, 1], ['FoldX Article', 'Av_Cabs_DDG'])
plt.tight_layout()
plt.savefig("Absolute_Error_Boxplot.png")
plt.show()

print("==== Error Metrics ====")
print(f"FoldX Article - RMSE: {rmse_article:.4f} | R²: {r2_article:.4f}")
print(f"Av_Cabs_DDG   - RMSE: {rmse_cabs:.4f} | R²: {r2_cabs:.4f}")
