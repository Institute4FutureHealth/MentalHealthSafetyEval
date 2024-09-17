# We converted all the extracted results we will have separate csv files. we combined them into single Result.xlsx file to make the rest of the evaluation easier.
# This file checkes the normality of the Experts scores agains automated evaluation scores. This will help us choose the ttest method. The result was showing that
# the distributions are normal and so we used paired ttest.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

sheet_names = ['GPT4_evals', 'Misteral_eval', "Claude", "Gemini", "Semantic-mini-MiniLM-L6-v2", "Seamantic large mpnet-base-v2", "Agent"]  # Replace with your actual sheet names
columns_to_plot = [
    ["GPT4 eval 1", "average", "average.1"],
    ["eval", "average", "average.1"],
    ["eval", "average", "average.1"],
    ["eval", "average", "average.1"],
    ["out of 10"],
    ["out of 10"],
    ["average.1"]
]

file_path = 'Result.xlsx'
data = pd.read_excel(file_path, sheet_name="Drs")["Avg"]
# Generate or load your data
# data = np.random.normal(loc=0, scale=1, size=100)  # Example data
for i, sheet in enumerate(sheet_names):
    data2 = pd.read_excel(file_path, sheet_name=sheet)
    for column in columns_to_plot[i]:
        print("sheet", sheet, column)
        diff = data-data2[column]
        print("data", data.shape, data2[column].shape, diff.shape)
        # Histogram
        plt.hist(diff, bins='auto', alpha=0.7, color='blue')
        plt.show()

        # Q-Q plot
        stats.probplot(diff, dist="norm", plot=plt)
        plt.show()

        # Shapiro-Wilk Test
        shapiro_test = stats.shapiro(diff)
        print('Shapiro-Wilk Test:', shapiro_test)

        # Kolmogorov-Smirnov Test
        ks_test = stats.kstest(diff, 'norm')
        print('Kolmogorov-Smirnov Test:', ks_test)

        # Anderson-Darling Test
        ad_test = stats.anderson(diff, dist='norm')
        print('Anderson-Darling Test:', ad_test)
