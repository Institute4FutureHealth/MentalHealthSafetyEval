# We converted all the extracted results we will have separate csv files. we combined them into single Result.xlsx file to make the rest of the evaluation easier.
# This file runs the paired ttest between experts scores and other evaluation methods.
import numpy as np
from scipy import stats
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

for i, sheet in enumerate(sheet_names):
    data2 = pd.read_excel(file_path, sheet_name=sheet)
    for column in columns_to_plot[i]:
        print(sheet, column)
        # Example data
        before_treatment = data
        after_treatment = data2[column]

        # Calculate the t-test on TWO RELATED samples
        t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

        print("t-statistic:", t_statistic)
        print("p-value:", p_value)

        # Interpretation
        if p_value < 0.05:
            print("Reject the null hypothesis - there is a significant difference between the two conditions.")
        else:
            print("Fail to reject the null hypothesis - there is no significant difference between the two conditions.")
