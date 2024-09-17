# Despite the style and format enforcement given to LLMs, they provide scores in different formats. We wrote different score extraction codes based on the output.
# This file generates the plots shown in the paper

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import scipy.stats as stats
# Load the Excel file
file_path = 'Result.xlsx'  # Replace with your file path

# Load multiple sheets
sheet_names = ['Drs', 'GPT4_evals', 'Misteral_eval', "Claude", "Gemini", "Semantic-mini-MiniLM-L6-v2", "Seamantic large mpnet-base-v2", "Agent"]  # Replace with your actual sheet names
sheet_renames = {
    'Drs_Avg': "Evaluators",
    'GPT4_evals_GPT4 eval 1': "GPT4 M1",
    'GPT4_evals_average': "GPT4 M2",
    'GPT4_evals_average.1': "GPT4 M3",
    'Misteral_eval_eval': "Misteral M1",
    'Misteral_eval_average': "Misteral M2",
    'Misteral_eval_average.1': "Misteral M3",
    'Claude_eval': "Claude M1",
    'Claude_average': "Claude M2",
    'Claude_average.1': "Claude M3",
    'Gemini_eval': "Gemini M1",
    'Gemini_average': "Gemini M2",
    'Gemini_average.1': "Gemini M3",
    'Semantic-mini-MiniLM-L6-v2_out of 10': "Embeddings\nM1",
    'Seamantic large mpnet-base-v2_out of 10': "Embeddings\nM2",
    'Agent_average.1': "Agent"
}
columns_order = [
    "Evaluators", 
    "GPT4 M1", "Misteral M1", "Claude M1", "Gemini M1",
    "GPT4 M2", "Misteral M2", "Claude M2", "Gemini M2",
    "GPT4 M3", "Misteral M3", "Claude M3", "Gemini M3",
    "Agent",
    "Embeddings\nM1", "Embeddings\nM2",
]

labels = [
    "Ground Truth", 
    "LLM Method 1",
    "LLM Method 2",
    "LLM Method 3",
    "Agent",
    "Embeddings",
]
legend_colors = [
    "LimeGreen", 
    "Purple", 
    "RebeccaPurple",
    "MediumPurple",
    "lightblue", 
    "OrangeRed"
]

columns_to_plot = [
    ["Avg"],
    ["GPT4 eval 1", "average", "average.1"],
    ["eval", "average", "average.1"],
    ["eval", "average", "average.1"],
    ["eval", "average", "average.1"],
    ["out of 10"],
    ["out of 10"],
    ["average.1"]
]

colors = [
    "LimeGreen", 
    "Purple", "Purple", "Purple", "Purple", 
    "RebeccaPurple", "RebeccaPurple", "RebeccaPurple", "RebeccaPurple",
    "MediumPurple", "MediumPurple", "MediumPurple", "MediumPurple",
    "lightblue", 
    "OrangeRed", "OrangeRed"
]

axes_indexes = {
    'GPT4_evals_GPT4 eval 1': 0,
    'GPT4_evals_average': 4,
    'GPT4_evals_average.1': 8,
    'Misteral_eval_eval': 1,
    'Misteral_eval_average': 5,
    'Misteral_eval_average.1': 9,
    'Claude_eval': 2,
    'Claude_average': 6,
    'Claude_average.1': 10,
    'Gemini_eval': 3,
    'Gemini_average': 7,
    'Gemini_average.1': 11,
    'Semantic-mini-MiniLM-L6-v2_out of 10': 13,
    'Seamantic large mpnet-base-v2_out of 10': 14,
    'Agent_average.1': 12
}

def boxplot():
# Initialize an empty DataFrame to hold all the data
    combined_data = pd.DataFrame()

    # Iterate over each sheet
    for i, sheet in enumerate(sheet_names):
        data = pd.read_excel(file_path, sheet_name=sheet)
        # Select the specific column and add a new column to indicate the sheet source
        for column in columns_to_plot[i]:
            data1 = data[[column]]
            data1 = data1.rename(columns={column: 'Scores'})
            data1['Methods'] = sheet_renames[sheet + "_" + column]     
            combined_data = pd.concat([combined_data, data1], ignore_index=True)

    print("combined_data", combined_data)
    # Create the boxplot
    sns.boxplot(x='Methods', y='Scores', data=combined_data, color='lightblue', palette=colors, order=columns_order)

    # Add observations using stripplot with jitter
    sns.stripplot(x='Methods', y='Scores', data=combined_data, color='0', alpha=0.5, size=3, jitter=True, order=columns_order)
    legend_handles = [Patch(color=color, label=label) for label, color in zip(labels, legend_colors)]
    plt.legend(handles=legend_handles, title='Methods')
    # Show the plot
    # plt.title('Box Plot with Observations from Multiple Sheets')

    plt.show()

def boxplotdiff():
    first_sheet_data = pd.read_excel(file_path, sheet_name=sheet_names[0])
    # Initialize an empty DataFrame to hold all the data
    combined_data = pd.DataFrame()

    # Iterate over each sheet
    for i, sheet in enumerate(sheet_names):
        if sheet == "Drs":
            continue
        data = pd.read_excel(file_path, sheet_name=sheet)
        # Select the specific column and add a new column to indicate the sheet source
        for column in columns_to_plot[i]:
            data1 = data[[column]]
            data1 = data1.rename(columns={column: 'Scores'})
            data1['Scores'] = (data1['Scores']-first_sheet_data[columns_to_plot[0][0]]).abs()
            print("data", first_sheet_data[[columns_to_plot[0][0]]], data1)
            data1['Methods'] = sheet_renames[sheet + "_" + column]     
            combined_data = pd.concat([combined_data, data1], ignore_index=True)

    print("data", data)
    # Create the boxplot
    sns.boxplot(x='Methods', y='Scores', data=combined_data, color='lightblue', palette=colors[1:], order=columns_order[1:])

    # Add observations using stripplot with jitter
    sns.stripplot(x='Methods', y='Scores', data=combined_data, color='0', alpha=0.5, size=3, jitter=True, order=columns_order[1:])
    legend_handles = [Patch(color=color, label=label) for label, color in zip(labels[1:], legend_colors[1:])]
    plt.legend(handles=legend_handles, title='Methods')
    # Show the plot
    # plt.title('Box Plot with Observations from Multiple Sheets')
    plt.show()


def bland_altman_plot(data1, data2, sheet_name, ax):
    # Calculate means and differences
    means = (data1 + data2) / 2
    differences = data1 - data2
    
    # Calculate average difference and limits of agreement
    mean_diff = differences.mean()
    std_diff = differences.std()
    lower = mean_diff - 1.96 * std_diff
    upper = mean_diff + 1.96 * std_diff
    
    # Plot using Seaborn for styling and Matplotlib for the plot elements
    sns.scatterplot(x=means, y=differences, ax=ax, color="blue", alpha=0.6)
    ax.axhline(mean_diff, color='red', linestyle='--')
    ax.axhline(lower, color='green', linestyle='--')
    ax.axhline(upper, color='green', linestyle='--')
    ax.set_title(f'{sheet_name} vs Evaluators')
    ax.set_xlabel('Means')
    ax.set_ylabel('Differences')
    ax.set_ylim(-10, 10)
    ax.set_xlim(0, 10)
    print(f"Bland-Altman Plot: {sheet_name} vs Evaluators. Lower: {lower} Upper: {upper} {upper-lower} Meant Diff {mean_diff} STD Diff {std_diff}")

def bland_altman():
    first_sheet_data = pd.read_excel(file_path, sheet_name=sheet_names[0])
    num_sheets = len(sheet_renames.keys()) - 1
    cols = 4  # Define number of columns
    rows = int(np.ceil(num_sheets / cols))  # Calculate the required number of rows

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4 * rows), constrained_layout=True)
    if num_sheets > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    # Generate a plot for each sheet comparison
    j = 0
    for i, sheet_name in enumerate(sheet_names[1:], start=1):
        current_sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
        for column in columns_to_plot[i]:
            bland_altman_plot(first_sheet_data[columns_to_plot[0][0]], current_sheet_data[column], sheet_renames[sheet_name + "_" + column] , axes[axes_indexes[sheet_name + "_" + column]])
            j+=1
            print(j)

    # plt.tight_layout()
    plt.legend()
    plt.show()

def regression():
    first_sheet_data = pd.read_excel(file_path, sheet_name=sheet_names[0])
    num_sheets = len(sheet_renames.keys()) - 1
    cols = 3  # Define number of columns
    rows = int(np.ceil(num_sheets / cols))  # Calculate the required number of rows

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 4 * rows), constrained_layout=True)
    if num_sheets > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    # Generate a plot for each sheet comparison
    j = 0
    for i, sheet_name in enumerate(sheet_names[1:], start=1):
        current_sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
        for column in columns_to_plot[i]:
            axes[j].scatter(first_sheet_data[columns_to_plot[0][0]], current_sheet_data[column], color='blue', label='Data points')
            axes[j].set_title(f'{sheet_renames[sheet_name + "_" + column]} vs Evaluators')
            axes[j].set_ylabel(f'{sheet_renames[sheet_name + "_" + column]}')
            axes[j].set_xlabel('Evaluators')   
            slope, intercept, r_value, p_value, std_err = stats.linregress(first_sheet_data[columns_to_plot[0][0]], current_sheet_data[column])
            print("namme///////", sheet_renames[sheet_name + "_" + column], r_value, p_value, std_err)
            polynomial = np.poly1d([slope, intercept])
            x_lin_reg = np.linspace(first_sheet_data[columns_to_plot[0][0]].min(), first_sheet_data[columns_to_plot[0][0]].max(), 100)
            y_lin_reg = polynomial(x_lin_reg)
            axes[j].set_xlim(0, 10.5)
            axes[j].set_ylim(0, 10.5)
            # Adding regression line to the scatter plot
            axes[j].plot(x_lin_reg, y_lin_reg, color='red', label='Regression line')     
            j+=1
            print(j)

    # plt.tight_layout()
    plt.legend()
    plt.show()
    
boxplot()
bland_altman()
boxplotdiff()
regression()