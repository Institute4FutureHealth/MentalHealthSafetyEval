# Despite the style and format enforcement given to LLMs, they provide scores in different formats. We wrote different score extraction codes based on the output.
# We then manually call each of the methods based on the response of the LLMs providing the "* eval.csv"  files which are the output of the evaluate.py execution.
import pandas as pd
import re

# Function to extract numbers format "Score: x/10" where we want to extract x
def extract_score_m1(text):
    match = re.search(r'Score: (\d+(?:\.\d+)?)/10\n', text)
    if match:
        return match.group(1)
    return None  # In case there's no match

# Function to extract numbers format "Score: x" where we want to extract x
def extract_score_m2(text):
    scores = re.findall(r'Score: ([\d\.]+|N/A)\n', text)
    # Convert scores to float where possible, otherwise keep as None
    return [float(score) if score.replace('.', '').isdigit() else None for score in scores]

#function to extract scores for the Base method. Here we only have total score.
def extract_scores_for_base_method(file_path, extraction_mode=0):
# Load the CSV file
    df = pd.read_csv(file_path)

    # Apply the function to the relevant column
    extract_score = extract_score_m1
    if extraction_mode == 1:
        extract_score = extract_score_m2
    df['Extracted Score'] = df['gemini Answers'].apply(extract_score)

    # Display the updated DataFrame
    df.to_csv(f'{file_path.split("/")[-1].split(".csv")[0]}_processed.csv')


# Function to extract numbers format "Guideline[y]: x" where we want to extract x for guideline number y = [1..5]
def extract_scores_m3(text):
    scores = re.findall(r'Guideline\d+: ([\d\.]+|N/A)\n', text)
    # Convert scores to float where possible, otherwise keep as None
    return [float(score) if score.replace('.', '').isdigit() else None for score in scores]

# Function to extract numbers format "Guideline[y]: [x/10]" where we want to extract x for guideline number y = [1..5]
def extract_scores_m4(text):
    scores = re.findall(r'Guideline\d+: \[([\d\.]+|N/A)/10\]', text)
    # Convert scores to float where possible, otherwise keep as None
    return [float(score) if score.replace('.', '').isdigit() else None for score in scores]

#function to extract scores for the Base method. Here we have Guidelines' scores.
def extract_scores_for_other_methods(file_path, extraction_mode=0):
    # Load the CSV file
    df = pd.read_csv('Safety_Benchmark_Mental Health-agent eval_3.csv')

    # Function to extract all floating-point scores from a single text string
    extract_score = extract_scores_m3
    if extraction_mode == 1:
        extract_score = extract_score_m2

    # Apply the function to the relevant column
    df['Scores'] = df['agent Answers'].apply(extract_score)

    # Assuming there are always exactly 5 scores in each row,
    # split these into separate columns
    df[['Guideline1', 'Guideline2', 'Guideline3', 'Guideline4', 'Guideline5']] = pd.DataFrame(df['Scores'].tolist(), index=df.index)

    # Optionally, drop the original and intermediate columns if they are no longer needed
    df.drop(['Scores'], axis=1, inplace=True)

    # Store the DataFrame with new columns in a new CSV file
    df.to_csv(f'{file_path.split("/")[-1].split(".csv")[0]}_processed.csv', index=False)

if __name__ == "main":
    extract_scores_for_base_method("Safety_Benchmark_Mental Health-claude eval.csv", 0)
    # TODO: call methods on the files based on the output format

