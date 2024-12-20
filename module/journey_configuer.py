
#Import Library
import pandas as pd
import google.generativeai as genai
import os
import torch
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bert_score import score
from sacrebleu.metrics import BLEU
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#from unidecode import unidecode

# Check if GPU is available

def device(config):
    if config["GPU"]=="YES":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    else:
        device= "cpu"
        print(f"Using device: {device}")
    return device

#CASE REPORT LOADED
def case_report_load(config):
    return pd.read_csv(config["CASE_REPORT_CSV_PATH"])


#text generation by gemini api
def generate_text_with_gemini(prompt, config):
    genai.configure(api_key=config["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")  # Or use gemini-1.5-flash if required
    response = model.generate_content(prompt)
    return response.text


#ADMISSION REPORT GEN
def admission_report_generation(config):


    case_report=case_report_load(config)
    if config["N_TESTING_ROW"]:
        case_report=case_report[0:int(config["N_TESTING_ROW"])]
    elif config["N_TESTING_ROW"]=="all":
        case_report
    else:
        case_report

    print("\n number of row:",len(case_report))

    # Example usage with your DataFrame
    for index, row in case_report.iterrows():
        clinical_narrative = row[config["CASE_REPORT_COLUMN_NAME"]]
        if clinical_narrative:  # Check if 'Clinical_Narrative' is not empty
            prompt = f"""
            "{clinical_narrative}"
            
            Based on the information above, write a realistic medical admission report in {config["GEN_LANGUAGE"]}
            for a patient upon arrival at the hospital. Use the information provided in {config["GEN_LANGUAGE"]} 
            and follow the writing style and terminology consistent with provided {config["GEN_LANGUAGE"]} case report. 
            While writing, adopt the perspective of a doctor and rebember this is not discharge report. 
            
            Follow these guidelines:

            1. Write the report as a single, unstructured paragraph in clinical language.
            2. Include only symptoms, signs, and relevant history of previous diseases, 
            using appropriate medical abbreviations (e.g., HTA, DM).
            3. Do not include treatment details, exam results, specific diagnoses, or follow-up treatments.
            4. Conclude the report with an indication of the initial treatment provided, 
            specifying the administered dose, but avoid explicitly labelling this section as 'initial treatment.'

            Ensure the report is in {config["GEN_LANGUAGE"]}
            and feels authentic, mimicking how a doctor might write the admission scenario. 
            Also remember, doctors can make simple mistakes while writing (e.g., typographical mistakes).
            """
            #, not Brazilian Portuguese,
            
            report = generate_text_with_gemini(prompt, config)

            print(f"""Number of GEN:{index+1}/{len(case_report)}
            \nGenerated Admission  Report:
            \n-------------------------\n{report}""")
            
            if report:
                case_report.loc[index, 'syn_admission_report'] = report  # Store the summary in a new column
            else:
                case_report.loc[index, 'syn_admission_report'] = "Report generation failed" # Handle cases where summary generation fails

            # Save the DataFrame with the new summary column
            updated_file_path = config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv"
            case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')

            updated_file_path_output = config["OUTPUT_PATH"]+"/synthetic_admission_report.csv"
            case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')
    print(f"DataFrame with summaries saved to: {updated_file_path}")

#DISCHARGE REPORT GEN

def discharge_report_generation(config):

    case_report=pd.read_csv(config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv")
    # Example usage with your DataFrame
    for index, row in case_report.iterrows():
        clinical_narrative = row[config["CASE_REPORT_COLUMN_NAME"]]
        admission_report = row['syn_admission_report']
        if clinical_narrative:  # Check if 'Clinical_Narrative' is not empty
            prompt = f"""
            "{clinical_narrative}" and "{admission_report}"

            Based on the information above, write a realistic medical discharge report in {config["GEN_LANGUAGE"]} 
            for a patient upon leaving the hospital. Use the information provided in {config["GEN_LANGUAGE"]} and follow 
            the writing style and terminology consistent with {config["GEN_LANGUAGE"]} case report. 
            While writing, adopt the perspective of a doctor and remember this is not an admission report. 

            Follow these guidelines:
            1. Write the report as a single, unstructured paragraph in clinical language.
            2. Include a summary of the patient's stay in the hospital.
            3. Include treatment summary, details of exams and their results, discharge medications, and follow-up instructions.

            Ensure the report is in {config["GEN_LANGUAGE"]}.
            And feels authentic, mimicking how a doctor might write the discharge scenario. 
            Also, remember that doctors can make simple mistakes while writing (e.g., typographical mistakes).
            """
            report = generate_text_with_gemini(prompt, config)

            print(f"""Number of GEN:{index+1}/{len(case_report)}
                \nGenerated Discharge Report:
                \n-------------------------\n{report}""")

            if report:
                case_report.loc[index, 'syn_discharge_report'] = report  # Store the summary in a new column
            else:
                case_report.loc[index, 'syn_discharge_report'] = "Report generation failed" # Handle cases where summary generation fails

            # Save the DataFrame with the new summary column
            updated_file_path =config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv"
            case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')

            updated_file_path_output = config["OUTPUT_PATH"]+"/synthetic_discharge_report.csv"
            case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')
    print(f"DataFrame with summaries saved to: {updated_file_path}")



#FULL JOURNEY GEN


def patients_full_journey(config):

    case_report=case_report=pd.read_csv(config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv")

    # Example usage with your DataFrame
    for index, row in case_report.iterrows():
        #clinical_narrative = row[config["CASE_REPORT_COLUMN_NAME"]]
        admission_report = row['syn_admission_report']
        discharge_report = row['syn_discharge_report']

        prompt = f"""

        "{admission_report}" and "{discharge_report}"

        Based on the admission and discharge reports provided, generate a detailed 
        report of the patient's full journey during their hospital stay. 
        Divide the information into multiple reports, such as 'divided days in different report based  patients situations' 
        'Surgery Report,' and so on, as appropriate to the events mentioned in the discharge report. 
        If the patient underwent surgery or any operation during their stay, 
        create a separate report detailing that specific event. 

        Write from the perspective of a doctor, ensuring the language feels authentic and mimics how 
        a doctor might document such scenarios. The report should be written in {config["GEN_LANGUAGE"]} 
        and formatted as a single, unstructured paragraph in clinical language. 
        Introduce small, natural errors like typographical mistakes to reflect a realistic documentation style.

        The generation should be in this order,
        1. Admission report (do not include date in the heading)
        2. Several reports based  patients situations during stay in the hospital. The report should be in day wise.
        3. Discharge Report (do not include date in the heading and also must mention the whole day of staying in the hospital)
        """
        report = generate_text_with_gemini(prompt, config)

        print(f"""Number of GEN:{index+1}/{len(case_report)}
            \nGenerated full journey Report:
            \n-------------------------\n{report}""")

        if report:
            case_report.loc[index, 'syn_full_journey'] = report  
        else:
            case_report.loc[index, 'syn_full_journey'] = "full journey generation failed" 

        # Save the DataFrame with the new summary column
        updated_file_path = config["CASE_REPORT_CSV_PATH"][:-4]+"_new.csv"
        case_report.to_csv(updated_file_path, index=False, encoding='utf-8-sig')

        updated_file_path_output = config["OUTPUT_PATH"]+"/synthetic_full_journey_report.csv"
        case_report.to_csv(updated_file_path_output, index=False, encoding='utf-8-sig')

    print(f"Finished: DataFrame with summaries saved to: {updated_file_path_output}")


