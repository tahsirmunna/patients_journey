# Importing necessary libraries
import pandas as pd  # For handling and manipulating structured data
import google.generativeai as genai  # For using Google's Generative AI functionalities
import os  # For operating system-related functionalities (e.g., file path handling)
import torch  # For deep learning models and computations
from transformers import pipeline  # For using pre-trained models from Hugging Face
from sklearn.metrics.pairwise import cosine_similarity  # For computing similarity between vectors
import numpy as np  # For numerical computations
from bert_score import score  # For evaluating text similarity using BERT embeddings
from sacrebleu.metrics import BLEU  # For BLEU score calculation (text similarity evaluation)
from sklearn.feature_extraction.text import TfidfVectorizer  # For text vectorization using TF-IDF
from sklearn.cluster import DBSCAN  # Density-based clustering
from sklearn.cluster import KMeans  # K-means clustering

# Custom function from another module to define the device (CPU/GPU)
from module.journey_configuer import device

# Main evaluation function that performs various evaluations on clinical narratives
def evaluator(config):
    """
    Evaluates clinical narratives using different metrics: NER, BERT score, BLEU score, clustering, and classification.
    """
    # Load the input data from a specified CSV file
    gen_data = pd.read_csv(config["CASE_REPORT_CSV_PATH"][:-4] + "_new.csv")

    # Handling a subset of data based on the configuration
    if isinstance(config["N_TESTING_ROW"], int):  # Check if it's an integer
        gen_data = gen_data[0:config["N_TESTING_ROW"]]  # Use only the specified number of rows
    elif config["N_TESTING_ROW"] == "all":  # If it's the string "all"
        gen_data  # Use all rows
    else:
        gen_data  # Default case: no filtering

    
    # Initialize the NER pipeline using a pre-trained model
    print("NER model:", config["NER_MODEL"])
    ner_pipeline = pipeline('ner', model=config["NER_MODEL"], aggregation_strategy='average', device=device(config))

    # Function to extract named entities from text using the NER pipeline
    def extract_ner(text):
        """
        Extracts named entities from the provided text using the NER model pipeline.
        """
        try:
            return ner_pipeline(text)  # Extract entities using the NER pipeline
        except Exception as e:
            print(f"Error processing text: {e}")
            return []  # If there's an error, return an empty list

    

    def ner_similarity(ner1, ner2):
        """
        Checks if all named entities in ner2 exist in ner1.
        If true, returns 1. Otherwise, calculates the percentage of ner2 entities present in ner1.

        Parameters:
        - ner1: List of named entities, where each entity is a dictionary with an 'entity' key.
        - ner2: List of named entities, where each entity is a dictionary with an 'entity' key.

        Returns:
        - 1 if all entities in ner2 exist in ner1.
        - A float value representing the percentage of ner2 entities found in ner1 if not all match.
        """
        # Extract the entity names from ner1 and ner2
        entities1 = {entity.get('entity', 'O') for entity in ner1}  # Use a set for faster lookups
        entities2 = {entity.get('entity', 'O') for entity in ner2}
        
        # Check if all entities in ner2 exist in ner1
        if entities2.issubset(entities1):
            return 1  # All entities in ner2 exist in ner1
        
        # Calculate the percentage of ner2 entities found in ner1
        matching_entities = entities1.intersection(entities2)
        percentage = len(matching_entities) / len(entities2) if entities2 else 0
        return percentage

    
    '''
    # Function to calculate similarity between two sets of named entities
    def ner_similarity(ner1, ner2):
        """
        Calculates the similarity between two named entity sets using cosine similarity.
        """
        entities1 = [entity.get('entity', 'O') for entity in ner1]  # Extract the entity names from ner1
        entities2 = [entity.get('entity', 'O') for entity in ner2]  # Extract the entity names from ner2
        all_entities = set(entities1 + entities2)  # Combine the entities from both sets
        vector1 = [entities1.count(entity) for entity in all_entities]  # Vector representation for ner1
        vector2 = [entities2.count(entity) for entity in all_entities]  # Vector representation for ner2
        if vector1 and vector2:  # Ensure both vectors are non-empty
            return cosine_similarity([vector1], [vector2])[0][0]  # Calculate cosine similarity
        else:
            return 0  # Return similarity of 0 if vectors are empty
    '''

    # Function to calculate the average BERT score between references and candidates
    def calculate_bert_score(references, candidates):
        """
        Calculates BERT score (precision, recall, F1) between the reference and candidate text.
        """
        P, R, F1 = score(candidates, references, lang="PT", verbose=True)  # Calculate BERT score
        return F1.mean().item()  # Return the average F1 score

    # Function to calculate BLEU score for text similarity
    def calculate_bleu_score(references, candidates):
        """
        Calculates BLEU score for the similarity between references and candidates.
        """
        bleu = BLEU(effective_order=True)  # Initialize BLEU metric
        score = bleu.sentence_score(candidates, [references])  # Calculate BLEU score
        return score.score / 100.0  # Normalize BLEU score between 0 and 1
    


    if config["SCORING"].lower() == "yes":
        print("I AM SCORE")
        # List to store evaluation results
        results = []

        # Iterate over the rows of the data to perform evaluations
        for index, row in gen_data[0:5].iterrows():  # Limit processing to 5 rows for demonstration
            try:
                print("Processing NER calculation")
                # Extract named entities from various text columns
                admission_ner = extract_ner(row['syn_admission_report'])
                discharge_ner = extract_ner(row['syn_discharge_report'])
                clinical_ner = extract_ner(row[config["CASE_REPORT_COLUMN_NAME"]])
                journey_ner = extract_ner(row['syn_full_journey'])

                # Calculate NER-based similarity scores
                ner_similarity_admission = ner_similarity(clinical_ner, admission_ner)
                ner_similarity_discharge = ner_similarity(clinical_ner, discharge_ner)
                ner_similarity_journey = ner_similarity(clinical_ner, journey_ner)

                print("Processing BERT score calculation")
                # Calculate BERT scores for text similarity
                bert_score_admission = calculate_bert_score([row[config["CASE_REPORT_COLUMN_NAME"]]], [row['syn_admission_report']])
                bert_score_discharge = calculate_bert_score([row[config["CASE_REPORT_COLUMN_NAME"]]], [row['syn_discharge_report']])
                bert_score_journey = calculate_bert_score([row[config["CASE_REPORT_COLUMN_NAME"]]], [row['syn_full_journey']])

                print("Processing BLEU score calculation")
                # Calculate BLEU scores for text similarity
                bleu_score_admission = calculate_bleu_score(row[config["CASE_REPORT_COLUMN_NAME"]], row['syn_admission_report'])
                bleu_score_discharge = calculate_bleu_score(row[config["CASE_REPORT_COLUMN_NAME"]], row['syn_discharge_report'])
                bleu_score_journey = calculate_bleu_score(row[config["CASE_REPORT_COLUMN_NAME"]], row['syn_full_journey'])

                # Append the results for this row to the results list
                results.append({
                    'admission_ner_completeness': ner_similarity_admission,
                    'discharge_ner_completeness': ner_similarity_discharge,
                    'full_journey_ner_completeness': ner_similarity_journey,

                    'bert_score_admission': bert_score_admission,
                    'bert_score_discharge': bert_score_discharge,
                    'bert_score_full_journey': bert_score_journey,

                    'bleu_score_admission': bleu_score_admission,
                    'bleu_score_discharge': bleu_score_discharge,
                    'bleu_score_full_journey': bleu_score_journey
                })

            except KeyError as e:
                print(f"Warning: Column {e} not found in DataFrame for row {index}. Skipping.")
                results.append({
                    'admission_ner_similarity': np.nan,
                    'discharge_ner_similarity': np.nan,
                    'full_journey_ner_similarity': np.nan,

                    'bert_score_admission': None,
                    'bert_score_discharge': None,
                    'bert_score_full_journey': None,

                    'bleu_score_admission': None,
                    'bleu_score_discharge': None,
                    'bleu_score_full_journey': None
                })
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                results.append({
                    'admission_ner_similarity': np.nan,
                    'discharge_ner_similarity': np.nan,
                    'full_journey_ner_similarity': np.nan,

                    'bert_score_admission': None,
                    'bert_score_discharge': None,
                    'bert_score_full_journey': None,

                    'bleu_score_admission': None,
                    'bleu_score_discharge': None,
                    'bleu_score_full_journey': None
                })

        # Append evaluation results to the original data
        gen_data = pd.concat([gen_data, pd.DataFrame(results)], axis=1)
        # Save the evaluated data to a CSV file
        gen_data.to_csv(config["OUTPUT_PATH"] + "ner_bert_bleu_score_evaluation.csv", index=False)

        print("\nScoring Processing done..\n")

    # Perform clustering if enabled in the configuration
    if config["CLUSTERING"].lower() == "yes":
        print("\nClustering processing...\n")

        # Replace NaN values with empty strings to avoid issues during vectorization
        gen_data['syn_full_journey'] = gen_data['syn_full_journey'].fillna("")

        # Create a TfidfVectorizer object for text vectorization
        tfidf = TfidfVectorizer()

        # Fit the vectorizer to the 'syn_full_journey' column
        tfidf.fit(gen_data['syn_full_journey'])

        # Get the feature names (keywords) from the vectorizer
        keywords = tfidf.get_feature_names_out()

        # Add the top keywords to the DataFrame
        gen_data['processed_keywords'] = [
            ' '.join(keywords[index]) 
            for index in tfidf.transform(gen_data['syn_full_journey']).toarray().argsort()[:, -5:][:, ::-1]
        ]

        # Vectorize the processed keywords using TF-IDF
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(gen_data['processed_keywords'])

        # Create clusters using the K-means algorithm
        num_clusters = int(config["N_CLUSTER"])  # Define the number of clusters
        kmeans = KMeans(n_clusters=num_clusters)

        gen_data['cluster'] = kmeans.fit_predict(tfidf_matrix)

        # Save the results with clustering to a CSV file
        gen_data.to_csv(config["OUTPUT_PATH"] + "cluster_ner_bert_bleu_score_evaluation.csv", index=False)

        print("\nClustering Processing done..\n")


    if config["PT_CLASSIFYING"].lower() == "yes":
        # Initialize the pipeline for text classification (Portuguese language model)
        pipe = pipeline("text-classification", model="liaad/LVI_albertina-900m-portuguese-ptpt-encoder", device=device(config))

        # Define constants
        MAX_TOKENS = 722 # Maximum token length for the model

        # Function to chunk sentences exceeding max tokens
        def chunkSentence(text):
            """
            Splits text into smaller chunks if it exceeds the maximum token length.
            """
            words = text.split()
            return [' '.join(words[i:i + MAX_TOKENS]) for i in range(0, len(words), MAX_TOKENS)]

        # Function to clean processed lines (placeholder, implement as needed)
        def cleaningProcess(lines):
            """
            Clean the processed lines by stripping extra whitespace and removing empty lines.
            """
            return [line.strip() for line in lines if line.strip()]

        # Function to identify variant from a single text input
        def identifyVariantFromText(text):
            """
            Identifies the language variant from a single text input.
            """
            processed_lines = []

            # Split and chunk if necessary
            if len(text.split()) > MAX_TOKENS:
                sentences = chunkSentence(text)
                processed_lines.extend(sentences)
            else:
                processed_lines.append(text)

            # Clean the processed lines
            processed_lines = cleaningProcess(processed_lines)

            # Get predictions from the classifier
            variant_result = pipe(processed_lines)

            # Create DataFrame from results
            res_df = pd.DataFrame.from_dict(variant_result)

            # Adjust scores for PT-BR label
            res_df.loc[res_df["label"] == "PT-BR", "score"] *= -1

            # Calculate overall result
            ret_result = {
                "score": res_df["score"].sum(),
                "label": res_df["label"].value_counts().idxmax()
            }

            return ret_result

  
        # Apply the variant identification function to the relevant columns
        admission_results = gen_data["syn_admission_report"].apply(identifyVariantFromText)
        discharge_results = gen_data["syn_discharge_report"].apply(identifyVariantFromText)
        full_journey_results = gen_data["syn_full_journey"].apply(identifyVariantFromText)
        

        # Extract scores and labels into separate columns
        print("Finding variant for admission report")
        gen_data["admission_variant_score"] = admission_results.apply(lambda x: x["score"])
        gen_data["admission_variant_label"] = admission_results.apply(lambda x: x["label"])

        print("Finding variant for discharge report")
        gen_data["discharge_report_variant_score"] = discharge_results.apply(lambda x: x["score"])
        gen_data["discharge_report_variant_label"] = discharge_results.apply(lambda x: x["label"])

        print("Finding variant for full journey")
        gen_data["full_journey_variant_score"] = full_journey_results.apply(lambda x: x["score"])
        gen_data["full_journey_variant_label"] = full_journey_results.apply(lambda x: x["label"])

        # Save the results with PT classification to a new CSV file
        gen_data.to_csv(config["OUTPUT_PATH"] + "PT_cluster_ner_bert_bleu_score_evaluation.csv", index=False)

        print("\nPT Classifying Processing done..\n")
