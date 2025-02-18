# Report Generation Pipeline

A patient's interaction with a hospital, from admission to discharge, is typically recorded in a series of text documents that outline their journey. This repository provides a structured pipeline for creating a patient's journey based on case report data. The repository includes tools for processing data, analyzing it, and generating outputs.

## 📂 Repository Structure

📦 report-generation ├── 📂 data # Raw data used for generating reports and final generated data in one csv file ├── 📂 example_generation # One example generation with English translation ├── 📂 exploratory_results # Intermediate results and insights ├── 📂 module # Core modules for data processing and analysis ├── 📂 output # Final generated reports ├── 📄 Prompt.pdf # Prompt used for generating reports ├── 📄 README.md # Repository documentation ├── 📄 config.json # Configuration file for setting up the pipeline ├── 📄 requirements.txt # Dependencies required to run the project └── 📄 run.py # Main script to execute the report generation pipeline



## 🚀 Getting Started

### 1️⃣ Clone the Repository

` git clone https://github.com/tahsirmunna/report-generation.git`

` cd report-generation`

### 2️⃣ Install Dependencies
Ensure you have Python installed, then install the required dependencies using:

`pip install -r requirements.txt`


### 3️⃣ Run the Report Generation Script
Execute the main script to start the pipeline:

`python run.py`


### ⚙️ Configuration
Modify the config.json file to customize parameters for data processing and report generation.

#### 📑 Files and Folders
`data/:` Contains raw datasets used in the pipeline and generated all reports in one .csv file.

`example_generation/:` Includes one example of the generation with English translation.

`exploratory_results/:` Stores results from exploratory data analysis including a table of the most frequent terms and the number of terms in the generated reports, topic distribution plots, and word cloud plots to visualize the most frequent terms in the generated reports.

`module/:` Contains core functions for data processing.

`output/:` Final generated reports will be saved here.

`Prompt.pdf:` Document describing the report generation prompt.

`requirements.txt:` Lists dependencies for setting up the environment.

`run.py:` Main script to execute the pipeline.

#### 📝 Contributions
Feel free to fork this repository, submit issues, and contribute via pull requests.

##### Happy Coding! 🎯
