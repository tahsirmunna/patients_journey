import json
from module.report_generation import generator
from module.report_evaluation import evaluator


# Specify the path to your JSON config file
json_file_path = "config.json"

# Open the file and load the JSON data
with open(json_file_path, 'r') as json_file:
    config = json.load(json_file)


print("\nReport Gereration started...\n")
#generator(config)
print("\nGeneration Finished!\n")


print("\nEvaluation Started...\n")
evaluator(config)
print("\nEvaluation Finished...\n")

print("Full process done...")





