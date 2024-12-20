from module.journey_configuer import admission_report_generation
from module.journey_configuer import discharge_report_generation
from module.journey_configuer import patients_full_journey


def generator(config):

    if config["GENERATED_REPORT_TYPE"]=="admission":
        admission_report_generation(config)
    else:
        admission_report_generation(config)
        discharge_report_generation(config)
        patients_full_journey(config)