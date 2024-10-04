import csv
import random
from datetime import date, datetime, timedelta
from multiprocessing import Process, Manager

from faker import Faker
import numpy as np
from tqdm import tqdm

from pyinstrument import Profiler

from lib.riskcalculator import estimate_health_risk
from lib.utils import (prepare_cvd_input, prepare_hypertension_input)

from pprint import pprint

# Ref: https://www.nber.org/system/files/working_papers/h0108/h0108.pdf
# It should be remembered, however, that distributions of height, weight and BMI do not have the same parameters.
# Adult male height is normally distributed with a standard deviation of about 2.5 inches (6.35 cm)
# while female height is normally distributed with a standard deviation of about 2.2 inches (5.59 cm).


# Helpers
def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def generate_population(num_samples: int, split: float, age_min: int, age_max: int, random_seed: int,
                        instance: int, return_list: Manager) -> None:

    samples = []
    date_start = date(2024, 1, 1)
    date_end = date(2024, 10, 4)

    hospitals = [
        {"name": "Hospital 1", "weight": 0.25},
        {"name": "Hospital 2", "weight": 0.4},
        {"name": "Hospital 3", "weight": 0.1},
        {"name": "Hospital 4", "weight": 0.25},
    ]

    # Random seeds
    #np.random.seed(random_seed)
    #random.seed(random_seed)
    faker = Faker()
    #faker.seed_instance(random_seed)

    for hosp in hospitals:

        for i in tqdm(range(int(num_samples * hosp["weight"]))):
            # Setup constants per sample
            random_visits: int = random.randint(1, 8)  # Number of visits for the patient
            i_exercise = bool(random.getrandbits(1))
            i_gender = np.random.choice(["M", "F"], p=[split, 1-split]).item()
            i_height = faker.random_int(min=130, max=190)
            i_weight = faker.random_int(min=40, max=160)
            i_first_visit_date = random_date(date_start, date_end)
            hospital = hosp["name"]
            coe = np.random.choice(["CoE 1", "CoE 2", "CoE 3", "CoE 4"], p=[0.4, 0.17, 0.2, 0.23]).item()
            i_bmi = round(i_weight / ((i_height/100) ** 2), 2)

            # Create record
            record = {
                "patient_id": faker.uuid4(),
                "gender": i_gender,
                "first_name": faker.first_name_male() if i_gender == "M" else Faker().first_name_female(),
                "last_name": faker.last_name(),
                "age": faker.random_int(min=age_min, max=age_max),
                "height": i_height,
                "weight": i_weight,
                "bmi_value": i_bmi,
                "bmi_model": i_bmi,
                "bmi_rating": "n/a",
                "bmi_rating_model": "n/a",
                "bp_systolic": np.random.uniform(size=1, low=100, high=200)[0].astype(int).item(), # Fix this
                "bp_diastolic": np.random.uniform(size=1, low=70, high=120)[0].astype(int).item(), # fix this
                "bp_rating": "n/a", # fix this
                "is_smoker": bool(random.getrandbits(1)),
                "waist_circumference": round(np.random.normal(119.50, 40.56), 2),
                "diabetes": bool(random.getrandbits(1)),
                "diabetes_family_history": bool(random.getrandbits(1)),
                "stroke_parents_siblings_before_65": bool(random.getrandbits(1)),
                "hypertension_medication": bool(random.getrandbits(1)),
                "hypertension_diagnosed": bool(random.getrandbits(1)),
                "hypertension_family_history": bool(random.getrandbits(1)),
                "exercise": i_exercise,
                "exercise_hours": faker.random_int(min=1, max=10) if i_exercise else 0,
                "diabetes_on_medication": bool(random.getrandbits(1)),
                "diabetes_type": random.choice(["1", "2"]),
                "ckd_diagnosed": bool(random.getrandbits(1)),
                "assessment_date": i_first_visit_date.isoformat(),
                "cvd_risk_score": -1,
                "ckd_risk_score": -1,
                "hypertension_risk_score": -1,
                "diabetes_risk_score": -1,
                "overall_risk_score": -1,
                "ncd_risk_rating": "n/a",
                "coe": coe,
                "hospital": hospital,
                "number_of_assessments": random_visits,
            }

            hr = get_hr(record)
            # NCD risk scores
            record["cvd_risk_score"] = hr["cvdRiskScore"]
            record["ckd_risk_score"] = hr["ckdRiskScore"]
            record["hypertension_risk_score"] = hr["hypertensionRiskScore"]
            record["diabetes_risk_score"] = hr["diabetesRiskScore"]
            record["overall_risk_score"] = hr["overallRiskScore"]
            record["cvd_ckd_absolute_score"] = hr["cvdCkdAbsoluteScore"]

            # NCD risk rating
            if record["overall_risk_score"] < 10:
                record["ncd_risk_rating"] = "Low Risk"
            elif 10 <= record["overall_risk_score"] < 20:
                record["ncd_risk_rating"] = "Medium Risk"
            elif 20 <= record["overall_risk_score"] < 30:
                record["ncd_risk_rating"] = "High Risk"
            elif record["overall_risk_score"] > 30:
                record["ncd_risk_rating"] = "Very High Risk"

            # BMI rating
            if i_bmi < 18.5:
                record["bmi_rating"] = "Underweight"
            elif 18.5 <= i_bmi < 25:
                record["bmi_rating"] = "Normal"
            elif 25 <= i_bmi < 27:
                record["bmi_rating"] = "Overweight"
            elif i_bmi > 27:
                record["bmi_rating"] = "Obese"

            record["bmi_rating_model"]= record["bmi_rating"]

            # Blood pressure rating
            if record["bp_systolic"] < 120 and record["bp_diastolic"] < 80:
                record["bp_rating"] = "Normal"
            elif 120 <= record["bp_systolic"] < 129 and record["bp_diastolic"] < 80:
                record["bp_rating"] = "Elevated"
            elif 130 <= record["bp_systolic"] < 139 or 80 <= record["bp_diastolic"] < 89:
                record["bp_rating"] = "hypertension stage 1"
            elif record["bp_systolic"] >= 140 or record["bp_diastolic"] >= 90:
                record["bp_rating"] = "hypertension stage 2"
            elif record["bp_systolic"] > 180 or record["bp_diastolic"] > 120:
                record["bp_rating"] = "hypertensive crisis"

            samples.append(record)

    return_list.extend(samples)

def get_hr(record: dict) -> dict:
    raw_cvd_input = {
        "gender": record["gender"],
        "age": record["age"],
        "bmi_model": record["bmi_value"],
        "waist_circumference": record["waist_circumference"],
        "hypertension_diagnosed": record["hypertension_diagnosed"],
        "hypertension_medication": record["hypertension_medication"],
        "stroke_parents_siblings_before_65": record["stroke_parents_siblings_before_65"],
        "is_smoker": record["is_smoker"],
    }

    #print("input raw:", raw_cvd_input)
    cvd_input = prepare_cvd_input(raw_cvd_input)
    #print("cvd input:", cvd_input)

    raw_hyp_input = {
        "age": record["age"],
        "is_smoker": record["is_smoker"],
        "gender": record["gender"],
        "exercise": record["exercise"],
        "exercise_hours": record["exercise_hours"],
        "hypertension_family_history": record["hypertension_family_history"],
        "bmi_model": record["bmi_value"],
        "diabetes_currently": record["diabetes"],
        "blood_pressure_systolic": record["bp_systolic"],
        "blood_pressure_diastolic": record["bp_diastolic"],
    }
    hyp_input = prepare_hypertension_input(raw_hyp_input)

    return estimate_health_risk(hyp_input, cvd_input, debug=False)

def do_the_do():
    manager = Manager()
    result_list = manager.list()

    cores = 4
    total_samples = 350_000
    processes = []

    for i in range(cores):
        n_samples = total_samples // cores
        p = Process(target=generate_population, args=(n_samples, 0.45, 25, 85, 42 + i, i-1, result_list))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    pop = result_list
    #pop = [vals for result in result_list for vals in result]
    #pop = generate_population( 1_000, 0.45, 25, 85)

    #with open("sample_data.json", "w") as f:
    #    json.dump(pop, f)

    keys = pop[0].keys()
    with open("sample_data.csv", "w") as csvfile:
        dict_writer = csv.DictWriter(csvfile, keys, quoting=csv.QUOTE_NONNUMERIC, doublequote=True)
        dict_writer.writeheader()
        dict_writer.writerows(pop)

do_the_do()

# Average heights by country
# https://en.wikipedia.org/wirecord.py:136ki/Average_human_height_by_country

# Human weight
# https://en.wikipedia.org/wiki/Human_body_weight