import uuid
from datetime import datetime, timedelta
import random

from faker import Faker
import numpy as np
from tqdm import tqdm

from pyinstrument import Profiler

from lib.riskcalculator import estimate_health_risk
from lib.utils import (prepare_cvd_input, prepare_hypertension_input)

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


np.random.seed(42)
faker = Faker()
faker.seed_instance(42)

def generate_population(random_seed: int, num_samples: int, split: float, age_min: int, age_max: int,
                        height: float, weight: float) -> list:

    male_samples: float = float(num_samples//split)
    #male_weights = np.random.normal(63.6, 6.35, male_samples)

    female_samples = num_samples//1-split
    #female_ages = np.random.normal(154.1, 5.59, female_samples)
    #female_weights = np.random.normal(59.8, 26.05, female_samples)

    samples = []
    for i in tqdm(range(num_samples)):

        exercise = bool(random.getrandbits(1))
        gender = np.random.choice(["M", "F"], p=[split, 1-split])
        record = {
            "patient_id": faker.uuid4(),
            "gender": gender,
            "first_name": faker.first_name_male() if gender == "M" else Faker().first_name_female(),
            "last_name": faker.last_name(),
            "age": faker.random_int(min=age_min, max=age_max),
            "height": faker.random_int(min=100, max=178),
            "weight": faker.random_int(min=40, max=160),
            "bmi_value": round(weight / (height * height), 2),
            "bmi_model": round(weight / (height * height), 2),
            "bmi_rating": 0,
            "bp_systolic": np.random.uniform(size=1, low=90, high=140), # Fix this
            "bp_diastolic": np.random.uniform(size=1, low=90, high=140), # fix this
            "bp_rating": 0, # fix this
            "is_smoker": bool(random.getrandbits(1)),
            "waist_circumference": np.random.uniform(size=1, low=90, high=140),
            "diabetes": bool(random.getrandbits(1)),
            "diabetes_family_history": bool(random.getrandbits(1)),
            "stroke_parents_siblings_before_65": bool(random.getrandbits(1)),
            "hypertension_medication": bool(random.getrandbits(1)),
            "hypertension_diagnosed": bool(random.getrandbits(1)),
            "hypertension_family_history": bool(random.getrandbits(1)),
            "exercise": exercise,
            "exercise_hours": faker.random_int(min=1, max=10) if exercise else 0,
            "diabetes_on_medication": bool(random.getrandbits(1)),
            "diabetes_type": random.choice(["1", "2"]),
            "ckd_diagnosed": bool(random.getrandbits(1)),
            "client_timestamp": 0,
            "server_timestamp": 0,
        }

    return samples

def get_hr(gender: str, age: int, bmi_value: float, waist_circumference: float,
                         hypertension_diagnosed: bool, hypertension_medication: bool,
                         stroke_parents_siblings_before_65: bool,
                         is_smoker: bool, exercise: bool, exercise_hours: int, diabetes: bool,
                         hypertension_family_history: bool, diabetes_on_medication: bool,
                         bp_systolic: float, bp_diastolic: float) -> dict:
    raw_cvd_input = {
        "gender": gender,
        "age": age,
        "bmi_model": bmi_value,
        "waist_circumference": waist_circumference,
        "hypertension_diagnosed": hypertension_diagnosed,
        "hypertension_medication": hypertension_medication,
        "stroke_parents_siblings_before_65": stroke_parents_siblings_before_65,
        "is_smoker": is_smoker,
    }

    cvd_input = prepare_cvd_input(raw_cvd_input)

    raw_hyp_input = {
        "age": age,
        "is_smoker": is_smoker,
        "gender": gender,
        "exercise": exercise,
        "exercise_hours": exercise_hours,
        "hypertension_family_history": hypertension_family_history,
        "bmi_model": bmi_value,
        "diabetes_currently": diabetes,
        "blood_pressure_systolic": bp_systolic,
        "blood_pressure_diastolic": bp_diastolic,
    }

    hyp_input = prepare_hypertension_input(raw_hyp_input)

    return estimate_health_risk(hyp_input, cvd_input)

def do_the_do():
    pop = generate_population(42, 35_000, 0.45, 25, 85,
                              height=159.1, weight=68.1)


with Profiler(interval=0.1) as p:
    do_the_do()

p.print()

# Average heights by country
# https://en.wikipedia.org/wirecord.py:136ki/Average_human_height_by_country

# Human weight
# https://en.wikipedia.org/wiki/Human_body_weight