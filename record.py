import uuid
from datetime import datetime
import random

from faker import Faker
import numpy as np

from lib.riskcalculator import estimate_health_risk
from lib.utils import (prepare_cvd_input, prepare_hypertension_input)

# Ref: https://www.nber.org/system/files/working_papers/h0108/h0108.pdf
# It should be remembered, however, that distributions of height, weight and BMI do not have the same parameters.
# Adult male height is normally distributed with a standard deviation of about 2.5 inches (6.35 cm)
# while female height is normally distributed with a standard deviation of about 2.2 inches (5.59 cm).

class Patient:
    patient_id: str
    gender: str           # CVD, HYP
    first_name: str
    last_name: str
    age: int              # CVD, HYP
    height: float
    weight: float
    bmi_value: float      # CVD, HYP
    bmi_rating: str
    bp_systolic: float    # HYP
    bp_diastolic: float   # HYP
    bp_rating: str
    client_timestamp: datetime
    server_timestamp: datetime

    # health risk fields
    is_smoker: bool                            # HYP
    waist_circumference: float                 # CVD
    diabetes: bool                             # HYP
    diabetes_family_history: bool
    stroke_parents_siblings_before_65: bool    # CVD
    hypertension_medication: bool              # CVD (antihypertensives)
    hypertension_diagnosed: bool               # CVD (hypertension_medication)
    hypertension_family_history: bool
    exercise: int                              # HYP
    exercise_hours: int                        # HYP


    # Optional fields
    diabetes_on_medication: bool
    diabetes_type: int
    ckd_diagnosed: bool

    def estimate_health_risk(self) -> None:
        # CKD fields
        # fields = {
        #     'gender': {'type': str, 'default': 'gender'},
        #     'age': {'type': int, 'default': 'age'},
        #     'bmi_model': {'type': float, 'default': 'bmi_model'},
        #     'waist_circumference': {'type': float, 'default': 'waist_circumference'},
        #     'antihypertensives': {'type': str, 'default': 'hypertension_diagnosed'},
        #     'hypertension_on_medication': {'type': str, 'default': 'hypertension_medication'},
        #     'mi_or_stroke_family_history': {'type': str, 'default': 'stroke_parents_siblings_before_65'},
        #     'is_smoker': {'type': bool, 'default': 'is_smoker'},
        # }

        raw_cvd_input = {
            "gender": self.gender,
            "age": self.age,
            "bmi_model": self.bmi_value,
            "waist_circumference": self.waist_circumference,
            "hypertension_diagnosed": self.hypertension_diagnosed,
            "hypertension_medication": self.hypertension_medication,
            "stroke_parents_siblings_before_65": self.stroke_parents_siblings_before_65,
            "is_smoker": self.is_smoker,
        }

        cvd_input = prepare_cvd_input(raw_cvd_input)


        # Hypertension payload
        # fields = {
        #     'age': {'type': int, 'default_name': 'age', 'default_value': ''},
        #     'is_smoker': {'type': bool, 'default_name': 'is_smoker', 'default_value': False},
        #     'gender': {'type': str, 'default_name': 'gender', 'default_value': ''},
        #     'exercise': {'type': str, 'default_name': 'exercise', 'default_value': ''},
        #     'exercise_hours': {'type': int, 'default_name': 'exercise_hours', 'default_value': 0},
        #     'hypertension_family_history': {'type': str, 'default_name': 'hypertension_family_history',
        #                                     'default_value': ''},
        #     'bmi_model': {'type': float, 'default_name': 'bmi_model', 'default_value': 0.0},
        #     'diabetes_currently': {'type': str, 'default_name': 'diabetes', 'default_value': ''},
        #     'blood_pressure_systolic': {'type': float, 'default_name': 'm_0_obs_1_arm_left_systolic',
        #                                 'default_value': 0.0},
        #     'blood_pressure_diastolic': {'type': float, 'default_name': 'm_0_obs_1_arm_left_diastolic',
        #                                  'default_value': 0.0},
        # }

        raw_hyp_input = {
            "age": self.age,
            "is_smoker": self.is_smoker,
            "gender": self.gender,
            "exercise": self.exercise,
            "exercise_hours": self.exercise_hours,
            "hypertension_family_history": self.hypertension_family_history,
            "bmi_model": self.bmi_value,
            "diabetes_currently": self.diabetes,
            "blood_pressure_systolic": self.bp_systolic,
            "blood_pressure_diastolic": self.bp_diastolic,
        }

        hyp_input = prepare_hypertension_input(raw_hyp_input)


def generate_population(random_seed: int, num_samples: int) -> None:

    male_num = num_samples//2
    male_ages = np.random.normal(164.0, 6.35, male_num)
    male_weights = np.random.normal(63.6, 6.35, male_num)

    female_num = num_samples//2
    female_ages = np.random.normal(154.1, 5.59, female_num)
    female_weights = np.random.normal(59.8, 26.05, male_num)


def new_patient(random_seed: int) -> Patient:
    np.random.seed(random_seed)
    Faker().seed_instance(random_seed)

    p = Patient()
    p.patient_id = Faker().uuid4()
    p.gender = np.random.choice(["M", "F"], p=[0.5, 0.5])
    p.first_name = Faker().first_name_male() if p.gender == "M" else Faker().first_name_female()
    p.last_name = Faker().last_name()
    p.age = Faker().random_int(min=25, max=85)
    p.height = round(random.uniform(1.4, 1.82), 2)
    p.weight = round(random.uniform(40, 160), 2)
    p.bmi_value = round(p.weight / (p.height * p.height), 2)
    return p


user1 = new_patient(random_seed=10)
print(user1.patient_id)
print(user1.first_name)
print(user1.last_name)
print(user1.gender)
print(user1.age)
print(user1.height)
print(user1.weight)
print(user1.bmi_value)
#user1.estimate_health_risk()

# Average heights by country
# https://en.wikipedia.org/wirecord.py:136ki/Average_human_height_by_country

# Human weight
# https://en.wikipedia.org/wiki/Human_body_weight