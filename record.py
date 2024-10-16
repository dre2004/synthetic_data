import csv
import random
from copy import deepcopy
from datetime import date, datetime, timedelta
from multiprocessing import Process, Manager

from faker import Faker
import numpy as np
import pandas as pd
from tqdm import tqdm

from pyinstrument import Profiler

from lib.riskcalculator import estimate_health_risk
from lib.utils import (prepare_cvd_input, prepare_hypertension_input)

from pprint import pprint

# Ref: https://www.nber.org/system/files/working_papers/h0108/h0108.pdf
# It should be remembered, however, that distributions of height, weight and BMI do not have the same parameters.
# Adult male height is normally distributed with a standard deviation of about 2.5 inches (6.35 cm)
# while female height is normally distributed with a standard deviation of about 2.2 inches (5.59 cm).


# Rating constants
BP_RATING_NORMAL = "Normal"
BP_RATING_ELEVATED = "Elevated"
BP_RATING_HYPERTENSION_STAGE1 = "hypertension stage 1"
BP_RATING_HYPERTENSION_STAGE2 = "hypertension stage 2"
BP_RATING_HYPERTENSION_CRISIS = "hypertension stage 3"
BP_RATING_UNCLASSIFIED = "Unclassified"

BMI_RATING_UNDERWEIGHT = "Underweight"
BMI_RATING_NORMAL = "Normal"
BMI_RATING_OVERWEIGHT = "Overweight"
BMI_RATING_OBESE = "Obese"

NCD_RISK_RATING_LOW = "Low Risk"
NCD_RISK_RATING_MEDIUM = "Medium Risk"
NCD_RISK_RATING_HIGH = "High Risk"
NCD_RISK_RATING_VERY_HIGH = "Very High Risk"


# Helpers
def random_date(start, end) -> date:
    """
    This function will return a random date between two given dates
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def generate_population(num_samples: int, split: float, age_min: int, age_max: int, random_seed: int) -> list:
                        #instance: int, return_list: Manager) -> None:
    """
    Generate a random population of data based on the parameters provided.

    :param num_samples:
    :param split:
    :param age_min:
    :param age_max:
    :param random_seed:
    :param instance:
    :param return_list:
    :return:
    """

    samples = []
    date_start = date(2024, 1, 1)
    date_end = date(2024, 10, 4)

    hospitals = [
        {"name": "Hospital 1", "weight": 0.25, "lat": 0.0, "long": 0.0},
        {"name": "Hospital 2", "weight": 0.4, "lat": 0.0, "long": 0.0},
        {"name": "Hospital 3", "weight": 0.1, "lat": 0.0, "long": 0.0},
        {"name": "Hospital 4", "weight": 0.25, "lat": 0.0, "long": 0.0},
    ]

    # Random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)
    faker = Faker()
    faker.seed_instance(random_seed)

    for hosp in hospitals:

        for i in tqdm(range(int(num_samples * hosp["weight"]))):
            # Setup values per sample
            random_visits: int = random.randint(1, 7)  # Number of visits for the patient
            i_exercise = bool(random.getrandbits(1))
            i_gender = np.random.choice(["M", "F"], p=[split, 1-split]).item()
            i_height = faker.random_int(min=160, max=190)
            i_weight = faker.random_int(min=60, max=180)
            i_first_visit_date = random_date(date_start, date_end)
            hospital = hosp["name"]
            coe = np.random.choice(["CoE 1", "CoE 2", "CoE 3", "CoE 4"], p=[0.4, 0.17, 0.2, 0.23]).item()
            i_bmi = round(i_weight / ((i_height/100) ** 2), 2)
            bp_measurement = bias_bp_ranges()

            # Create record
            record = {
                "assessment_date": i_first_visit_date.isoformat(),
                "assessment_type": "assessment_date_0",
                "patient_id": faker.uuid4(),
                "gender": i_gender,
                "first_name": faker.first_name_male() if i_gender == "M" else Faker().first_name_female(),
                "last_name": faker.last_name(),
                "age": faker.random_int(min=age_min, max=age_max),
                "height": i_height,
                "weight": i_weight,
                "bmi_value": i_bmi,
                "bmi_model_value": i_bmi,
                "bmi_range": "",
                "bmi_range_model": "",
                "blood_pressure_systolic": bp_measurement["systolic"],
                "blood_pressure_diastolic": bp_measurement["systolic"],
                "blood_pressure_range": "",
                "is_smoker": bool(random.getrandbits(1)),
                "waist_circumference": round(np.random.normal(119.50, 40.56), 2),
                "diabetes": "",
                "diabetes_family_history": bool(random.getrandbits(1)),
                "stroke_parents_siblings_before_65": bool(random.getrandbits(1)),
                "hypertension_family_history": bool(random.getrandbits(1)),
                "exercise": i_exercise,
                "exercise_hours": faker.random_int(min=1, max=10) if i_exercise else 0,
                "diabetes_type": "",
                "diabetes_diagnosed": "",
                "ckd_diagnosed": "",
                "cvd_diagnosed": "",
                "hypertension_diagnosed": "",
                "hypertension_on_medication": "",
                "diabetes_on_medication": "",
                "ckd_on_medication": "",
                "cvd_on_medication": "",
                "cvd_risk_score": -1,
                "ckd_risk_score": -1,
                "hypertension_risk_score": -1,
                "diabetes_risk_score": -1,
                "overall_risk_score": -1,
                "ncd_risk_rating": "n/a",
                "coe": coe,
                "hospital": hospital,
                "hospital_lat": hosp["lat"],
                "hospital_long": hosp["long"],
                "number_of_assessments": random_visits,
                "visit_due_to_prior_risk_assessment": False,
                "consultation_fee": 0,
                "medication_fee_diabetes": 0,
                "medication_fee_hypertension": 0,
                "medication_fee_cvd": 0,
                "medication_fee_ckd": 0,
                "cvd_risk_rating": "",
                "ckd_risk_rating": "",
                "diabetes_risk_rating": "",
                "hypertension_risk_rating": "",
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
            record["ncd_risk_rating"] = _calc_ncd_risk_rating(record["overall_risk_score"])
            record["diabetes_risk_rating"] = _calc_ncd_risk_rating(record["diabetes_risk_score"])
            record["cvd_risk_rating"] = _calc_ncd_risk_rating(record["cvd_risk_score"])
            record["ckd_risk_rating"] = _calc_ncd_risk_rating(record["ckd_risk_score"])
            record["hypertension_risk_rating"] = _calc_ncd_risk_rating(record["hypertension_risk_score"])

            # Diagnosis
            record["hypertension_diagnosed"] = diagnosis_chance(record=record, ncd_name="hypertension")
            record["diabetes_diagnosed"] = diagnosis_chance(record=record, ncd_name="diabetes")
            record["diabetes"] = record["diabetes_diagnosed"]
            record["diabetes"] = record["diabetes_diagnosed"]
            record["diabetes_type"] = random.choice(["1", "2"]) if record["diabetes_diagnosed"] else ""
            record["ckd_diagnosed"] = diagnosis_chance(record=record, ncd_name="ckd")
            record["cvd_diagnosed"] = diagnosis_chance(record=record, ncd_name="cvd")


            # Ranges
            record["bmi_range"] = _calc_bmi_range(record["bmi_value"])
            record["bmi_model_range"] = _calc_bmi_range(record["bmi_model_value"])
            record["blood_pressure_range"] = _calc_blood_pressure_risk_rating(
                bp_systolic=record["blood_pressure_systolic"],
                bp_diastolic=record["blood_pressure_diastolic"]
            )

            # Medication
            record["hypertension_on_medication"] = _calc_hypertension_medication(
                hypertension_diagnosed=record["hypertension_diagnosed"],
                bp_range=record["blood_pressure_range"]
            )
            record["diabetes_on_medication"] = _calc_diabetes_medication(
                diabetes_diagnosed=record["diabetes_diagnosed"],
                diabetes_risk_rating=record["diabetes_risk_rating"]
            )
            record["cvd_on_medication"] = _calc_cvd_medication(
                cvd_diagnosed=record["cvd_diagnosed"],
                cvd_risk_rating=record["cvd_risk_rating"]
            )
            record["ckd_on_medication"] = _calc_ckd_medication(
                ckd_diagnosed=record["ckd_diagnosed"],
                ckd_risk_rating=record["ckd_risk_rating"]
            )

            prev_diagnosed = {
                "hypertension": record["hypertension_diagnosed"],
                "diabetes": record["diabetes_diagnosed"],
                "cvd": record["cvd_diagnosed"],
                "ckd": record["ckd_diagnosed"],
            }

            samples.append(deepcopy(record))
            visits_samples = sample_populate(
                record=record, prev_diagnosed=prev_diagnosed, visits=random_visits)
            samples.extend(deepcopy(visits_samples))

    return samples


def bias_bp_ranges() -> dict:
    """
    Returns a biased set of blood pressure values for systolic and diastolic
    :return:
    """

    #    if bp_systolic < 120 and bp_diastolic < 80:
    #        return BP_RATING_NORMAL
    #    elif 120 <= bp_systolic < 129 and bp_diastolic < 80:
    #        return BP_RATING_ELEVATED
    #    elif 130 <= bp_systolic < 139 or 80 <= bp_diastolic < 89:
    #        return BP_RATING_HYPERTENSION_STAGE1
    #    elif bp_systolic >= 140 or bp_diastolic >= 90:
    #        return BP_RATING_HYPERTENSION_STAGE2
    #    elif bp_systolic > 180 or bp_diastolic > 120:
    #        return BP_RATING_HYPERTENSION_CRISIS

    #    return BP_RATING_UNCLASSIFIED

    #thresholds = [0.05, 0.09, 0.28, 0.50, 0.08]
    thresholds = [0.48, 0.06, 0.01, 0.22, 0.13, 0.10]

    u = np.random.choice(thresholds, p=thresholds).item()
    if u <= thresholds[0]:
        NORMAL_RANGE_SYS = [110, 120]
        NORMAL_RANGE_DIA = [70, 80]

        systolic = random.randint(NORMAL_RANGE_SYS[0], NORMAL_RANGE_SYS[1])
        diastolic = random.randint(NORMAL_RANGE_DIA[0], NORMAL_RANGE_DIA[1])
        return {"systolic": systolic, "diastolic": diastolic}

    if u <= thresholds[1]:
        ELEVATED_RANGE_SYS = [120, 129]
        ELEVATED_RANGE_DIA = [70, 80]

        systolic = random.randint(ELEVATED_RANGE_SYS[0], ELEVATED_RANGE_SYS[1])
        diastolic = random.randint(ELEVATED_RANGE_DIA[0], ELEVATED_RANGE_DIA[1])
        return {"systolic": systolic, "diastolic": diastolic}

    if u <= thresholds[2]:
        HYP_S1_RANGE_SYS = [130, 139]
        HYP_S1_RANGE_DIA = [80, 89]

        systolic = random.randint(HYP_S1_RANGE_SYS[0], HYP_S1_RANGE_SYS[1])
        diastolic = random.randint(HYP_S1_RANGE_DIA[0], HYP_S1_RANGE_DIA[1])
        return {"systolic": systolic, "diastolic": diastolic}

    if u <= thresholds[3]:
        HYP_S2_RANGE_SYS = [140, 179]
        HYP_S2_RANGE_DIA = [90, 119]

        systolic = random.randint(HYP_S2_RANGE_SYS[0], HYP_S2_RANGE_SYS[1])
        diastolic = random.randint(HYP_S2_RANGE_DIA[0], HYP_S2_RANGE_DIA[1])
        return {"systolic": systolic, "diastolic": diastolic}

    if u <= thresholds[4]:
        HYP_S3_RANGE_SYS = [180, 199]
        HYP_S3_RANGE_DIA = [120, 139]

        systolic = random.randint(HYP_S3_RANGE_SYS[0], HYP_S3_RANGE_SYS[1])
        diastolic = random.randint(HYP_S3_RANGE_DIA[0], HYP_S3_RANGE_DIA[1])
        return {"systolic": systolic, "diastolic": diastolic}




def get_hr(record: dict) -> dict:
    raw_cvd_input = {
        "gender": record["gender"],
        "age": record["age"],
        "bmi_model": record["bmi_value"],
        "waist_circumference": record["waist_circumference"],
        "hypertension_diagnosed": record["hypertension_diagnosed"],
        "hypertension_medication": record["hypertension_on_medication"],
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
        "blood_pressure_systolic": record["blood_pressure_systolic"],
        "blood_pressure_diastolic": record["blood_pressure_diastolic"],
    }
    hyp_input = prepare_hypertension_input(raw_hyp_input)

    return estimate_health_risk(hyp_input, cvd_input, debug=False)


def diagnosis_chance(record: dict, ncd_name: str) -> bool:
    """
    Chance of diagnosis based on NCD rating.

    :param record: dict
    :param ncd_name: str
    :return:
    """
    if record[f"{ncd_name}_risk_rating"] in [NCD_RISK_RATING_VERY_HIGH, NCD_RISK_RATING_HIGH]:
        return np.random.choice([True, False], p=[0.13, 0.87]).item()

    if record[f"{ncd_name}_risk_rating"] in [NCD_RISK_RATING_MEDIUM]:
        return np.random.choice([True, False], p=[0.05, 0.95]).item()

    return False


def assign_fees_and_diagnosis(record: dict, ncd_name: str, diagnosis: bool) -> dict:
    """
    Assign diagnosis fee and medication fee for a given ncd.
    :param record: dict
    :param ncd_name: str
    :param diagnosis: bool
    :return: dict
    """

    record["consultation_fee"] = 65
    record[f"medication_fee_{ncd_name}"] = 75 if diagnosis else 0
    record["visit_due_to_prior_risk_assessment"] = True
    record[f"{ncd_name}_diagnosed"] = True if diagnosis else False
    record[f"{ncd_name}_on_medication"] = True if diagnosis else False

    return record


def clear_fees(record: dict, ncd_name: str) -> dict:
    """
    Clear diagnosis fee and medication fee for a given ncd.
    :param record:
    :param ncd_name:
    :return: dict
    """

    record["consultation_fee"] = 0
    record[f"medication_fee_{ncd_name}"] = 0
    record["visit_due_to_prior_risk_assessment"] = False

    return record

def sample_populate(record: dict, prev_diagnosed: dict, visits: int) -> list:
    """
    Create sample visits given an initial record
    :param record: dict
    :param visit_date: date
    :param prev_diagnosed: dict
    :param visits: int
    :return:
    """

    generated_samples = []
    org_record = record

    # We create the number of samples given by the visits parameter
    for idx in range(1, visits):
        visit_record = deepcopy(org_record)
        days_since_last_visit = np.random.choice(a=[28, 29, 30, 31, 32, 33],
                                                 p=[0.28, 0.17, 0.13, 0.23, 0.07, 0.12]).item()

        # Set new visit date
        visit_date = datetime.strptime(record["assessment_date"], "%Y-%m-%d").date() + timedelta(days=days_since_last_visit*idx)
        visit_record["assessment_date"] = visit_date.isoformat()
        visit_record["assessment_type"] = f"assessment_date_{idx}"

        for ncd_name in ["hypertension", "diabetes", "cvd", "ckd"]:

            # Patient was previously classified as "High Risk" or "Very High Risk"
            if record[f"{ncd_name}_risk_rating"] in [NCD_RISK_RATING_HIGH, NCD_RISK_RATING_VERY_HIGH]:

                # Patient has not been previously diagnosed for given NCD.
                if not prev_diagnosed[ncd_name]:
                    visit_record = assign_fees_and_diagnosis(
                        record=visit_record,
                        ncd_name=ncd_name,
                        diagnosis=True)
                # Patient has been previously diagnosed for given NCD
                else:
                    visit_record = clear_fees(record=visit_record, ncd_name=ncd_name)

                prev_diagnosed[ncd_name] = visit_record[f"{ncd_name}_diagnosed"]

            if any([
                visit_record["medication_fee_hypertension"],
                visit_record["medication_fee_diabetes"],
                visit_record["medication_fee_cvd"],
                visit_record["medication_fee_ckd"]
            ]):
                record["consultation_fee"] = 65

            generated_samples.append(visit_record)
            org_record = deepcopy(visit_record)

    return generated_samples


def _calc_hypertension_medication(hypertension_diagnosed: bool, bp_range: str) -> bool:
    """
    Calculate the likelihood of hypertension medication being taken based on patient being diagnosed
    and having a high risk rating.

    :param hypertension_diagnosed:
    :param bp_range:
    :return: bool
    """

    if bp_range == BP_RATING_HYPERTENSION_CRISIS:
        if hypertension_diagnosed:
            return np.random.choice([True, False], p=[0.11, 0.89]).item()
    elif bp_range == BP_RATING_HYPERTENSION_STAGE2:
        if hypertension_diagnosed:
            return np.random.choice([True, False], p=[0.11, 0.89]).item()
    elif bp_range == BP_RATING_HYPERTENSION_STAGE1:
        if hypertension_diagnosed:
            return np.random.choice([True, False], p=[0.01, 0.99]).item()
    #elif bp_range == BP_RATING_ELEVATED:
    #    if hypertension_diagnosed:
    #        return np.random.choice([True, False], p=[0.08, 0.92]).item()

    return False


def _calc_diabetes_medication(diabetes_diagnosed: bool, diabetes_risk_rating: str) -> bool:
    """
    Calculate diabetes medication based on diabetes diagnosis status and diabetes risk rating.
    :param diabetes_diagnosed:
    :param diabetes_risk_rating:
    :return:
    """
    if diabetes_diagnosed:
        return True

    # TODO: revisit this if needed
    #if not diabetes_diagnosed:
    #    if diabetes_risk_rating in [NCD_RISK_RATING_VERY_HIGH, NCD_RISK_RATING_HIGH]:
    #        return np.random.choice([True, False], p=[0.5, 0.5]).item()

    return False


def _calc_cvd_medication(cvd_diagnosed: bool, cvd_risk_rating: str) -> bool:
    """
    Calculate cvd medication based on cvd diagnosis status and cvd risk rating.
    :param cvd_diagnosed:
    :param cvd_risk_rating:
    :return:
    """
    if cvd_diagnosed:
        return True

    # TODO: revisit this if needed
    # if not cvd_diagnosed:
    #     if cvd_risk_rating in [NCD_RISK_RATING_VERY_HIGH, NCD_RISK_RATING_HIGH]:
    #         return np.random.choice([True, False], p=[0.5, 0.5]).item()
    #     if cvd_risk_rating in [NCD_RISK_RATING_MEDIUM]:
    #         return np.random.choice([True, False], p=[0.8, 0.2]).item()

    return False


def _calc_ckd_medication(ckd_diagnosed: bool, ckd_risk_rating: str) -> bool:
    """
    Calculate cvd medication based on ckd diagnosis status and ckd risk rating.
    :param ckd_diagnosed:
    :param ckd_risk_rating:
    :return:
    """
    if ckd_diagnosed:
        return True

    return False


def _calc_ncd_risk_rating(overall_risk_score: float) -> str:
    """
    Calculate the NCD risk rating based on the overall risk score.

    :param overall_risk_score:
    :return:
    """
    if overall_risk_score < 10:
        return NCD_RISK_RATING_LOW
    elif 10 <= overall_risk_score < 20:
        return NCD_RISK_RATING_MEDIUM
    elif 20 <= overall_risk_score < 30:
        return NCD_RISK_RATING_HIGH
    # > 30
    return NCD_RISK_RATING_VERY_HIGH


def _calc_bmi_range(bmi: float) -> str:
    """
    Calculate the BMI risk rating based on the bmi value.
    :param bmi:
    :return:
    """

    if bmi < 18.5:
        return BMI_RATING_UNDERWEIGHT
    elif 18.5 <= bmi < 25:
        return BMI_RATING_NORMAL
    elif 25 <= bmi < 27:
        return BMI_RATING_OVERWEIGHT
    # > 27
    return BMI_RATING_OBESE

def _calc_blood_pressure_risk_rating(bp_systolic: float, bp_diastolic: float) -> str:
    """
    Calculate blood pressure risk rating based on the blood pressure systolic and diastolic.
    :param bp_systolic:
    :param bp_diastolic:
    :return: str
    """

    if bp_systolic < 120 and bp_diastolic < 80:
        return BP_RATING_NORMAL
    elif 120 <= bp_systolic < 129 and bp_diastolic < 80:
        return BP_RATING_ELEVATED
    elif 130 <= bp_systolic < 139 or 80 <= bp_diastolic < 89:
        return BP_RATING_HYPERTENSION_STAGE1
    elif bp_systolic >= 140 or bp_diastolic >= 90:
        return BP_RATING_HYPERTENSION_STAGE2
    elif bp_systolic > 180 or bp_diastolic > 120:
        return BP_RATING_HYPERTENSION_CRISIS

    return BP_RATING_UNCLASSIFIED


def do_the_do():
    #manager = Manager()
    #result_list = manager.list()

    cores = 4
    total_samples = 15_000
    processes = []

    #for i in range(cores):
    #    n_samples = total_samples // cores
    #    p = Process(target=generate_population, args=(n_samples, 0.41, 25, 85, 42 + i, i-1, result_list))
    #    processes.append(p)
    #    p.start()

    #for p in processes:
    #    p.join()

    #pop = result_list
    #pop = [vals for result in result_list for vals in result]
    pop = generate_population( total_samples, 0.45, 25, 85, 42)

    #with open("sample_data.json", "w") as f:
    #    json.dump(pop, f)

    df = pd.DataFrame.from_records(list(pop))
    df.drop_duplicates(inplace=True)
    print(df.head(5))
    df.to_csv(index=False, path_or_buf='sample_data.csv', quoting=csv.QUOTE_NONNUMERIC, doublequote=True)


    #keys = pop[0].keys()
    #with open("sample_data.csv", "w") as csvfile:
    #    dict_writer = csv.DictWriter(csvfile, keys, quoting=csv.QUOTE_NONNUMERIC, doublequote=True)
    #    dict_writer.writeheader()
    #    dict_writer.writerows(pop)

do_the_do()

# Average heights by country
# https://en.wikipedia.org/wirecord.py:136ki/Average_human_height_by_country

# Human weight
# https://en.wikipedia.org/wiki/Human_body_weight