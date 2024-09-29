"""Cardiometabolic disease risk estimation.

Note that this model predicts CVD, Type 2 Diabetes, and the composite risk of both.

Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3308277/
"""
from dataclasses import dataclass

from lib.hr_expected_types import CvdFields

from .lookup import RangeLookup
from .util import strict_bool
from .waist_circumference import estimate_waist_circumference

CVD_RISK_POINTS_MALE_AGE = RangeLookup(
    {
        (0, 45): 0,
        (45, 50): 13,
        (50, 55): 17,
        (55, 60): 22,
        (60, 65): 33,
        (65, 70): 37,
        (70, 75): 46,
        (75, 999): 61,
    }
)

CVD_RISK_POINTS_FEMALE_AGE = RangeLookup(
    {
        (0, 45): 0,
        (45, 50): 10,
        (50, 55): 16,
        (55, 60): 23,
        (60, 65): 29,
        (65, 70): 37,
        (70, 75): 49,
        (75, 999): 60,
    }
)

CVD_RISK_POINTS_MALE_BMI = RangeLookup({(0, 25): 0, (25, 30): 4, (30, 999): 12})
CVD_RISK_POINTS_MALE_WAIST = RangeLookup({(0, 94): 0, (94, 999): 3})
CVD_RISK_POINTS_MALE_ANTIHYPERTENSIVES = 10
CVD_RISK_POINTS_MALE_CURRENT_SMOKER = 9
CVD_RISK_POINTS_MALE_FAMILY_MI_OR_STROKE = 1
CVD_RISK_POINTS_MALE_FAMILY_DIABETES = 4

CVD_RISK_POINTS_FEMALE_BMI = RangeLookup({(0, 25): 0, (25, 30): 4, (30, 999): 7})
CVD_RISK_POINTS_FEMALE_WAIST = RangeLookup({(0, 80): 0, (80, 88): 2, (88, 999): 6})
CVD_RISK_POINTS_FEMALE_ANTIHYPERTENSIVES = 11
CVD_RISK_POINTS_FEMALE_CURRENT_SMOKER = 9
CVD_RISK_POINTS_FEMALE_FAMILY_MI_OR_STROKE = 4
CVD_RISK_POINTS_FEMALE_FAMILY_DIABETES = 3


MALE_CARDIOMETABOLIC_RISK = RangeLookup(
    {
        (0, 25): {"composite": 6.9, "cvd": 3.0, "diabetes": 3.5, "ckd": 0.5},
        (25, 30): {"composite": 11.5, "cvd": 4.0, "diabetes": 6.0, "ckd": 2.0},
        (30, 35): {"composite": 21.8, "cvd": 9.6, "diabetes": 8.6, "ckd": 2.8},
        (35, 40): {"composite": 32.6, "cvd": 15.6, "diabetes": 10.1, "ckd": 9.9},
        (40, 45): {"composite": 34.6, "cvd": 19.2, "diabetes": 11.7, "ckd": 10.8},
        (45, 50): {"composite": 44.0, "cvd": 23.0, "diabetes": 15.3, "ckd": 14.3},
        (50, 55): {"composite": 51.2, "cvd": 25.1, "diabetes": 20.6, "ckd": 19.1},
        (55, 60): {"composite": 54.5, "cvd": 27.7, "diabetes": 14.3, "ckd": 23.6},
        (60, 101): {"composite": 76.2, "cvd": 41.5, "diabetes": 22.3, "ckd": 44.6},
    }
)

FEMALE_CARDIOMETABOLIC_RISK = RangeLookup(
    {
        (0, 25): {"composite": 3.7, "cvd": 2.1, "diabetes": 0.4, "ckd": 1.3},
        (25, 30): {"composite": 13.6, "cvd": 6.1, "diabetes": 3.2, "ckd": 4.7},
        (30, 35): {"composite": 17.9, "cvd": 6.3, "diabetes": 7.5, "ckd": 4.1},
        (35, 40): {"composite": 19.3, "cvd": 7.1, "diabetes": 6.0, "ckd": 6.8},
        (40, 45): {"composite": 26.3, "cvd": 9.5, "diabetes": 9.8, "ckd": 8.8},
        (45, 50): {"composite": 35.3, "cvd": 13.5, "diabetes": 13.1, "ckd": 12.6},
        (50, 55): {"composite": 37.6, "cvd": 13.1, "diabetes": 17.4, "ckd": 11.8},
        (55, 60): {"composite": 49.2, "cvd": 18.9, "diabetes": 16.5, "ckd": 24.9},
        (60, 101): {"composite": 72.3, "cvd": 35.2, "diabetes": 20.5, "ckd": 42.8},
    }
)


def estimate_cardiometabolic_risk(questions):
    answers = extract_answers(questions)
    if answers.gender == "M":
        abs_score, risk = _male_cardiometabolic_risk(answers)
    elif answers.gender == "F":
        abs_score, risk = _female_cardiometabolic_risk(answers)
    else:
        raise ValueError("Gender must be 'M' or 'F'")

    return abs_score, risk


def is_smoker(questions):
    is_smoker = None
    if CvdFields.is_smoker in questions:
        is_smoker = strict_bool(questions[CvdFields.is_smoker])
    elif (
        CvdFields.smoking_frequency in questions
    ):  # use this to know if it came from old or new questionnaire
        is_smoker = questions[CvdFields.smoking_frequency].lower() in (
            "everyday",
            "few_times_a_week",
        )

    return is_smoker


@dataclass
class CardioMetabolicRelatedAnswers:
    gender: str
    age: float
    bmi: float
    waist_circumference: float
    antihypertensives: bool
    current_smoker: bool
    family_mi_or_stroke: bool
    family_diabetes: bool


def extract_answers(questions):
    gender = questions.get("gender")
    age = float(questions.get("age"))
    bmi = float(questions.get("bmi_model"))
    waist_circumference = estimate_waist_circumference(gender, age, bmi)
    antihypertensives = strict_bool(questions.get("hypertension_on_medication"))
    current_smoker = is_smoker(questions)

    if "stroke_parents_siblings_before_65" in questions:
        family_mi_or_stroke = strict_bool(
            questions.get("stroke_parents_siblings_before_65")
        )
    else:
        family_mi_or_stroke = strict_bool(questions.get("mi_or_stroke_family_history"))

    family_diabetes = strict_bool(questions.get("diabetes_family_history"))

    return CardioMetabolicRelatedAnswers(
        gender,
        age,
        bmi,
        waist_circumference,
        antihypertensives,
        current_smoker,
        family_mi_or_stroke,
        family_diabetes,
    )


def _male_cardiometabolic_risk(answers: CardioMetabolicRelatedAnswers):
    score = 0

    score += CVD_RISK_POINTS_MALE_AGE[answers.age]
    score += CVD_RISK_POINTS_MALE_BMI[answers.bmi]
    score += CVD_RISK_POINTS_MALE_WAIST[answers.waist_circumference]

    if answers.antihypertensives:
        score += CVD_RISK_POINTS_MALE_ANTIHYPERTENSIVES

    if answers.current_smoker:
        score += CVD_RISK_POINTS_MALE_CURRENT_SMOKER

    if answers.family_mi_or_stroke:
        score += CVD_RISK_POINTS_MALE_FAMILY_MI_OR_STROKE

    if answers.family_diabetes:
        score += CVD_RISK_POINTS_MALE_FAMILY_DIABETES

    return score, MALE_CARDIOMETABOLIC_RISK[score]


def _female_cardiometabolic_risk(answers: CardioMetabolicRelatedAnswers):
    score = 0

    score += CVD_RISK_POINTS_FEMALE_AGE[answers.age]
    score += CVD_RISK_POINTS_FEMALE_BMI[answers.bmi]
    score += CVD_RISK_POINTS_FEMALE_WAIST[answers.waist_circumference]

    if answers.antihypertensives:
        score += CVD_RISK_POINTS_FEMALE_ANTIHYPERTENSIVES

    if answers.current_smoker:
        score += CVD_RISK_POINTS_FEMALE_CURRENT_SMOKER

    if answers.family_mi_or_stroke:
        score += CVD_RISK_POINTS_FEMALE_FAMILY_MI_OR_STROKE

    if answers.family_diabetes:
        score += CVD_RISK_POINTS_FEMALE_FAMILY_DIABETES

    return score, FEMALE_CARDIOMETABOLIC_RISK[score]
