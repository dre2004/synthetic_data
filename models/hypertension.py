"""Prediction of Hypertension risk.

Reference: https://onlinelibrary.wiley.com/doi/full/10.1111/j.1751-7176.2010.00343.x
"""

from lib.hr_expected_types import HypertensionFields

from .lookup import RangeLookup
from .util import strict_bool


class BMIMissing(Exception):
    pass


class ExerciseError(Exception):
    pass


HT_RISK_POINTS_AGE = RangeLookup(
    {(0, 45): 0, (45, 55): 0, (55, 65): 2, (65, 75): 3, (75, 999): 4}
)

HT_RISK_POINTS_BMI = RangeLookup({(0, 25): 0, (25, 30): 1, (30, 40): 2, (40, 999): 3})

HT_RISK_SYSTOLIC_BP = RangeLookup(
    {
        (0, 110): 0,
        (110, 115): 2,
        (115, 120): 3,
        (120, 125): 4,
        (125, 130): 6,
        (130, 135): 8,
        (135, 999): 14,
    }
)

# Diastolic blood pressure risk points depend on age as well
HT_RISK_POINTS_DIASTOLIC_BP_AGE = RangeLookup(
    {
        (0, 55): RangeLookup({(0, 70): 0, (70, 80): 2, (80, 999): 3}),
        (55, 65): RangeLookup({(0, 70): 0, (70, 80): -1, (80, 999): -1}),
        (65, 75): RangeLookup({(0, 70): 0, (70, 80): -2, (80, 999): -3}),
        (75, 999): RangeLookup({(0, 70): 0, (70, 80): -1, (80, 999): -2}),
    }
)

HT_RISK_POINTS_FEMALE = 1
HT_RISK_POINTS_SMOKER = 1
HT_RISK_POINTS_NO_EXERCISE = 1
HT_RISK_POINTS_FAMILY_HYPERTENSION = 1
HT_RISK_POINTS_DIABETES = 2

HT_SIX_YEAR_RISK = [
    # fmt: off
    5.44, 5.63, 6.52, 8.05, 9.5, 11.21, 14.41, 17.94, 22.29, 26.83, 31.73, 36.63, 40.21,
    43.73, 47.93, 52.93, 53.36, 48.56, 50.67, 54.23, 59.11, 64.18, 65.97, 70.51
    # fmt: on
]


def _hypertension_score(
    age: int,
    is_smoker: bool,
    gender: int,
    exercise: bool,
    family_hypertension: bool,
    bmi_model: float,
    has_diabetes: bool,
    systolic_bp: float,
    diastolic_bp: float,
):

    # print(f"age: {age}")
    # print(f"is_smoker: {is_smoker}")
    # print(f"gender: {gender}")
    # print(f"exercise: {exercise}")
    # print(f"family_hypertension: {family_hypertension}")
    # print(f"bmi_model: {bmi_model}")
    # print(f"has_diabetes: {has_diabetes}")
    # print(f"systolic_bp: {systolic_bp}")
    # print(f"daiastolic_bp: {diastolic_bp}")
    score = 0

    score += HT_RISK_POINTS_AGE[age]
    score += HT_RISK_POINTS_BMI[bmi_model]
    score += HT_RISK_SYSTOLIC_BP[systolic_bp]

    # Diastolic is 2-tiered
    diastolic_range_lookup = HT_RISK_POINTS_DIASTOLIC_BP_AGE[age]
    score += diastolic_range_lookup[diastolic_bp]

    if is_smoker:
        score += HT_RISK_POINTS_SMOKER

    if has_diabetes:
        score += HT_RISK_POINTS_DIABETES

    if gender == "F":
        score += HT_RISK_POINTS_FEMALE

    if not exercise:
        score += HT_RISK_POINTS_NO_EXERCISE

    if family_hypertension:
        score += HT_RISK_POINTS_FAMILY_HYPERTENSION

    return score


def estimate_hypertension_risk(questions):
    # Error handling
    assert "age" in questions.keys(), "missing age field"
    assert "is_smoker" in questions.keys(), "missing is_smoker field"
    assert "gender" in questions.keys(), "missing gender field"
    assert "exercise" in questions.keys(), "missing gender field"
    assert "bmi_model" in questions.keys(), "missing bmi_model field"
    assert "diabetes_currently" in questions.keys(), "missing has_diabetes field"
    assert "blood_pressure_systolic" in questions.keys(), "missing systolic_bp field"
    assert "blood_pressure_diastolic" in questions.keys(), "missing diastolic_bp field"


    age = int(questions.get(HypertensionFields.age))

    is_smoker = None
    exercise = None


    if HypertensionFields.is_smoker in questions:
        is_smoker = strict_bool(questions.get(HypertensionFields.is_smoker))
    elif HypertensionFields.smoking_frequency in questions:
        # use this to know if it came from old or new questionnaire
        is_smoker = questions[HypertensionFields.smoking_frequency].lower() in (
            "everyday",
            "few_times_a_week",
        )

    if HypertensionFields.exercise_hours in questions:
        exercise = int(questions[HypertensionFields.exercise_hours]) > 0
    else:
        exercise = strict_bool(questions.get(HypertensionFields.exercise))
        if not exercise:
            raise ExerciseError

    gender = questions.get(HypertensionFields.gender)

    family_hypertension = strict_bool(
        questions.get(HypertensionFields.ht_family_history)
    )

    if not questions[HypertensionFields.bmi_model]:
        raise BMIMissing

    bmi = float(questions[HypertensionFields.bmi_model])
    has_diabetes = strict_bool(questions.get(HypertensionFields.diabetes))

    systolic_bp = float(questions.get(HypertensionFields.bp_systolic))
    diastolic_bp = float(questions.get(HypertensionFields.bp_diastolic))

    score = _hypertension_score(
        age,
        is_smoker,
        gender,
        exercise,
        family_hypertension,
        bmi,
        has_diabetes,
        systolic_bp,
        diastolic_bp,
    )

    return score, HT_SIX_YEAR_RISK[min(score, len(HT_SIX_YEAR_RISK) - 1)]
