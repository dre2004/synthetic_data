from typing import List

from lib.expected_type import BasicExpectedType as BType
from lib.expected_type import ConstrainedExpectedType as CType
from lib.expected_type import Optional as Opt
from lib.expected_type import check_dict_against_fields

BOOL_VALS = ("yes", "no", True, False)


class HrCommonFields:
    id = "id"
    tenant = "tenant"
    risk_model = "risk_model"


hr_common_fields = (
    BType(HrCommonFields.id, str),
    BType(HrCommonFields.tenant, str),
    CType(HrCommonFields.risk_model, str, ("hypertension", "cvd")),
)


class HypertensionFields:
    age = "age"
    is_smoker = "is_smoker"
    gender = "gender"
    exercise = "exercise"
    bmi_model = "bmi_model"
    diabetes = "diabetes_currently"
    bp_systolic = "blood_pressure_systolic"
    bp_diastolic = "blood_pressure_diastolic"
    ht_diagnosed = "hypertension_diagnosed"
    ht_medication = "hypertension_on_medication"
    ht_family_history = "hypertension_family_history"
    smoking_frequency = "smoking_frequency"
    exercise_hours = "exercise_hours"
    coffee_frequency = "coffee_frequency"
    alcohol_frequency = "alcohol_frequency"
    bp_finger_systolic = "blood_pressure_finger_systolic"
    bp_finger_diastolic = "blood_pressure_finger_diastolic"
    forget_medication = "forget_medication"
    heart_rate_user_entered = "heart_rate_user_entered"
    blood_oxygenation_user_entered = "blood_oxygenation_user_entered"
    fruit_vegetable_consumption = "fruit_vegetable_consumption"


class CvdFields:
    age = "age"
    is_smoker = "is_smoker"
    smoking_frequency = "smoking_frequency"
    gender = "gender"
    bmi_model = "bmi_model"
    waist_circumference = "waist_circumference"
    ht_medication = "hypertension_on_medication"
    mi_or_stroke_family = "mi_or_stroke_family_history"
    diabetes_family_history = "diabetes_family_history"
    stroke_family_65 = "stroke_parents_siblings_before_65"
    heart_attack_family_65 = "heart_attack_parents_siblings_before_65"
    diabetes = "diabetes_currently"
    diabetes_medication = "diabetes_on_medication"
    diabetes_type = "diabetes_type"
    ckd_diagnosed = "ckd_diagnosed"
    hc_medication = "high_cholesterol_on_medication"
    heart_attack_diagnosed = "heart_attack_diagnosed"
    heart_bypass_surg = "heart_bypass_surgery_performed"
    chest_pain_heart_disease = "chest_pain_due_to_heart_disease_diagnosed"
    stroke_diagnosed = "stroke_diagnosed"
    heart_failure_diagnosed = "heart_failure_diagnosed"
    leg_pain_narrow_arteries = "leg_muscle_pain_caused_by_narrowed_arteries_diagnosed"
    artery_disease_treated = "artery_disease_treated"
    heart_blocked_diagnosed = "heart_arteries_blocked_diagnosed"
    reason_for_puskemas_visit = "reason_for_puskemas_visit"
    medical_history = "medical_history"


hr_risk_model_fields = {
    "hypertension": (
        Opt(BType(HypertensionFields.age, int)),
        Opt(CType(HypertensionFields.is_smoker, (str, bool), BOOL_VALS)),
        Opt(CType(HypertensionFields.gender, str, ("M", "F"))),
        Opt(CType(HypertensionFields.exercise, (str, bool), BOOL_VALS)),
        Opt(BType(HypertensionFields.bmi_model, (float, int))),
        Opt(CType(HypertensionFields.diabetes, (str, bool), (*BOOL_VALS, "used_to"))),
        Opt(BType(HypertensionFields.bp_systolic, int)),
        Opt(BType(HypertensionFields.bp_diastolic, int)),
        Opt(
            CType(HypertensionFields.ht_diagnosed, (str, bool), (*BOOL_VALS, "used_to"))
        ),
        Opt(
            CType(
                HypertensionFields.ht_medication,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(
            CType(
                HypertensionFields.ht_family_history,
                (str, bool),
                (*BOOL_VALS, "do_not_know"),
            )
        ),
        Opt(
            CType(
                HypertensionFields.smoking_frequency,
                str,
                ("everyday", "few_times_a_week", "used_to", "no"),
            ),
        ),
        Opt(BType(HypertensionFields.exercise_hours, int)),
        Opt(BType(HypertensionFields.coffee_frequency, int)),
        Opt(BType(HypertensionFields.alcohol_frequency, int)),
        Opt(BType(HypertensionFields.bp_finger_systolic, int)),
        Opt(BType(HypertensionFields.bp_finger_diastolic, int)),
        Opt(BType(HypertensionFields.forget_medication, int)),
        Opt(BType(HypertensionFields.heart_rate_user_entered, int)),
        Opt(BType(HypertensionFields.blood_oxygenation_user_entered, int)),
        Opt(BType(HypertensionFields.fruit_vegetable_consumption, int)),
    ),
    "cvd": (
        Opt(BType(CvdFields.age, int)),
        Opt(CType(CvdFields.is_smoker, (str, bool), BOOL_VALS)),
        Opt(
            CType(
                CvdFields.smoking_frequency,
                str,
                ("everyday", "few_times_a_week", "used_to", "no"),
            ),
        ),
        Opt(CType(CvdFields.gender, str, ("M", "F"))),
        Opt(BType(CvdFields.bmi_model, (float, int))),
        Opt(BType(CvdFields.waist_circumference, (float, int))),
        Opt(CType(CvdFields.ht_medication, (str, bool), BOOL_VALS)),
        Opt(
            CType(
                CvdFields.mi_or_stroke_family, (str, bool), (*BOOL_VALS, "do_not_know")
            )
        ),
        Opt(
            CType(
                CvdFields.diabetes_family_history,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(
            CType(CvdFields.stroke_family_65, (str, bool), (*BOOL_VALS, "do_not_know"))
        ),
        Opt(
            CType(
                CvdFields.heart_attack_family_65,
                (str, bool),
                (*BOOL_VALS, "do_not_know"),
            )
        ),
        Opt(CType(CvdFields.diabetes, (str, bool), (*BOOL_VALS, "used_to"))),
        Opt(CType(CvdFields.diabetes_medication, (str, bool), BOOL_VALS)),
        Opt(
            CType(
                CvdFields.diabetes_type,
                str,
                ("type_1", "type_2", "gestational", "none", "no"),
            )
        ),
        Opt(CType(CvdFields.ckd_diagnosed, (str, bool), BOOL_VALS)),
        Opt(CType(CvdFields.hc_medication, (str, bool), BOOL_VALS)),
        Opt(
            CType(
                CvdFields.heart_attack_diagnosed,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(CType(CvdFields.heart_bypass_surg, (str, bool), BOOL_VALS)),
        Opt(
            CType(
                CvdFields.chest_pain_heart_disease,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(CType(CvdFields.stroke_diagnosed, (str, bool), BOOL_VALS)),
        Opt(
            CType(
                CvdFields.heart_failure_diagnosed,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(
            CType(
                CvdFields.leg_pain_narrow_arteries,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(
            CType(
                CvdFields.artery_disease_treated,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(
            CType(
                CvdFields.heart_blocked_diagnosed,
                (str, bool),
                BOOL_VALS,
            )
        ),
        Opt(BType(CvdFields.reason_for_puskemas_visit, list)),
        Opt(BType(CvdFields.medical_history, list)),
    ),
}


def __check_health_risk_dict_common_fields(d: dict):
    return check_dict_against_fields(d, hr_common_fields)


def check_health_risk_dict(d: dict):
    errors = __check_health_risk_dict_common_fields(d)
    if errors is None:
        errors = check_dict_against_fields(
            d, hr_risk_model_fields.get(d[HrCommonFields.risk_model])
        )

    return errors


def extract_data_dict_from_overall_dict(hr: dict):
    extracted = hr.copy()

    for field in hr_common_fields:
        if field.key in extracted:
            del extracted[field.key]

    return extracted
