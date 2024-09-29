# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

class Patient:
    patient_id
    tenant
    client_timestamp
    server_timestamp
    enumerator_id
    first_name
    last_name
    age
    height
    weight
    gender
    follow_up_trial_participation
    covid_diagnose
    vaccinated_for_covid
    bmi_model
    is_smoker
    waist_circumference
    diabetes_currently
    diabetes_family_history
    diabetes_on_medication
    diabetes_type
    ckd_diagnosed
    high_cholesterol_on_medication
    heart_attack_diagnosed
    heart_bypass_surgery_performed
    chest_pain_due_to_heart_disease_diagnosed
    stroke_diagnosed
    heart_failure_diagnosed
    leg_muscle_pain_caused_by_narrowed_arteries_diagnosed
    artery_disease_treated
    heart_arteries_blocked_diagnosed
    stroke_parents_siblings_before_65
    heart_attack_parents_siblings_before_65
    heart_rate_user_entered
    blood_oxygenation_user_entered
    visit_for_back_pain
    visit_for_covid19_vaccination
    visit_for_confusion
    visit_for_coughing
    visit_for_diarrhea
    visit_for_fatigue
    visit_for_fever
    visit_for_headache
    visit_for_join pain_discomfort
    visit_for_nausea
    visit_for_nosebleeds
    visit_for_shortness_of_breath_during_activities
    visit_for_stomach_ache
    visit_for_vision_changes
    visit_for_vomiting
    visit_for_other_reason
    visit_for_other_reason_text
    cvd_risk_score
    ckd_risk_score
    diabetes_risk_score
    hypertension_risk_score
    blood_pressure_systolic
    blood_pressure_diastolic
    blood_pressure_finger_systolic
    blood_pressure_finger_diastolic
    confidence_level
    hypertension_diagnosed
    hypertension_on_medication
    hypertension_family_history
    smoking_frequency
    heart_rate
    exercise_hours
    coffee_frequency
    alcohol_frequency
    forget_medication


fields = {
    "id":  "id",
    "first_name": "string",
    "last_name": "string",
    "age": "25 - 85",
    "gender": "m/f",
    "height": 120,
    "weight": 90,
    "latitude": 123.00,
    "longitude": 123.00,
    "bmi_range": 10,
    "bmi_value": 15.1,
    "is_smoker": True,
    "is_former_smoker": True,
    "cvd_risk_score": 13,
    "diabetes_risk_score": 15,
    "diabetes_currently": True,
    "diabetes_family_history": True,
    "diabetes_type": "type1 or type2",
    "ckd_diagnosed": True,
    
}

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
