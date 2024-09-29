"""Waist circumference estimation from BMI.

This is a simple linear regression model where the first coefficient is the bias.

Reference:
https://www.researchgate.net/publication/230617989_Predicting_waist_circumference_from_body_mass_index
"""

MALE_COEF = [22.61306, 2.520738, 0.1583812]
FEMALE_COEF = [28.81919, 2.218007, -3.688953, 0.125975]


def _dot(x, y):
    return sum([x_ * y_ for x_, y_ in zip(x, y)])


def estimate_waist_circumference(gender, age, bmi):
    if gender == "M":
        x = [1, bmi, age]
        return _dot(MALE_COEF, x)
    elif gender == "F":
        # Extra indicator variable required for females
        is_over_35 = int(age >= 35)

        x = [1, bmi, is_over_35, age * is_over_35]
        return _dot(FEMALE_COEF, x)
    else:
        raise ValueError("Gender must be 'M' or 'F'")
