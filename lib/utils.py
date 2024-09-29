import json
import pandas as pd

# import tensorflow as tf
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.layers import Layer
#
#
# class Normalization2D(Layer):
#     """
#
#     ### `Normalization2D`
#
#     `kapre.utils.Normalization2D`
#
#     A layer that normalises input data in ``axis`` axis.
#
#     #### Parameters
#
#     * input_shape: tuple of ints
#         - E.g., ``(None, n_ch, n_row, n_col)`` if theano.
#
#     * str_axis: str
#         - used ONLY IF ``int_axis`` is ``None``.
#         - ``'batch'``, ``'data_sample'``, ``'channel'``, ``'freq'``, ``'time')``
#         - Even though it is optional, actually it is recommended to use
#         - ``str_axis`` over ``int_axis`` because it provides more meaningful
#         - and image data format-robust interface.
#
#     * int_axis: int
#         - axis index that along which mean/std is computed.
#         - `0` for per data sample, `-1` for per batch.
#         - `1`, `2`, `3` for channel, row, col (if channels_first)
#         - if `int_axis is None`, ``str_axis`` SHOULD BE set.
#
#     #### Example
#
#     A frequency-axis normalization after a spectrogram::
#         ```python
#         model.add(Spectrogram())
#         model.add(Normalization2D(str_axis='freq'))
#         ```
#     """
#
#     def __init__(self, str_axis=None, int_axis=None, image_data_format='default', eps=1e-10, **kwargs):
#         assert not (int_axis is None and str_axis is None), \
#             'In Normalization2D, int_axis or str_axis should be specified.'
#
#         assert image_data_format in ('channels_first', 'channels_last', 'default'), \
#             'Incorrect image_data_format: {}'.format(image_data_format)
#
#         if image_data_format == 'default':
#             self.image_data_format = K.image_data_format()
#         else:
#             self.image_data_format = image_data_format
#
#         self.str_axis = str_axis
#         if self.str_axis is None:  # use int_axis
#             self.int_axis = int_axis
#         else:  # use str_axis
#             # warning
#             if int_axis is not None:
#                 print('int_axis={} passed but is ignored, str_axis is used instead.'.format(int_axis))
#             # do the work
#             assert str_axis in ('batch', 'data_sample', 'channel', 'freq', 'time'), \
#                 'Incorrect str_axis: {}'.format(str_axis)
#             if str_axis == 'batch':
#                 int_axis = -1
#             else:
#                 if self.image_data_format == 'channels_first':
#                     int_axis = ['data_sample', 'channel', 'freq', 'time'].index(str_axis)
#                 else:
#                     int_axis = ['data_sample', 'freq', 'time', 'channel'].index(str_axis)
#
#         assert int_axis in (-1, 0, 1, 2, 3), 'invalid int_axis: ' + str(int_axis)
#         self.axis = int_axis
#         self.eps = eps
#         super(Normalization2D, self).__init__(**kwargs)
#
#     def call(self, x, mask=None):
#         if self.axis == -1:
#             mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
#             std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
#         elif self.axis in (0, 1, 2, 3):
#             all_dims = [0, 1, 2, 3]
#             del all_dims[self.axis]
#             mean = K.mean(x, axis=all_dims, keepdims=True)
#             std = K.std(x, axis=all_dims, keepdims=True)
#         return (x - mean) / (std + self.eps)
#
#     def get_config(self):
#         config = {'int_axis': self.axis,
#                   'str_axis': self.str_axis,
#                   'image_data_format': self.image_data_format}
#         base_config = super(Normalization2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def prepare_blood_pressure_data(user_data_file: str, bp_file: str, debug: bool = False) -> list:
    """
    Prepare the blood pressure inference data so that it can be merged back with the user record data.

    :param user_data_file: (str) the file path of the user_data JSON.
    :param bp_file: (str) the file path of the blood pressure inference results.
    :param debug: (bool) Whether print debug information, defaults to false.

    :returns: list

    """

    try:
        with open(user_data_file, 'r') as fp:
            user_data = json.load(fp)
        with open(bp_file, 'r') as fp:
            bp_video_inference = json.load(fp)
    except Exception as e:
        raise f"Exception raised while loading data files: {e}"

    for reading in bp_video_inference:
        for user in user_data:
            if user['user_id'] == reading['id']:
                try:
                    for v in range(0, 10):
                        if reading['video_filename'] in user[f"video_{v}"]:
                            user[f"measurement_{v}_systolic_estimate"] = round(reading['systolic_estimate'], 2)
                            user[f"measurement_{v}_diastolic_estimate"] = round(reading['diastolic_estimate'], 2)
                            user[f"measurement_{v}_video"] = reading['video_filename']
                            continue
                except TypeError as e:
                    if debug:
                        print(f"E: {e} - video {user['user_id']} - reading: {reading}")
                    continue

    return user_data


def __who_rating(row, column):
    if row[column] < 5.0:
        return 'LOW'
    elif 10.0 > row[column] > 5.0:
        return 'MEDIUM'
    elif 20.0 > row[column] > 10.0:
        return 'HIGH'
    elif row[column] >= 30.0:
        return 'VERY HIGH'
    return 'UNDEFINED'


def add_WHO_risk_ratings(user_data_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add the WHO (world health organisation) risk ratings for the given risk model results.

    :param user_data_dataframe:
    :return: pd.DataFrame
    """

    if not isinstance(user_data_dataframe, pd.DataFrame):
        raise TypeError('user_data_dataframe is {type(user_data_dataframe)} expected {type(pd.Dataframe)}')

    # WHO ratings for risk scores
    for metric in [
        'calculated_bmi_cvd_risk_score',
        'calculated_bmi_hypertension_risk_score',
        'calculated_bmi_diabetes_risk_score',
        'calculated_bmi_overall_risk_score',
        'bmi_model_cvd_risk_score',
        'bmi_model_hypertension_risk_score',
        'bmi_model_diabetes_risk_score',
        'bmi_model_overall_risk_score',
        # 'max_bp_model_cvd_risk_score',
        # 'max_bp_model_hypertension_risk_score',
        # 'max_bp_model_diabetes_risk_score',
        # 'max_bp_model_overall_risk_score',
    ]:
        user_data_dataframe[f"{metric}_who_rating"] = user_data_dataframe.apply(
            lambda row: __who_rating(row, metric), axis=1)

    return user_data_dataframe


def calculate_bmi(weight, height):
    """
    Calculates the body mass index (BMI) for a given height and weight.
    :param weight:
    :param height:
    :return: float
    """
    return round(weight / (height / 100) ** 2, 2)


def add_bmi_inference_data(user_data: dict, bmi_inference_data: dict) -> dict:
    """
    Add the predicted and the calculated BMI values for a given user.
    :param user_data:
    :param bmi_inference_data:
    :return:
    """

    id_map = set(k for k in bmi_inference_data.keys())

    user_id = user_data['user_id']
    if user_id in id_map:
        user_data['bmi_new_model'] = round(bmi_inference_data[user_id], 2)
    else:
        user_data['bmi_new_model'] = 0.0
    user_data['calculated_bmi'] = calculate_bmi(user_data['weight'], user_data['height'])

    return user_data


def prepare_cvd_input(user_data: dict, override_fields: dict = None) -> dict:
    """
    Prepare CVD model input, allows for overriding of fields with inputs from another source.

    :param user_data: (dict)
    :param override_fields: (dict)
    :return: (dict) User data in the required format for the CVD risk model.
    """

    fields = {
        'gender': {'type': str, 'default': 'gender'},
        'age': {'type': int, 'default': 'age'},
        'bmi_model': {'type': float, 'default': 'bmi_model'},
        'waist_circumference': {'type': float, 'default': 'waist_circumference'},
        'antihypertensives': {'type': str, 'default': 'hypertension_diagnosed'},
        'hypertension_on_medication': {'type': str, 'default': 'hypertension_medication'},
        'mi_or_stroke_family_history': {'type': str, 'default': 'stroke_parents_siblings_before_65'},
        'is_smoker': {'type': bool, 'default': 'is_smoker'},
    }

    input_data = {}

    for field, field_params in fields.items():
        field_type = field_params['type']
        field_default = field_params['default']

        data = None
        value = None
        if override_fields and field in override_fields.keys():
            try:
                data = override_fields[field]
                value = field_type(data)
                input_data[field] = value
            except TypeError as te:
                raise TypeError(f"Field {field}, received: {type(data)} expected: {field_type} value: {value}")
        else:
            input_data[field] = field_type(user_data.get(field_default))

    return input_data


def prepare_hypertension_input(user_data: dict, override_fields: dict = None) -> dict:
    """
    Prepare Hypertension model input, allows for overriding of fields with inputs from another source.

    :param user_data: (dict)
    :param override_fields: (dict)
    :return: (dict) User data in the required format for the hypertension risk model.
    """

    fields = {
        'age': {'type': int, 'default_name': 'age', 'default_value': ''},
        'is_smoker': {'type': bool, 'default_name': 'is_smoker', 'default_value': False},
        'gender': {'type': str, 'default_name': 'gender', 'default_value': ''},
        'exercise': {'type': str, 'default_name': 'exercise', 'default_value': ''},
        'exercise_hours': {'type': int, 'default_name': 'exercise_hours', 'default_value': 0},
        'hypertension_family_history': {'type': str, 'default_name': 'hypertension_family_history', 'default_value': ''},
        'bmi_model': {'type': float, 'default_name': 'bmi_model', 'default_value': 0.0},
        'diabetes_currently': {'type': str, 'default_name': 'diabetes', 'default_value': ''},
        'blood_pressure_systolic': {'type': float, 'default_name': 'm_0_obs_1_arm_left_systolic', 'default_value': 0.0},
        'blood_pressure_diastolic': {'type': float, 'default_name': 'm_0_obs_1_arm_left_diastolic', 'default_value': 0.0},
    }

    input_data = {}

    for field, field_params in fields.items():
        field_type = field_params['type']
        field_default = field_params['default_name']
        value = field_params['default_value']
        data = None
        if override_fields and field in override_fields.keys():
            try:
                data = override_fields[field]
                value = field_type(data)
                input_data[field] = value
            except TypeError as te:
                raise TypeError(f"Field {field}, received: {type(data)} expected: {field_type} value: {value}")
        else:
            try:
                value = field_type(user_data.get(field_default))
                input_data[field] = value
            except TypeError as te:
                continue
                # raise TypeError(f"Field {field}, received: {type(data)} expected: {field_type} value: {value} exception: {te}")

    return input_data


def get_bp_reference_points(user_data: dict) -> dict:
    """
    Extracts the max, min and average values for both the ground truth and inference blood pressure values.
    :param user_data:
    :return:
    """

    manual_measurement_keys = {
        "ld_0_obs_1_arm_left_systolic",
        "ld_0_obs_2_arm_left_systolic",
        "ld_0_obs_1_arm_right_systolic",
        "ld_0_obs_2_arm_right_systolic",
        "ld_1_obs_1_arm_left_systolic",
        "ld_1_obs_2_arm_left_systolic",
        "ld_1_obs_1_arm_right_systolic",
        "ld_1_obs_2_arm_right_systolic",
        "ld_2_obs_1_arm_left_systolic",
        "ld_2_obs_2_arm_left_systolic",
        "ld_2_obs_1_arm_right_systolic",
        "ld_2_obs_2_arm_right_systolic",
        "m_0_obs_1_arm_left_systolic",
        "m_0_obs_2_arm_left_systolic",
        "m_1_obs_1_arm_left_systolic",
        "m_1_obs_2_arm_left_systolic",
        "m_2_obs_1_arm_left_systolic",
        "m_2_obs_2_arm_left_systolic",
        "m_3_obs_1_arm_left_systolic",
        "m_3_obs_2_arm_left_systolic",
        "m_4_obs_1_arm_right_systolic",
        "m_4_obs_2_arm_right_systolic",
        "m_5_obs_1_arm_right_systolic",
        "m_5_obs_2_arm_right_systolic",
        "m_6_obs_1_arm_right_systolic",
        "m_6_obs_2_arm_right_systolic",
        "m_7_obs_1_arm_right_systolic",
        "m_7_obs_2_arm_right_systolic",
        "m_8_obs_1_arm_right_systolic",
        "m_8_obs_2_arm_right_systolic",
        "m_9_obs_1_arm_right_systolic",
        "m_9_obs_2_arm_right_systolic",
    }
    model_measurement_keys = {
        "measurement_0_systolic_estimate",
        "measurement_1_systolic_estimate",
        "measurement_2_systolic_estimate",
        "measurement_3_systolic_estimate",
        "measurement_4_systolic_estimate",
        "measurement_5_systolic_estimate",
        "measurement_6_systolic_estimate",
        "measurement_7_systolic_estimate",
        "measurement_8_systolic_estimate",
        "measurement_9_systolic_estimate",
    }

    # Max manual values
    manual_measurement_values = {}
    for k in manual_measurement_keys:
        if k in user_data.keys():
            manual_measurement_values[k] = user_data[k]

    try:
        max_manual = max(manual_measurement_values)
        min_manual = min(manual_measurement_values)
        avg_manual = sum(manual_measurement_values.values())/len(manual_measurement_values.keys())
    except ValueError as e:
        print(f"No max value for {user_data['user_id']} manual keys: {e}")
        return None
    except Exception as e:
        print(f"Exception manual: {e}")
        return None

    # Max model values
    model_measurement_values = {}
    for k in model_measurement_keys:
        if k in user_data.keys():
            model_measurement_values[k] = user_data[k]
    try:
        max_model = max(model_measurement_values)
        min_model = min(model_measurement_values)
        avg_model = sum(model_measurement_values.values())/len(model_measurement_values.keys())
    except ValueError as e:
        print(f"No max value for {user_data['user_id']} model keys: {e}")
        return None
    except Exception as e:
        print(f"Exception model: {e}")
        return None

    return {
        "max_manual": {"key": max_manual, "value": manual_measurement_values[max_manual]},
        "min_manual": {"key": min_manual, "value": manual_measurement_values[min_manual]},
        "avg_manual": avg_manual,
        "max_model": {"key": max_model, "value": model_measurement_values[max_model]},
        "min_model": {"key": min_model, "value": model_measurement_values[min_model]},
        "avg_model": avg_model,
    }






