#!/usr/bin/env python
# coding: utf-8
import os
import psycopg2
import json
import boto3
import base64
import copy
import numpy as np

import matplotlib.pyplot as plt
from botocore.exceptions import ClientError
from lib.bp_model import build_bp_model


class TextFormatter:
    """
    Contains numerous ANSI escape sequences used to apply
    formatting and styling to text.
    """
    # Blue colouring
    BLUE_COL = '\033[94m'
    # Red Colouring
    RED_COL = '\033[91m'
    # Green colouring
    GREEN_COL = '\033[92m'

    # Reset formatting and styling
    RESET = '\033[0m'
    # Underlined text
    UNDERLINE = '\033[4m'
    # Yellow colouring
    YELLOW_COL = '\033[93m'


# AWS secret Manager connectivity
# Get secret from secrets manager
def get_secret(secret_arn: str, region: str = "ap-southeast-1"):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region
    )
    secret = ""

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_arn
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            secret = base64.b64decode(get_secret_value_response['SecretBinary'])

    secret = json.loads(secret)

    return secret


#pg_secret_arn = "arn:aws:secretsmanager:ap-southeast-1:939699530625:secret:rk-db-1/password/usersvc-9woPGQ"
#secret_response = get_secret(pg_secret_arn)


# psycopg2 style
#connection = psycopg2.connect(user=secret_response['username'], password=secret_response['password'], host="rk-db-1.crzs4rm8gzfb.ap-southeast-1.rds.amazonaws.com", port=secret_response['port'], database='usersvc')
#cur = connection.cursor()
#cur.execute("select * from rkuser limit 100")


# SQL alchemy style
#from sqlalchemy import create_engine
#conn_string = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (secret_response['username'], secret_response['password'], "rk-db-1.crzs4rm8gzfb.ap-southeast-1.rds.amazonaws.com", secret_response['port'], 'usersvc')
#db_conn = create_engine(conn_string)
#db_conn.execute("select * from rkuser limit 100")


# Load pre-trained blood Pressure model
def load_bp_model(bp_model_file: str):
    print(TextFormatter.RED_COL + 'Loading and initializing Deep Learning Blood Pressure model' + TextFormatter.RESET)

    # Recreate the exact same model, including its weights and the optimizer
    #new_model = tf.keras.models.load_model(bp_model_file)

    # Show the model architecture
    #new_model.summary()

    period_to_train_on = 20
    model_path = bp_model_file
    pressure_systolic_max = 15
    pressure_diastolic_max = 10
    model = None

    fps = 30
    n_channels = 3

    n_input_features_dim = period_to_train_on * fps
    input_shape = (n_input_features_dim, 1)
    output_shape = (2, 1)

    model = build_bp_model(input_shape=input_shape, num_channels=n_channels, print_model_summary=False)

    try:
        model.load_weights(filepath=model_path)
        print(TextFormatter.GREEN_COL + 'Deep Learning Blood Pressure model loaded and initialized')
        print(TextFormatter.RESET)
    except(Exception,) as error:
        print('Error retrieving Blood Pressure model parameters', error)

    return model, n_input_features_dim


def clean_bp_data(rcm, gcm, bcm):
    # Trim rcm 50 frames from the start and end
    start_frame_count_trim = 50
    end_frame_count_trim = 50

    x_time = np.arange(rcm.shape[0])
    x_time = x_time[start_frame_count_trim:-end_frame_count_trim]
    gcm = gcm[start_frame_count_trim:-end_frame_count_trim]
    bcm = bcm[start_frame_count_trim:-end_frame_count_trim]
    print(TextFormatter.BLUE_COL + 'Blood Pressure data cleaned')
    print(TextFormatter.RESET)
    
    return rcm, gcm, bcm


# Prepare the input data with ppg = extract float array
def bp_data_structure_prep(filtered_ppg_y, n_input_features_dim, ppg_file: str):
    ppg = filtered_ppg_y

    all_raw_data = []
    all_first_diff = []
    all_second_diff = []
    if len(ppg) < (n_input_features_dim + 3):
        raise ValueError(f"File: {ppg_file} does not have long enough recording, skipping.")
    all_raw_data.append(ppg[:n_input_features_dim])

    first_diff = np.diff(copy.deepcopy(ppg))
    all_first_diff.append(first_diff[:n_input_features_dim])

    second_diff = np.diff(copy.deepcopy(first_diff))
    all_second_diff.append(second_diff[:n_input_features_dim])

    all_raw_data = np.expand_dims(all_raw_data, axis=2)
    all_first_diff = np.expand_dims(all_first_diff, axis=2)
    all_second_diff = np.expand_dims(all_second_diff, axis=2)

    input_data = [all_raw_data, all_first_diff, all_second_diff]

    return input_data


# Hide module
def download_bp_video(bpvideo_filename: str, bucket: str, bp_video_file: str):
    # Download the video file and decode from base 64
    #print(bpvideo_filename)

    s3 = boto3.client('s3')

    bpvideo_local_file_path = '/Users/astro/Documents/rt/bloodPressure/temp_downloads'
    os.chdir(bpvideo_local_file_path)
    local_bpvideo_filename = healthrisk_hypertension_data.id[0] + '.mp4'
    s3.download_file(bucket, bp_video_file, local_bpvideo_filename)

    return bpvideo_local_file_path, local_bpvideo_filename

# Change found located:
# Hide module

# Retrieve the video file name from the database table rkbp_video
def find_blood_pressure_video(id):

    querystring_bpvideo = "SELECT filename FROM public.rkbp_video WHERE id = '" + id + "'"

    try:
        bpvideo_data = sqlio.read_sql_query(querystring_bpvideo, connection_healthrisk)
    except(Exception, psycopg2.Error) as error:
        print("Error connecting to healthrisk database, rkbp_video table", error)
        connection = None

    bpvideo_filename = bpvideo_data.filename[0]
    #print(bpvideo_filename)

    bpvideo_bucketname = "rkbpvideo"
    client = boto3.client('s3')
    
    return bpvideo_filename, client, bpvideo_data


def plot_intermediate_bp_values(rcm, gcm, bcm):

    # Plot RGB channels
    #print(TextFormatter.GREEN_COL + 'Plot the extracted raw RGB channel data')
    #print(TextFormatter.RESET)
    plt.plot(np.arange(rcm.shape[0]), rcm,'r')
    plt.plot(np.arange(gcm.shape[0]), gcm,'g')
    plt.plot(np.arange(bcm.shape[0]), bcm,'b')
    plt.xlabel('Frame count')
    plt.ylabel('Channel mean pixel intensity')
    plt.title('Raw data extracted from the video for RGB frames')
    plt.grid(color='k', linestyle=':', linewidth=1)
    plt.show()

    #Plot segment with lowest SD
    #find_frame_range_lowest_sd(rcm)

    # Plot R channel zoomed over 200 data point
    #print(TextFormatter.GREEN_COL + 'Plot the extracted raw RGB channel data')
    #print(TextFormatter.RESET)
    x_rcm = np.arange(rcm.shape[0])
    plt.plot(x_rcm[800:1000], rcm[800:1000],'r')
    plt.xlabel('Frame count')
    plt.ylabel('Channel mean pixel intensity')
    plt.title('Raw Red channel data extracted of a subset 800 to 1000 frames')
    plt.grid(color='k', linestyle=':', linewidth=1)
    plt.show()
                   
                   
    # Plot spectrogram
    #print(TextFormatter.GREEN_COL + 'Plot the extracted frequencies over time')
    #print(TextFormatter.RESET)
    start_frame_count = 0
    end_frame_count = rcm.shape[0]
    signal = rcm[start_frame_count:end_frame_count] - np.mean(rcm[start_frame_count:end_frame_count])
    plt.specgram(signal, NFFT=128, Fs=Fs, noverlap=100)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of the signal extracted from the red channel of the video')
    plt.show()

    return


# Predict Blood Pressure and return vlaues for Systolic and Diastolic
def estimated_blood_pressure(input_data, model):

    # Run the Blood Pressure model
    systolic_multipler = 15
    diastolic_multipler = 10

    all_predictions = model.predict(x=input_data)
    estimated_systolic_bp = all_predictions[0][0]*systolic_multipler
    estimated_diastolic_bp = all_predictions[0][0]*diastolic_multipler
    
    return estimated_systolic_bp, estimated_diastolic_bp


def bmi_two_five_class_prediction(svm_2_bins,svm_5_bins,features):
    # Classification Model - 5 Class category definitions
    # WHO
    bmi_five_class = {
        0: "underweight < 18",
        1: "normal 18 - 23",
        2: "overweight 23 - 27.4",
        3: "pre-obese 27.5 - 32.4",
        4: "obese > 32.5"
    }


    # Classification Model - 2 Class category definitions
    #
    # AIHealth based on NCD risk catergories
    bmi_two_class = {
        0:"underweight or normal",
        1:"overweight to obese"
    }

    five_class_prediction = svm_5_bins.predict(features)
    #print(five_class_prediction[0])
    bmi_five_class_category = bmi_five_class[five_class_prediction[0]]
    #print(bmi_five_class_category)

    two_class_prediction = svm_2_bins.predict(features)
    #print("Two bin classification: ", bmi_two_class[two_class_prediction[0]])
    bmi_two_class_category = bmi_two_class[two_class_prediction[0]]
    #print(bmi_two_class_category)
    
    return bmi_two_class_category, bmi_five_class_category, bmi_five_class, bmi_two_class


def calculated_bmi_two_categories(bmi_prediction):
    
    bmi_calculated_two_category = 0
    
    if bmi_prediction < 23:
        bmi_calculated_two_category = 0
    else:
        bmi_calculated_two_category = 1
        
    return bmi_calculated_two_category


def calculated_bmi_five_categories(bmi_prediction):
    
    bmi_calculated_five_category = 0
    
    if bmi_prediction < 18:
        bmi_calculated_five_category = 0
    elif bmi_prediction >= 18 and bmi_prediction < 23:
        bmi_calculated_five_category = 1
    elif bmi_prediction >= 23 and bmi_prediction < 27.4:
        bmi_calculated_five_category = 2
    elif bmi_prediction >= 27.4 and bmi_prediction < 32.4:
        bmi_calculated_five_category = 3
    else:
        bmi_calculated_five_category = 4
        
    return bmi_calculated_five_category
        

def load_bmi_two_five_class_models(features):
    # Load pre-trained BMI model
    BMI_pretrained_model_local_file_path = '/Users/astro/Documents/rt/bloodPressure/arcanum_model_data'
    os.chdir(BMI_pretrained_model_local_file_path)
    bp_model_file = './model-epoch_6887-val_loss_0.671.hdf5'

    svm_regression_file = 'SVM_regression.pkl'

    svm_reg_model = pickle.load(open('./SVM_regression.pkl', 'rb'))
    svm_5_bins = pickle.load(open('./SVM_5_bins.pkl', 'rb'))
    svm_2_bins = pickle.load(open('./SVM_2_bins.pkl', 'rb'))

    bmi_prediction = svm_reg_model.predict(features)
    #print('Predicted BMI:', bmi_prediction[0])
    
    return svm_2_bins, svm_5_bins, bmi_prediction[0]


def plot_facial_landmarks_boundingbox(local_bmi_image_filename, bmi_image):
    # Call the BMI model
    (size, bounding_boxes, landmarks, conf) = get_facial_features(local_bmi_image_filename)

    # Plot the bounding box and land marks 
    f, axarr = plt.subplots(1,1,figsize=(24,8))
    plt.imshow(bmi_image)
    plt.gca().add_patch(Rectangle((bounding_boxes[0][0],bounding_boxes[0][1]),bounding_boxes[0][2]-bounding_boxes[0][0],bounding_boxes[0][3]-bounding_boxes[0][1],
                        edgecolor='red',
                        facecolor='none',
                        lw=3))

    plt.plot(landmarks[0][0][0],landmarks[0][0][1],'ro')
    plt.plot(landmarks[0][1][0],landmarks[0][1][1],'ro')
    plt.plot(landmarks[0][2][0],landmarks[0][2][1],'ro')
    plt.plot(landmarks[0][3][0],landmarks[0][3][1],'ro')
    plt.plot(landmarks[0][4][0],landmarks[0][4][1],'ro')
    
    return


def risk_category(risk_score):
    risk_category_label = ''
    if risk_score < 10:
        risk_category_label='low'
    elif risk_score > 10 and risk_score <= 20:
        risk_category_label='medium'
    elif risk_score > 20 and risk_score <= 30:
        risk_category_label='high'
    else:
        risk_category_label='very high'
        
    return risk_category_label


def download_selfie_image(id):
    from io import BytesIO

    now = datetime.datetime.now()
    date_dir = now.strftime('%Y-%m-%d')

    # Get bucket object
    bmi_bucketname = "rkfaces"
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bmi_image_bucket = s3.Bucket(bmi_bucketname)

    
    #Find the image filename
    bmi_file_prefix = date_dir + "/" + user_id
    #bmi_file_prefix = "2022-10-04" + "/" + user_id
    print('bmi_file_prefix=',bmi_file_prefix)
    #Find the image filename
    bmi_file = ""
    
    for obj in bmi_image_bucket.objects.filter(Prefix=bmi_file_prefix):
        #print(obj)
        bmi_file = obj.key

    def image_from_s3(bucket, key):
        
        bucket = s3.Bucket(bucket)
        image = bucket.Object(key)
        img_data = image.get().get('Body').read()
        
        return Image.open(io.BytesIO(img_data))

    bmi_image = image_from_s3(bmi_bucketname, bmi_file)

    bmi_image_local_file_path = '/Users/astro/Documents/rt/bloodPressure/temp_downloads'
    os.chdir(bmi_image_local_file_path)
    local_bmi_image_filename = user_id + '.jpg'
    pre_fix = date_dir + "/"
    bmi_filename = pre_fix + user_id
    
    client.download_file(bmi_bucketname,bmi_file,local_bmi_image_filename)
    print(local_bmi_image_filename)


    print("Selfie image downloaded ", bmi_filename)
    #plt.imshow(bmi_image)

    f, axarr = plt.subplots(1,1,figsize=(24,8))
    face_img = Image.open(local_bmi_image_filename)
    axarr.imshow(bmi_image)
    plt.show()

    return local_bmi_image_filename,bmi_image


def find_frame_range_lowest_sd(rcm):
    
    rcm_len = rcm.shape[0]
    time_step = 20 #seconds

    step = int(Fs)*time_step
    ts_stats = {}
    min_sd = []

    for i in range(rcm_len-step):    
        ts_sample = rcm[i:i+1*step]
        ts_stats[i] = {"sample_size":len(ts_sample), "sample_mean":ts_sample.mean(),"sample_std":ts_sample.std()}

        if i > 0:
            if ts_sample.std() < min_sd[1]:
                min_sd = [i,ts_sample.std()]
        else:
            min_sd = [i,ts_stats[0]['sample_std']]

    #print(min_sd[0],min_sd[1])

    #ts_stats
    rcm_optimal_range = rcm[min_sd[0]:min_sd[0]+step]
    rcm_optimal_range_min = min_sd[0]
    rcm_optimal_range_max = min_sd[0]+step
    #print("rcm index min, max", rcm_optimal_range_min, rcm_optimal_range_max)

    x_time = np.arange(rcm.shape[0])
    #plt.plot(x_time[rcm_optimal_range_min:rcm_optimal_range_max], rcm[rcm_optimal_range_min:rcm_optimal_range_max])
    #plt.xlabel('Frame count')
    #plt.ylabel('Red channel mean pixel intensity')
    #plt.title('Trimmed PPG signal')
    #plt.show()
    
    return rcm_optimal_range_min, rcm_optimal_range_max, x_time[rcm_optimal_range_min:rcm_optimal_range_max], rcm[rcm_optimal_range_min:rcm_optimal_range_max]


def find_frame_range_lowest_sd(rcm):
  
    # Prep data and find the best 20 sec with the minimum Standard Deviation

    #rcm = np.c_[red_channel]
    
    #rcm = np.asarray(rcm, dtype=float)
    rcm_optimal_range_min, rcm_optimal_range_max, x_time_opt,rcm_opt = find_frame_range_lowest_sd(rcm)

    print(raw_ppg_data.id[0])


    plt.plot(x_time_opt,rcm_opt)
    plt.xlabel('Frame count')
    plt.ylabel('Red channel mean pixel intensity')
    plt.title('Trimmed PPG signal')
    plt.show()
    
    return


def decode_single_video(input_file_path, video_filename):
    os.chdir(input_file_path)
    with open(video_filename, 'rb') as video:
        video_read = video.read()
    video_64_decode = base64.b64decode(video_read)
    ppg_video = video_filename[:-4] + '_decoded.mp4'

    # create a writable video and write the decoding result
    with open(ppg_video, 'wb') as video_result:
        video_result.write(video_64_decode)
    
    return ppg_video


def load_bp_model(bp_model_weights: str):
    print(TextFormatter.RED_COL + 'Loading and initializing Deep Learning Blood Pressure model' + TextFormatter.RESET)

    period_to_train_on = 20
    pressure_systolic_max = 15
    pressure_diastolic_max = 10

    fps = 30
    n_channels = 3

    n_input_features_dim = period_to_train_on * fps
    input_shape = (n_input_features_dim, 1)
    output_shape = (2, 1)

    try:
        model = build_bp_model(input_shape=input_shape, num_channels=n_channels, print_model_summary=False)
        model.load_weights(filepath=bp_model_weights)
        print(TextFormatter.GREEN_COL + 'Deep Learning Blood Pressure model loaded and initialized')
        print(TextFormatter.RESET)
    except Exception as e:
        print('Error retrieving Blood Pressure model parameters', e)

    return model, n_input_features_dim

