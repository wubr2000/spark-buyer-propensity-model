import numpy as np
import pandas as pd
import cPickle
import json

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector, Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD

def initialize_sparkcontext():
    '''
    INPUT:
        - None
    OUTPUT:
        - sc: a configured and initialized SparkContext 
    
    Initialize SparkContext to be used by other functions to create RDD
    '''
    
    CLUSTER_URL = open('/root/spark-ec2/cluster-url').read().strip()
    
    conf = SparkConf().set(
                        "spark.executor.memory","25g").set(
                        "spark.driver.memory","25g").set(
                        "spark.akka.frameSize","2000").set(
                        "spark.default.parallelism","500").set(
                        "spark.storage.memoryFraction","0.80").setAppName(
                        "dart_app").setMaster(CLUSTER_URL)
    sc = SparkContext(conf = conf) 
    
    return sc


def pickle_file(p, pkl_filename):
    f = open(pkl_filename,"wb")
    cPickle.dump(p, f, protocol=2)
    f.close()
    

def aud_segment_id_master_list(filename_and_path, sc):
    '''
    INPUT: 
        - filename_and_path: full S3 path & filename ('audience_segment_master.csv') of the latest master list containing 
            all audience segment IDs, corresponding column numbers, and names from DART
        - sc: SparkContext already initialized
    OUTPUT: 
        - all_aud_id: a Python list of tuples: (Move ID, Dart ID, Audience Segment Names)
            Move IDs are integers representing our own index number for each audience segment. 
            Move IDs are necessary because we need to use column numbers in creating a matrix for LabeledPoint RDDs
            Dart IDs are integers which are DART assigned audience segment IDs
            Audience Segment Names are strings
    '''
    
    masterlist_aud_segs = sc.textFile(filename_and_path, 64)
    header = masterlist_aud_segs.first()
    all_aud_id = masterlist_aud_segs\
                    .filter(lambda line: line != header)\
                    .map(lambda x: x.split(','))\
                    .map(lambda audience: (int(audience[0]),int(audience[1]),audience[2]) if 'NULL' not in audience[1] else 0 ).distinct()\
                    .filter(lambda x: x != 0)\
                    .sortBy(lambda x: x).collect()
    
    return all_aud_id


def process_raw_log_into_rdd(filename_and_path, major_metro_names, sc):
    '''
    INPUT:
        - filename_and_path: full S3 path & filename of the DART raw logs to be processed
        - major_metro_names: a list of the major metros (strings) for which there are individual metro models
        - sc: SparkContext already initialized
    OUTPUT:
        - dart_has_audience_id: an RDD containing every row of the input DART raw log split into 37 items
        - dart_headers: a Python list of strings containing the header names for each of the 37 items
        - metro_id_name_dict: a Python dictionary containing (metro name, metro IDs) pairs. 
            Metro names are strings and metro IDs are integers.
    
    Read in DART raw logs, split on delimiter, get rid of header row.
    '''
    
    s3_input_test_file = sc.textFile(filename_and_path, 192)
    delimiter = list(s3_input_test_file.take(1)[0])[4:5]
    raw_log_delimited = s3_input_test_file.map(lambda line: line.split(delimiter[0]))
    dart_headers = raw_log_delimited.first()
    dart_has_audience_id = raw_log_delimited.filter(lambda line: line != dart_headers)
    
    #Create Metro IDs and Metro Names dictionary
    metro_id_name_dict = {}
    for metro in major_metro_names:
        metro_id = dart_has_audience_id.filter(lambda dart: metro in dart[dart_headers.index('Metro')]).map(lambda dart: dart[dart_headers.index('MetroId')]).take(1)
        metro_id_name_dict[metro] = int(metro_id[0])
    
    return dart_has_audience_id, dart_headers, metro_id_name_dict


def build_sparse_vector_and_leads_labels_rdd(all_aud_id, dart_has_audience_id, dart_headers):
    '''
    INPUT:
        - all_aud_id: a Python list of integers containing all audience segment IDs in DART
        - dart_has_audience_id: an RDD containing every row of the input DART raw log split into 37 items
        - dart_headers: a Python list of strings containing the header names for each of the 37 items
    OUTPUT:
        - jy_sparse_vector_rdd: an RDD of tuples: ( jy_tag, Sparse Vector RDD ) where 
            jy_tag is a tuple of strings consisting of (jy tag, metro ID) and
            Sparse Vector RDD can be used for constructing LabeledPoints later
        - leads_label_rdd: an RDD of the same length as jy_sparse_vector_rdd indicating 
            which user has submitted a buyer's lead before as recorded in DART
    
    Notes:
    - Each (jy tag, metro ID) combination is unique and every audience segment ID belonging to that JY is also unique.
    - The audience segment IDs belonging to each JY comprises all audience segments 
        that a particular user has been given within the timeframe of the raw log files which were processed.
    - Sparse Vector RDD contains not the actual audience segment IDs (DartID) but a distinct column number (MoveID) that is
        mapped for that particular audience segment ID. This is because we need to constrain the number of columns
        of the RDD matrix to a manageable size (DartID can be very large numerically) in order for regression model
        to run without any memory issues later.
    '''
    
    #Get relevant column numbers from Dart raw log for constructing RDD
    col_customtargeting = dart_headers.index('CustomTargeting')
    col_metroid = dart_headers.index('MetroId')
    col_audsegs = dart_headers.index('AudienceSegmentIds')
    
    #Audience segment IDs containing leads and also those to be excluded from 
    showcase_leads_id = 17532010
    cobroke_leads_id = 17531770
    find_broker_leads_id = 17531650
    home_values_leads_id = 17553490
    
    buyer_leads = [showcase_leads_id, cobroke_leads_id, find_broker_leads_id]
    land_mobile_foreclosures = [17535490, 18230530, 17535370]
    excluded = buyer_leads + [home_values_leads_id] + land_mobile_foreclosures
    
    #Convert DartID (audience segment IDs by DFP) and Excluded DartIDs from a list of integers into a list of strings
    #This is for creating sparse vector RDD which outputs a list of DartIDs as strings
    all_aud_id_string = [str(dart_id) for move_id, dart_id, segment_name in all_aud_id]
    excluded_string = [str(excl) for excl in excluded]
    
    #Create a dictionary for mapping each DART audience segment ID (DartID) to a unique column number (MoveID)
    aud_id_dict = dict(zip( [dart_id for move_id, dart_id, segment_names in all_aud_id], [move_id for move_id, dart_id, segment_names in all_aud_id]))
    
    #Create an RDD that contains all audience segment IDs for each user within the period of time covered by the raw log input files
    jy_aud_segments = dart_has_audience_id\
            .map(lambda dart: ((dart[col_customtargeting],dart[col_metroid]), dart[col_audsegs]))\
            .map(lambda ((jy_tag, metro_id), aud_segs): (\
        #To get JY tag for each row
                ([tag.split('=')[1] for tag in jy_tag.split(";") if "jy=" in tag], metro_id),
        #To get audience segment IDs for each row
                [ a for a in aud_segs.split(",") ] \
                    if "," in aud_segs else \
                [ a for a in aud_segs.split("|") ] ))\
            .map(lambda ((jy_tag, metro_id), aud_segs): ((jy_tag[0] if jy_tag else '', metro_id), set(aud_segs)) )\
            .reduceByKey(lambda x, y: x | y).cache()

    #Create leads label to identify users who have previously submitted buyer's leads as recorded in DART
    leads_label_rdd = jy_aud_segments\
        .map(lambda (jy_tag, aud_segs): 1 if aud_segs.intersection([str(leads) for leads in buyer_leads]) else 0 ).cache()
                
    #Build a sparse vector of corresponding column for audience segment IDs of each user
    #Include only audience segments that are in the master list and exclude those that are in the excluded list
    #Then map audience segment IDs to column numbers for the sparse vector
    jy_sparse_vector_rdd = jy_aud_segments\
            .map(lambda (jy_tag, aud_segs): (jy_tag, [int(l) for l in list( (set(all_aud_id_string) & aud_segs) - set(excluded_string) )] ))\
            .map(lambda (jy_tag, aud_segs): (jy_tag, [aud_id_dict[dart_id]-1 for dart_id in aud_segs]) if aud_segs else (jy_tag, aud_segs) )\
            .map(lambda (jy_tag, aud_ids): (jy_tag, 
                                            Vectors.sparse(max(aud_id_dict.values()), dict([ (int(u), 1.0) for u in aud_ids ])) )).cache()
    
    return jy_sparse_vector_rdd, leads_label_rdd


def build_nonsampled_labeledpoints(leads_label_rdd, sparse_vector_rdd):
    '''
    INPUT:
        - leads_label_rdd: an RDD of the same length as jy_sparse_vector_rdd indicating 
            which user has submitted a buyer's lead before as recorded in DART
        - jy_sparse_vector_rdd: an RDD of tuples: ( jy_tag, Sparse Vector RDD ) where 
            jy_tag is a tuple of strings consisting of (jy tag, metro ID) and
            Sparse Vector RDD can be used for constructing LabeledPoints later
    OUTPUT:
        - nonsampled_labeledpoints_with_metroid: LabeledPoints that have NOT been upsampled or downsampled
        
    Note: This set of labeledpoints is to be used for predictions (without any up- or down-sampling)
    '''
    
    nonsampled_labeledpoints_with_metroid = leads_label_rdd\
                                                .zip(sparse_vector_rdd)\
                                                .map(lambda (y, ( jy_tag, x )) : (jy_tag, LabeledPoint( y, x )) ).cache()

    return nonsampled_labeledpoints_with_metroid


def build_upsampled_labeledpoints(leads_label_rdd, sparse_vector_rdd):
    '''
    INPUT:
        - leads_label_rdd: an RDD of the same length as jy_sparse_vector_rdd indicating 
            which user has submitted a buyer's lead before as recorded in DART
        - jy_sparse_vector_rdd: an RDD of tuples: ( jy_tag, Sparse Vector RDD ) where 
            jy_tag is a tuple of strings consisting of (jy tag, metro ID) and
            Sparse Vector RDD can be used for constructing LabeledPoints later
    OUTPUT:
        - upsampled_labeledpoints_with_metroid: LabeledPoints that have been UPSAMPLED to 50%
        
    Note: This set of labeledpoints is to be used for training (upsampling because of severe class imbalance)
    '''

    #Percentage by which minority class will be upsampled to
    minority_class_increases_to = 0.50
    majority_class_decreases_to = 1 - minority_class_increases_to
    
    #Upsampling (using Stratified Sampling) using sampleByKey() function
    concat_rdd = leads_label_rdd.zip(sparse_vector_rdd).cache()
    
    num_rows_with_leads = leads_label_rdd.sum()
    num_rows = leads_label_rdd.count()
    upsampled_leads_rdd = concat_rdd.sampleByKey(True, \
        fractions={1: num_rows/num_rows_with_leads*minority_class_increases_to, \
                   0: majority_class_decreases_to}).cache()

    #Build LabeledPoints USING UPSAMPLED sparse vectors for MLLib models
    upsampled_labeledpoints_with_metroid = upsampled_leads_rdd.map(lambda (y, ( jy_tag, x )) : (jy_tag, LabeledPoint( y, x )) ).cache()
    
    return upsampled_labeledpoints_with_metroid


def train_and_save_national_logistic_models(upsampled_labeledpoints_with_metroid, all_aud_id):
    '''
    INPUT:
        - labeledpoints_with_metroid_rdd: ( (jy tag, metro ID), LabeledPoints RDD ) where
            jy tag is a string, metroID is a string, and a LabeledPoints RDD 
    OUTPUT:
        - logistic_model: a logistic regression model which is a MLLib object
        - accuracy: a float that provides the accuracy of the model (on 20% holdout sample)
        - top variables: top 15 variables with the highest coefficients
    '''
    
    # Split data into training and test data sets
    train_set, test_set = upsampled_labeledpoints_with_metroid.randomSplit([0.8, 0.2], seed=1)
    
    #Logistic Model (Regularized)
    logistic_model = LogisticRegressionWithSGD.train(train_set.map(lambda (jy, labeledpoints): labeledpoints),
                                                     iterations = 100,
                                                     regParam = 0.005,
                                                     regType = "l1",
                                                     intercept=False)
    
    #Get the top variables and their coefficients
    coefficients = logistic_model.weights.array
    top_variables = sorted([(aud[2],coefficients[aud[0]-1]) for aud in all_aud_id if coefficients[aud[0]-1] != 0.0], key=lambda x: x[1],reverse=True)
    
    #Calculate accuracy of the model
    logistic_prediction_and_labels = test_set\
                        .map(lambda (jy, labeledpoints): labeledpoints)\
                        .map(lambda test_data: (logistic_model.predict(test_data.features), test_data.label))          
    correct = logistic_prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    accuracy = correct.count() / float(test_set.count())
    
    #Save the model as a pickle file and save the ocoefficients as a json file
    model_json = dict(zip([str(aud[1]) for aud in all_aud_id], logistic_model.weights))
    model_tuple = (0, model_json)
    
    with open('leads_model_v2_0.json', 'w') as fp:
            json.dump(model_tuple, fp)

    pickle_file(logistic_model, 'leads_model_v2_0.pkl')
    
    return logistic_model, accuracy, top_variables


def train_and_save_metro_logistic_models(upsampled_labeledpoints_with_metroid, all_aud_id, metro_id_name_dict, model_prefix):
    '''
    INPUT:
        - labeledpoints_with_metroid_rdd: ( (jy tag, metro ID), LabeledPoints RDD ) where
            jy tag is a string, metroID is a string, and a LabeledPoints RDD 
    OUTPUT:
        saved pickled models for each metro and json files containing coefficients and DART Ids.
    '''
          
    for metro in metro_id_name_dict.keys():
        
        print "Now working on", metro, "metro:", metro
        
        #Filter Labeledpoint RDDs for specific metros
        upsampled_labeledpoints_with_metroid = upsampled_labeledpoints_with_metroid\
                    .filter(lambda ((jy, metroid), labeledpoint):  metroid == str(metro_id_name_dict[metro]))
    
        # Split data into training and test data sets
        train_set, test_set = upsampled_labeledpoints_with_metroid.randomSplit([0.8, 0.2], seed=1)

        #Logistic Model (Regularized)
        logistic_model = LogisticRegressionWithSGD.train(train_set.map(lambda (jy, labeledpoints): labeledpoints),
                                                         iterations = 100,
                                                         regParam = 0.005,
                                                         regType = "l1",
                                                         intercept=False)

        #Get the top variables and their coefficients
        coefficients = logistic_model.weights.array
        top_variables = sorted([(aud[2],coefficients[aud[0]-1]) for aud in all_aud_id if coefficients[aud[0]-1] != 0.0], key=lambda x: x[1],reverse=True)

        #Calculate accuracy of the model
        logistic_prediction_and_labels = test_set\
                            .map(lambda (jy, labeledpoints): labeledpoints)\
                            .map(lambda test_data: (logistic_model.predict(test_data.features), test_data.label))          
        correct = logistic_prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
        accuracy = correct.count() / float(test_set.count())
        
        print "Logistic accuracy for this metro: %2.2f" % accuracy
        print "Top 15 variables for this metro model:", top_variables[:15]

        #Save the model as a pickle file and save the ocoefficients as a json file
        model_json = dict(zip([str(aud[1]) for aud in all_aud_id], logistic_model.weights))
        model_tuple = (metro_id_name_dict[metro], model_json)

        with open(model_prefix + str(metro_id_name_dict[metro]) + '.json', 'w') as fp:
                json.dump(model_tuple, fp)

        pickle_file(logistic_model, model_prefix + str(metro_id_name_dict[metro]) + '.pkl')

    return


def train_and_save_logistic_models(upsampled_labeledpoints_with_metroid, all_aud_id, metro_id_name_dict, model_prefix):
    '''
    INPUT:
        - labeledpoints_with_metroid_rdd: ( (jy tag, metro ID), LabeledPoints RDD ) where
            jy tag is a string, metroID is a string, and a LabeledPoints RDD 
    OUTPUT:
        saved pickled models and json files containing coefficients and DART Ids, for national data and each metro.
    '''
          
    print "Now working on national model"
    logistic_model, model_tuple = run_logisitic_model(upsampled_labeledpoints_with_metroid, all_aud_id)
    with open('leads_model_v2_0.json', 'w') as fp:
        json.dump(model_tuple, fp)
    pickle_file(logistic_model, 'leads_model_v2_0.pkl')
    
    for metro in metro_id_name_dict.keys():
        print "Now working on", metro, "metro:", metro
        #Filter Labeledpoint RDDs for specific metros
        metro_points = upsampled_labeledpoints_with_metroid\
                    .filter(lambda ((jy, metroid), labeledpoint):  metroid == str(metro_id_name_dict[metro]))
        logistic_model, model_tuple = run_logistic_model(metro_points, all_aud_id)
        with open(model_prefix + str(metro_id_name_dict[metro]) + '.json', 'w') as fp:
                json.dump(model_tuple, fp)
        pickle_file(logistic_model, model_prefix + str(metro_id_name_dict[metro]) + '.pkl')

    return
    

def run_logistic_model(input_points, all_aud_id)
    # Split data into training and test data sets
    train_set, test_set = input_points.randomSplit([0.8, 0.2], seed=1)

    #Logistic Model (Regularized)
    logistic_model = LogisticRegressionWithSGD.train(train_set.map(lambda (jy, labeledpoints): labeledpoints),
                                                     iterations = 100,
                                                     regParam = 0.005,
                                                     regType = "l1",
                                                     intercept=False)

    #Get the top variables and their coefficients
    coefficients = logistic_model.weights.array
    top_variables = sorted([(aud[2],coefficients[aud[0]-1]) for aud in all_aud_id if coefficients[aud[0]-1] != 0.0], key=lambda x: x[1],reverse=True)

    #Calculate accuracy of the model
    logistic_prediction_and_labels = test_set\
                        .map(lambda (jy, labeledpoints): labeledpoints)\
                        .map(lambda test_data: (logistic_model.predict(test_data.features), test_data.label))          
    correct = logistic_prediction_and_labels.filter(lambda (predicted, actual): predicted == actual)
    accuracy = correct.count() / float(test_set.count())

    print "Logistic accuracy for model: %2.2f" % accuracy
    print "Top 15 variables for model:", top_variables[:15]


    #Save the model as a pickle file and save the ocoefficients as a json file
    model_json = dict(zip([str(aud[1]) for aud in all_aud_id], logistic_model.weights))
    model_tuple = (metro_id_name_dict[metro], model_json)

    return logistic_model, model_tuple


def predict_using_pickled_models(raw_dart_log_for_predict, master_list_filename, major_metro_names, model_prefix, sparkcontext):
    '''
    INPUT: 
        - model_prefix: filename prefix for the models that have been saved as pickled files
        
    First, unpickle and save all models.
    Then
    '''
    
    #Create sparse vectors and leads labels from the DART raw log in preparation for prediction 
    all_aud_id = aud_segment_id_master_list(master_list_filename, sparkcontext)
    dart_has_audience_id, dart_headers, metro_id_name_dict = process_raw_log_into_rdd(raw_dart_log_for_predict, major_metro_names, sparkcontext)
    jy_sparse_vector_rdd, leads_labels_rdd = build_sparse_vector_and_leads_labels_rdd(all_aud_id, dart_has_audience_id, dart_headers)
    
    #Unpickle and save all metro models into a dictionary {key=metroid, value=unpickled model}
    metro_logistic_model = {}
    for metro in metro_id_name_dict.keys():
        metro_logistic_model[metro_id_name_dict[metro]] = cPickle.load(open(model_prefix + str(metro_id_name_dict[metro]) + '.pkl','rb'))
        # Clear the default threshold. So that the output becomes a probability
        metro_logistic_model[metro_id_name_dict[metro]].clearThreshold()

    #Unpickle and save the national model (model 0) into the same dictionary
    metro_logistic_model[0] = cPickle.load(open(model_prefix + '0.pkl','rb'))
    metro_logistic_model[0].clearThreshold()
    
    #Build LabeledPoints RDD for Predict
    nonsampled_labeledpoints_with_metroid = build_nonsampled_labeledpoints(leads_labels_rdd, jy_sparse_vector_rdd)
    
    # Get predicted probability and actual labels for each user ACCORDING TO USER'S METRO ID
    jy_and_scores = nonsampled_labeledpoints_with_metroid\
            .map(lambda ((jy, metro_id), labeledpoint):\
                 (jy, metro_logistic_model[metro_id].predict(labeledpoint.features))\
                     if metro_id in metro_id_name_dict.values() else\
                 (jy, metro_logistic_model[0].predict(labeledpoint.features))\
                ).collect()
    
    return metro_logistic_model, jy_and_scores