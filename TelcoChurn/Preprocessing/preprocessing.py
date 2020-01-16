# Importing local files
from sklearn.ensemble import RandomForestClassifier
import config as c

# Importing Libraries required for analysis

import luigi
import time
import datetime
import boto3
import boto
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

class download_data(luigi.Task):
    global train, test
    def run(self):
        s3 = boto3.client('s3',
                          aws_access_key_id=c.aws_access_key_id,
                          aws_secret_access_key=c.aws_secret_access_key)
        obj = s3.get_object(Bucket=c.bucket, Key=c.file_name)
        telco = pd.read_csv(obj['Body'])  # 'Body' is a key word
        print(telco)

        telco.to_csv(self.output().path,sep=',',index=False)

    def output(self):
        return luigi.LocalTarget('telco-churn.csv')

class preprocess_data(luigi.Task):
    def requires(self):
        yield download_data()

    def run(self):
        churn = pd.read_csv(download_data().output().path)
        churn['TotalCharges'] = churn['TotalCharges'].astype(str)
        churn['TotalCharges'] = churn['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        churn.dropna(inplace=True)
        churn['TotalCharges'] = pd.to_numeric(churn['TotalCharges'])

        # Imputing binary columns with 1's and 0's
        churn = churn.drop('customerID', axis=1)
        churn.columns = churn.columns.str.lower()
        churn['gender'] = churn['gender'].replace('Male',0)
        churn['gender'] = churn['gender'].replace('Female',1)
        yes_or_no_cols = ['partner','dependents','phoneservice','paperlessbilling','churn']
        for column in yes_or_no_cols:
            churn[column] = churn[column].replace('Yes', 1)
            churn[column] = churn[column].replace('No', 0)

        # Getting dummies for multiple columns
        multiple_value_cols = ['multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
                          'techsupport', 'streamingtv','streamingmovies', 'contract', 'paymentmethod']
        churn = pd.get_dummies(data=churn, columns=multiple_value_cols)

        churn.to_csv(self.output().path,sep=',',index=False)

    def output(self):
        return luigi.LocalTarget('preprocessed.csv')

class train_models(luigi.Task):

    def requires(self):
        yield preprocess_data()

    def run(self):
        preprocessed_churn = pd.read_csv(preprocess_data().output().path)
        features = preprocessed_churn.drop(['churn'], axis=1)
        target = preprocessed_churn['churn']

        ## Logistic Regression
        X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)

        global model_metric_df
        model_metric_df = pd.DataFrame({'Model':[],
                                        'Training Accuracy Score':[],
                                        'Testing Accuracy Score': [],
                                        'True Positives (No Churn)':[],
                                        'False Positives (No Churn)': [],
                                        'False Negatives (Churn)':[],
                                        'True Negatives (Churn)':[],
                                        'Precision (No Churn)':[],
                                        'Precision (Churn)':[],
                                        'Recall (No Churn)':[],
                                        'Recall (Churn)':[],
                                        'FScore (No Churn)':[],
                                        'FScore (Churn)':[],
                                        'Support (No Churn)':[],
                                        'Support (Churn)':[]})

        def model_metrics(estimator_instance, model_name, X_train, X_test, y_train, y_test):
            global model_metric_df
            y_hat_test = estimator_instance.predict(X_test)
            cnf_matrix = confusion_matrix(y_hat_test, y_test)
            results = precision_recall_fscore_support(y_hat_test, y_test)
            true_positive = cnf_matrix[0][0]
            false_positive = cnf_matrix[0][1]
            false_negative = cnf_matrix[1][0]
            true_negative = cnf_matrix[1][1]
            precision_no_churn = results[0][0]
            precision_churn = results[0][1]
            recall_no_churn = results[1][0]
            recall_churn = results[1][1]
            fscore_no_churn = results[2][0]
            fscore_churn = results[2][1]
            support_no_churn = results[3][0]
            support_churn = results[3][1]
            model_train_score = estimator_instance.score(X_train, y_train)
            model_test_score = estimator_instance.score(X_test, y_test)
            estimator_instance_metrics = pd.DataFrame({'Model': [model_name],
                                                       'Training Accuracy Score': [model_train_score],
                                                       'Testing Accuracy Score': [model_test_score],
                                                       'True Positives (No Churn)': [true_positive],
                                                       'False Positives (No Churn)': [false_positive],
                                                       'False Negatives (Churn)': [false_negative],
                                                       'True Negatives (Churn)': [true_negative],
                                                       'Precision (No Churn)': [precision_no_churn],
                                                       'Precision (Churn)': [precision_churn],
                                                       'Recall (No Churn)': [recall_no_churn],
                                                       'Recall (Churn)': [recall_churn],
                                                       'FScore (No Churn)': [fscore_no_churn],
                                                       'FScore (Churn)': [fscore_churn],
                                                       'Support (No Churn)': [support_no_churn],
                                                       'Support (Churn)': [support_churn]})

            model_metric_df = pd.concat([model_metric_df, estimator_instance_metrics],sort=False)

        print('Trying Logistic Regression')
        # Initial Model
        logreg = LogisticRegression(fit_intercept=False)
        logreg.fit(X_train, y_train)
        model_metrics(logreg, 'Logistic Regression', X_train, X_test, y_train, y_test)
        print('Performed Logistic Regression')
        print('\n')

        print('Trying Random Forest')
        ranfor = RandomForestClassifier()
        ranfor.fit(X_train,y_train)
        model_metrics(ranfor,'Random Forest', X_train, X_test, y_train, y_test)
        print('Performed Random Forest')
        print('\n')

        print('Trying XGBoost')
        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)
        model_metrics(xgb, 'XGBoost', X_train, X_test, y_train, y_test)
        print("Performed XGBoost")
        print('\n')

        model_metric_df.to_csv(self.output().path,sep=',',index=False)

    def output(self):
        return luigi.LocalTarget('modelevaluation.csv')

class upload_to_s3(luigi.Task):

    inputLocation = luigi.Parameter()


    def requires(self):
        yield train_models()

    def run(self):
        awsaccess = c.aws_access_key_id
        awssecret = c.aws_secret_access_key
        inputLocation = self.inputLocation


        try:
            conn = boto.connect_s3(awsaccess,awssecret)
            print('Connected to s3')
        except:
            print('Invalid AWS Credentials')
            exit()

        loc = ''
        if inputLocation == 'APNortheast':
            loc = boto.s3.connection.Location.APNortheast
        elif inputLocation == 'APSoutheast':
            loc = boto.s3.connection.Location.APSoutheast
        elif inputLocation == 'APSoutheast2':
            loc = boto.s3.connection.Location.APSoutheast2
        elif inputLocation == 'CNNorth1':
            loc = boto.s3.connection.Location.CNNorth1
        elif inputLocation == 'EUCentral1':
            loc = boto.s3.connection.Location.EUCentral1
        elif inputLocation == 'EU':
            loc = boto.s3.connection.Location.EU
        elif inputLocation == 'SAEast':
            loc = boto.s3.connection.Location.SAEast
        elif inputLocation == 'USWest':
            loc = boto.s3.connection.Location.USWest
        elif inputLocation == 'USWest2':
            loc = boto.s3.connection.Location.USWest2

        try:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts)
            bucket_name = 'modelevaluatedtelcochurn' + str(st).replace(" ", "").replace("-", "").replace(":", "").replace(".","")
            bucket = conn.create_bucket(bucket_name, location=loc)

            print("bucket created")
            s3 = boto3.client('s3', aws_access_key_id=awsaccess, aws_secret_access_key=awssecret)

            print('s3 client created')
            print('The path is', train_models().output().path)
            s3.upload_file(train_models().output().path, bucket_name, 'modelevaluation.csv')

            print("File successfully uploaded to S3",bucket)

        except:
            print("Amazon keys are invalid!!")
            exit()


if __name__ == '__main__':
    luigi.run()


