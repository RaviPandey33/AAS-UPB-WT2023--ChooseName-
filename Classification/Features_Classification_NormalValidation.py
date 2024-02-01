import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

ELA_Data = pd.read_csv('median_features.csv')
ERT_Data = pd.read_csv('rel_ERT.csv')


class ClassifierTrainer_NV: # Classifier Trainer for Leave One out.
    def __init__(self, ELA_Data, ERT_Data, target_columns):
        """
        Init method for the initialization of the variables.

        Args:
            ELA_Data (string): File path of the ELA features extracted using Pflacco
            ERT_Data (string): ERT data set provide during the seminar phase 2. 
            target_columns (list): A list with the algorithm names as its values. This will be used during the classification task.
        """
        
        # print(type(ELA_Data), type(ERT_Data))
        
        self.data = pd.merge(ELA_Data, ERT_Data, on=['dim','fid'], how='left')
        
        self.target_columns = target_columns
        self.data['min_value_column'] = self.data[target_columns].idxmin(axis=1)
        

        self.label_encoder = LabelEncoder()
        self.data['min_value_column'] = self.label_encoder.fit_transform(self.data['min_value_column'])
        
        self.X = self.data.drop('min_value_column', axis=1)
        # self.Y = self.data('min_value_column')
        
        self.loo = LeaveOneOut()
        self.rf_classifier = None
        self.svm_classifier = None
        self.xgb_classifier = None

    def _train_classifier(self, classifier, name, Y_encoded):
        """
        This function is used for the traning of the classifier, and also for validation. This will also calculate different precision metrices
        like accuracy_score, f1_score, recall_score and confusion matrix. We will print the results based for every target algos.

        Args:
            classifier (object): Object of the classifier to be used.
            name (String): Name of the classifier used. 
            Y_encoded (DataFrame column): The target column (or Y column) of the dataFrame.
        
        """
        accuracies = []
        recall_scores = []
        f1_scores = []
        confusion_matrices = []

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.data["min_value_column"], test_size=0.2, random_state=42)
        
        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)

        accuracies.append(accuracy_score(Y_test, Y_pred))
        recall_scores.append(recall_score(Y_test, Y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(Y_test, Y_pred, average='weighted'))
        confusion_matrices.append(confusion_matrix(Y_test, Y_pred, labels=np.unique(Y_encoded)))
        

        mean_accuracy = np.mean(accuracies)
        mean_recall = np.mean(recall_scores)
        mean_f1_score = np.mean(f1_scores)
        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

        print("For not-normalized features :")
        print()
        print(f"Mean Accuracy ({name}): {mean_accuracy}")
        
        original_labels_Y_pred = self.label_encoder.inverse_transform(Y_pred)
        original_labels_Y_test = self.label_encoder.inverse_transform(Y_test)
        
        print(classification_report(original_labels_Y_test, original_labels_Y_pred))
        
        print(f"Mean Recall ({name}): {mean_recall}")
        print(f"Mean F1 Score ({name}): {mean_f1_score}")
        # print(f"Mean Confusion Matrix ({name}):")
        # print(mean_confusion_matrix)

    def train_random_forest(self):
        """ _summary_
        This function is used to initialize the RF classifier object, with 500 estimators. 
        We will then call the _train_classifier method, and pass the classifier object into it for training.
    
        """
        self.rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)
        self._train_classifier(self.rf_classifier, "Random Forest", self.data['min_value_column'])

    def train_svm(self):
        """_summary_
        
        For the training of the SVM classifier.
        """
        self.svm_classifier = SVC(kernel='linear')
        self._train_classifier(self.svm_classifier, "SVM", self.data['min_value_column'])

    def train_xgboost(self):
        """_summary_
        
        For the training of the XGBoost classifier.
        """
        self.xgb_classifier = XGBClassifier()
        
        
        # Encode the target variable Y to ensure sequential class labels
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform( self.data['min_value_column'])
        
        

        self._train_classifier(self.xgb_classifier, "XGBoost",  self.data['min_value_column'])

    def predict(self, classifier, X_new):
        """_summary_
        Given a test set, this method can present us the predicted Y values. 
        
        Args:
            classifier (object): Object of the classifier to be used.
            X_new (object): Input DataFrame object on which Y values are to be predicted. 

        Raises:
            ValueError: If the classifier is not trained yet, then this will raise an error.

        Returns:
            _type_: _description_
        """
        if classifier is not None:
            return classifier.predict(X_new)
        else:
            raise ValueError("Classifier not trained yet.")

# Create an instance of ClassifierTrainer
# target_column = ["BSqi", "BSrr", "CMA-CSA", "fmincon", "fminunc", "HCMA",
#                   "HMLSL", "IPOP400D", "MCS", "MLSL", "OQNLP", "SMAC-BBOB"]


# trainer = ClassifierTrainer_NV(ELA_Data, ERT_Data, target_column)

# # Train the Random Forest classifier
# trainer.train_random_forest()

# # Train the SVM classifier
# trainer.train_svm()

# # Train the XGBoost classifier
# trainer.train_xgboost()

# here we can make predictions using the trained classifiers
# For example, assuming you have a new data point X_new and you want to use the Random Forest classifier:
# predicted_labels_rf = trainer.predict(trainer.rf_classifier, X_new)
