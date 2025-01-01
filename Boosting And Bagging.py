#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tkinter as tk
import pandas as pd
from tkinter import filedialog
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetGUI:
    def __init__(self, master):
        self.master = master
        master.title("Boosting & Bagging ")

        self.filename_label = tk.Label(master, text="No file selected")
        self.filename_label.pack()
        
        self.text_widget = tk.Text(master, width=100, height=10)
        self.text_widget.pack()
        
        self.display_button = tk.Button(master, text="Load file", command=self.load_dataset)
        self.display_button.pack(pady=10)
        
        self.error_message = tk.Message(master, text="")
        self.error_message.pack(pady=10)
        
        self.data_label = tk.Label(master, text="")
        self.data_label.pack(pady=10)

        self.bag_button = tk.Button(master, text="Bagging", command=self.bagging)
        self.bag_button.pack(side="left", ipadx=30, ipady=10, padx=100)

        self.bos_button = tk.Button(master, text="Boosting", command=self.boosting)
        self.bos_button.pack(side="right", ipadx=30, ipady=10, padx=100)

    def load_dataset(self):
        file_path = filedialog.askopenfilename()
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            self.filename_label.config(text="Selected file: " + file_path)
            self.data = pd.read_csv(file_path)
            le = preprocessing.LabelEncoder()
            self.data['class'] = le.fit_transform(self.data['class'])
            self.text_widget.insert(tk.END, self.data.head(10))
            self.X = self.data.drop(columns=['class'])
            self.y = self.data['class']
        else:
            self.error_message.config(text="Invalid file. Please select a .csv file.")

    def evaluate_model(self, model, X_test, y_test, model_name):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=1)
        cm = confusion_matrix(y_test, predictions)
        
        self.data_label.config(text=f"{model_name} Accuracy: {accuracy}\n{model_name} Classification Report:\n{report}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    def boosting(self):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
            adaboost_classifier = AdaBoostClassifier(base_estimator=clf, n_estimators=10)
            adaboost_classifier.fit(X_train, y_train)
            self.evaluate_model(adaboost_classifier, X_test, y_test, "Boosting")
        else:
            self.data_label.config(text="Please select a dataset first.")

    def bagging(self):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=5)
            classifier = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=1)
            bagging_classifier = BaggingClassifier(
                base_estimator=classifier,
                n_estimators=5,
                max_samples=0.8,
                random_state=0,
                max_features=1.0,
                bootstrap=True
            )
            bagging_classifier.fit(X_train, y_train)
            self.evaluate_model(bagging_classifier, X_test, y_test, "Bagging")
        else:
            self.data_label.config(text="Please select a dataset first.")

root = tk.Tk()
width = 720
height = 480
root.geometry(f"{width}x{height}")
gui = DatasetGUI(root)
root.mainloop()

