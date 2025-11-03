# diabetes_test.py
# project: Diabetes Prediction using K-Nearest Neighbors (KNN)
"""
what this project does: uses machine learning to predict whether a patient has diabetes based on 8 medical measurements. It uses the Knn, which classifies a patient by looking at the K most similar 
patients in our training data and using a "majority vote" to make a prediction.

dataset: Pima Indians Diabetes Database from kaggle
Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

 The dataset has 768 female patients of Pima Indian heritage. Each patient has 8 medical measurements and a diagnosis (diabetes or not).

input Variables:
 1. pregnancies: Number of times pregnant
 2. glucose: Plasma glucose concentration 
 3. bloodPressure: Diastolic blood pressure 
 4. skinThickness: Triceps skin fold thickness 
 5. insulin: 2 hour serum insulin 
 6. BMI: Body mass index 
 7. DiabetesPedigreeFunction: Diabetes pedigree function (genetic factor)
 8. Age 

 how knn works:
 1. we normalize all features so they're on the same scale (0 to 1)
 2. For a new patient, we calculate the distance to all training patients
 3. we find the K nearest (most similar) patients
 4. we count how many of those K neighbors have diabetes
 5. we predict based on the majority vote
 For example: if K = 5 and 4 out of 5 nearest neighbors have diabetes, we predict this patient also has diabetes.
"""

import csv
from typing import List, Tuple
from random import shuffle
from math import sqrt
import matplotlib.pyplot as plt


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def normalize_by_feature_scaling(dataset: List[List[float]]) -> None:
    for col_num in range(len(dataset[0])):
        column = [row[col_num] for row in dataset]
        maximum = max(column)
        minimum = min(column)
        if maximum == minimum:
            for row_num in range(len(dataset)):
                dataset[row_num][col_num] = 0.0
        else:
            for row_num in range(len(dataset)):
                dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)


def knn_classify(train_data: List[List[float]], train_labels: List[int], 
                 test_point: List[float], k: int = 5) -> int:
    distances = []
    for i, train_point in enumerate(train_data):
        dist = euclidean_distance(test_point, train_point)
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    votes = {}
    for _, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1
    return max(votes.items(), key=lambda x: x[1])[0]


def validate_knn(train_data: List[List[float]], train_labels: List[int],
                 test_data: List[List[float]], test_labels: List[int],
                 k: int = 5) -> Tuple[int, int, float]:
    correct = 0
    for test_point, true_label in zip(test_data, test_labels):
        predicted = knn_classify(train_data, train_labels, test_point, k)
        if predicted == true_label:
            correct += 1
    total = len(test_labels)
    accuracy = correct / total if total > 0 else 0.0
    return correct, total, accuracy


def print_confusion_matrix(train_data: List[List[float]], train_labels: List[int],
                          test_data: List[List[float]], test_labels: List[int], k: int):
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    for test_point, true_label in zip(test_data, test_labels):
        predicted = knn_classify(train_data, train_labels, test_point, k)
        if predicted == 0 and true_label == 0:
            true_negative += 1
        elif predicted == 1 and true_label == 0:
            false_positive += 1
        elif predicted == 0 and true_label == 1:
            false_negative += 1
        elif predicted == 1 and true_label == 1:
            true_positive += 1
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX".center(70))
    print("=" * 70)
    print("\n                      PREDICTED")
    print("                 No Diabetes  |  Has Diabetes")
    print("              " + "-" * 40)
    print(f"ACTUAL No    |      {true_negative:3d}      |      {false_positive:3d}")
    print(f"       Yes   |      {false_negative:3d}      |      {true_positive:3d}")
    print("              " + "-" * 40)
    total = true_negative + false_positive + false_negative + true_positive
    accuracy = (true_positive + true_negative) / total * 100 if total > 0 else 0
    sensitivity = true_positive / (true_positive + false_negative) * 100 if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) * 100 if (true_negative + false_positive) > 0 else 0
    print(f"\nAccuracy:    {accuracy:.2f}%  (Overall correctness)")
    print(f"Sensitivity: {sensitivity:.2f}%  (Caught {true_positive} of {true_positive + false_negative} diabetes cases)")
    print(f"Specificity: {specificity:.2f}%  (Correctly identified {true_negative} of {true_negative + false_positive} non-diabetes cases)")
    print("=" * 70)


def print_header():
    print("\n" + "=" * 70)
    print("║" + " " * 68 + "║")
    print("║" + "Diabetes prediction using k nearest neighbors".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("=" * 70)
    print("\nThis program predicts diabetes using the KNN algorithm.")
    print("It learns from 614 patients and tests on 154 patients.")
    print("=" * 70)


if __name__ == "__main__":
    print_header()

    print("\n[1] Loading Pima Indians Diabetes Dataset...")
    diabetes_parameters: List[List[float]] = []
    diabetes_outcomes: List[int] = []
    with open('diabetes.csv', mode='r') as diabetes_file:
        reader = csv.reader(diabetes_file)
        next(reader)
        data: List = list(reader)
        shuffle(data)
        for row in data:
            parameters: List[float] = [float(x) for x in row[0:8]]
            diabetes_parameters.append(parameters)
            outcome: int = int(row[8])
            diabetes_outcomes.append(outcome)
    print(f"✓ Loaded {len(diabetes_parameters)} patient records")

    print("\n[ 2] Normalizing features (scaling 0 to 1 range)...")
    normalize_by_feature_scaling(diabetes_parameters)
    print("All features normalized")

    print("\n[ 3] Splitting data into training (80%) and testing (20%) sets...")
    split_index = int(len(diabetes_parameters) * 0.8)
    train_data = diabetes_parameters[:split_index]
    train_labels = diabetes_outcomes[:split_index]
    test_data = diabetes_parameters[split_index:]
    test_labels = diabetes_outcomes[split_index:]
    print(f" Training samples: {len(train_data)}")
    print(f" Testing samples:  {len(test_data)}")
    train_diabetic = sum(train_labels)
    train_non_diabetic = len(train_labels) - train_diabetic
    test_diabetic = sum(test_labels)
    test_non_diabetic = len(test_labels) - test_diabetic
    print(f"\nTraining set distribution:")
    print(f"  No Diabetes (0): {train_non_diabetic:3d} ({train_non_diabetic/len(train_labels)*100:.1f}%)")
    print(f"  Has Diabetes (1): {train_diabetic:3d} ({train_diabetic/len(train_labels)*100:.1f}%)")
    print(f"\nTesting set distribution:")
    print(f"  No Diabetes (0): {test_non_diabetic:3d} ({test_non_diabetic/len(test_labels)*100:.1f}%)")
    print(f"  Has Diabetes (1): {test_diabetic:3d} ({test_diabetic/len(test_labels)*100:.1f}%)")

    print("\n[4] Testing different values of K (number of neighbors)...")
    print("\nK is a hyperparameter: we need to find the best value")
    print("Too small K: Sensitive to noise (overfitting)")
    print("Too large K: Might miss local patterns (underfitting)")
    k_values = [3, 5, 7, 9, 11, 15, 21]
    results = []
    best_k = 5
    best_accuracy = 0.0
    print("\nTesting K values...")
    for k in k_values:
        correct, total, accuracy = validate_knn(train_data, train_labels, test_data, test_labels, k)
        results.append((k, accuracy * 100))
        print(f"  K={k:2d}: {correct}/{total} correct = {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # Visualization: line plot for accuracy vs K
    k_labels = [k for k, _ in results]
    accuracies = [value for _, value in results]
    plt.figure(figsize=(8, 5))
    plt.plot(k_labels, accuracies, marker='o', linestyle='-', color='teal')
    plt.title("KNN Accuracy for Different K Values")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\n{'*' * 70}")
    print(f"best result: K={best_k} achieved {best_accuracy * 100:.2f}% accuracy")
    print(f"{'*' * 70}")
    print_confusion_matrix(train_data, train_labels, test_data, test_labels, best_k)

    print("\n" + "=" * 70)
    print("final sumary".center(70))
    print("=" * 70)
    print(f"\n✓ Successfully classified diabetes with {best_accuracy * 100:.2f}% accuracy")
    print(f"✓ Used K={best_k} nearest neighbors")
    print(f"✓ Dataset: 768 total patients (Pima Indians)")
    print(f"✓ Features: 8 medical measurements")
    print(f"✓ Classes: Binary (No Diabetes / Has Diabetes)")
    if best_accuracy >= 0.70:
        print(f"\n Success! Achieved {best_accuracy * 100:.2f}% accuracy (requirement: 70%)")
    else:
        print(f"\n Accuracy {best_accuracy * 100:.2f}% is below 70% requirement")
    print("\n" + "=" * 70)
    print("Program completed successfully")
    print("=" * 70 + "\n")
