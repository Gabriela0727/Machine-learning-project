# diabetes_test.py
# project: Diabetes Prediction using K-Nearest Neighbors (KNN)
#Group 4: Gabriela Rubio, Greysen Kowolewski, Chase Mackey, Shrenik Tripathi

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
import os
from typing import List, Tuple
from random import shuffle, seed
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


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
    print(f"Specificity: {specificity:.2f}%  (Correctly identified {true_negative} of {true_negative + false_positive} non diabetes cases)")
    print("=" * 70)


def print_header(train_size: int, test_size: int):
    print("\n" + "=" * 70)
    print("║" + " " * 68 + "║")
    print("║" + "Diabetes prediction using k nearest neighbors".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("=" * 70)
    print("\nThis program predicts diabetes using the KNN algorithm.")
    print(f"It learns from {train_size} patients and tests on {test_size} patients.")
    print("=" * 70)


def find_csv_file() -> str:
    """Search for diabetes.csv in multiple locations."""
    possible_paths = [
        'diabetes.csv',  # same folder as script
        os.path.join(os.path.dirname(__file__), 'diabetes.csv'),  # script directory
        os.path.expanduser('~/Desktop/diabetes.csv'),  # user's Desktop
        os.path.expanduser('~/Downloads/diabetes.csv'),  # user's Downloads
        '../diabetes.csv',  # parent directory
        '../../Desktop/diabetes.csv'  # original path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        " Could not find diabetes.csv\n"
        "Place the file in one of these locations:\n"
        f"  1. Same folder as this script: {os.path.dirname(os.path.abspath(__file__))}\n"
        "  2. Your Desktop\n"
        "  3. Your Downloads folder"
    )

if __name__ == "__main__":
    try:
        # find the CSV file
        csv_path = find_csv_file()
        
        # Calculate split sizes first (80/20 split of 768 = 614/154)
        total_records = 768
        train_size = int(total_records * 0.8)
        test_size = total_records - train_size
        
        print_header(train_size, test_size)

        print("\n[1] Loading Pima Indians Diabetes Dataset...")
        print(f"    Found CSV at: {csv_path}")
        
        diabetes_parameters: List[List[float]] = []
        diabetes_outcomes: List[int] = []
        
        with open(csv_path, mode='r') as diabetes_file:
            reader = csv.reader(diabetes_file)
            next(reader)  
            data: List = list(reader)
            
            if not data:
                raise ValueError("CSV file is empty!")
            
            # sets random seed for reproducibility
            seed(42)
            shuffle(data)
            
            for row in data:
                if len(row) < 9:
                    continue  # skips incomplete rows
                try:
                    parameters: List[float] = [float(x) for x in row[0:8]]
                    diabetes_parameters.append(parameters)
                    outcome: int = int(row[8])
                    diabetes_outcomes.append(outcome)
                except ValueError:
                    continue  # skips rows with invalid data
        
        print(f"Loaded {len(diabetes_parameters)} patient records")

        print("\n[2] Normalizing features (scaling 0 to 1 range)...")
        normalize_by_feature_scaling(diabetes_parameters)
        print(" All features normalized")

        print("\n[3] Splitting data into training (80%) and testing (20%) sets...")
        split_index = int(len(diabetes_parameters) * 0.8)
        train_data = diabetes_parameters[:split_index]
        train_labels = diabetes_outcomes[:split_index]
        test_data = diabetes_parameters[split_index:]
        test_labels = diabetes_outcomes[split_index:]
        print(f"Training samples: {len(train_data)}")
        print(f"Testing samples:  {len(test_data)}")
        
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
        print("Too big K: Might miss local patterns (underfitting)")
        
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

        k_labels = [k for k, _ in results]
        accuracies = [value for _, value in results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_labels, accuracies, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)
        plt.title("KNN Accuracy for Different K Values", fontsize=14, fontweight='bold')
        plt.xlabel("K Value", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)

        print(f"\n{'*' * 70}")
        print(f"BEST RESULT: K={best_k} achieved {best_accuracy * 100:.2f}% accuracy")
        print(f"{'*' * 70}")
        print_confusion_matrix(train_data, train_labels, test_data, test_labels, best_k)

        print("\n" + "=" * 70)
        print("FINAL SUMMARY".center(70))
        print("=" * 70)
        print(f"\n✓ Successfully classified diabetes with {best_accuracy * 100:.2f}% accuracy")
        print(f"✓ Used K={best_k} nearest neighbors")
        print(f"✓ Dataset: {len(diabetes_parameters)} total patients (Pima Indians)")
        print(f" Features: 8 medical measurements")
        print(f" Classes: Binary (No Diabetes / Has Diabetes)")
        if best_accuracy >= 0.70:
            print(f"\n Achieved {best_accuracy * 100:.2f}% accuracy (requirement: 70%)")
        else:
            print(f"\n Accuracy {best_accuracy * 100:.2f}% is below 70% requirement")
        print("\n" + "=" * 70)
        print("Program completed successfully")
        print("=" * 70 + "\n")

       # visualization
        print("use the buttons to explore different feature combinations")
        
        
        FEATURE_NAMES = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]

        train_data_np = np.array(train_data)
        train_labels_np = np.array(train_labels)

        def detect_outliers(data, feature_idx):
            q1 = np.percentile(data[:, feature_idx], 25)
            q3 = np.percentile(data[:, feature_idx], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (data[:, feature_idx] < lower_bound) | (data[:, feature_idx] > upper_bound)

        # initializes state
        current_x_idx = 7  # age
        current_y_idx = 5  # BMI
        outliers_mask = np.zeros(train_data_np.shape[0], dtype=bool)
        filtered_mode = False

        fig, ax = plt.subplots(figsize=(11, 7))
        plt.subplots_adjust(bottom=0.25, left=0.12, right=0.95)

        def plot_scatter():
            ax.clear()
            if filtered_mode:
                data = train_data_np[~outliers_mask]
                labels = train_labels_np[~outliers_mask]
                title_suffix = " (outliers Removed)"
                num_removed = np.sum(outliers_mask)
            else:
                data = train_data_np
                labels = train_labels_np
                title_suffix = ""
                num_removed = 0
            
            x_vals = data[:, current_x_idx]
            y_vals = data[:, current_y_idx]
            colors = ['red' if label == 1 else 'blue' for label in labels]
            
            ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            ax.set_xlabel(FEATURE_NAMES[current_x_idx], fontsize=12, fontweight='bold')
            ax.set_ylabel(FEATURE_NAMES[current_y_idx], fontsize=12, fontweight='bold')
            
            title = f"{FEATURE_NAMES[current_y_idx]} vs {FEATURE_NAMES[current_x_idx]}{title_suffix}"
            if num_removed > 0:
                title += f"\n({num_removed} outliers removed)"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
            # creates custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', edgecolor='black', label='No Diabetes'),
                Patch(facecolor='red', edgecolor='black', label='Has Diabetes')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # updates button labels
            button_x.label.set_text(f'X: {FEATURE_NAMES[current_x_idx]}')
            button_y.label.set_text(f'Y: {FEATURE_NAMES[current_y_idx]}')
            
            plt.draw()

        def remove_outliers(event):
            global outliers_mask, filtered_mode
            outliers_x = detect_outliers(train_data_np, current_x_idx)
            outliers_y = detect_outliers(train_data_np, current_y_idx)
            outliers_mask = outliers_x | outliers_y
            filtered_mode = True
            plot_scatter()

        def restore_outliers(event):
            global outliers_mask, filtered_mode
            outliers_mask = np.zeros(train_data_np.shape[0], dtype=bool)
            filtered_mode = False
            plot_scatter()

        def change_x_feature(event):
            global current_x_idx
            current_x_idx = (current_x_idx + 1) % len(FEATURE_NAMES)
            if filtered_mode:
                restore_outliers(None)
            plot_scatter()

        def change_y_feature(event):
            global current_y_idx
            current_y_idx = (current_y_idx + 1) % len(FEATURE_NAMES)
            if filtered_mode:
                restore_outliers(None)
            plot_scatter()

        # creates buttons
        button_ax_x = plt.axes([0.12, 0.10, 0.18, 0.05])
        button_x = Button(button_ax_x, f'X: {FEATURE_NAMES[current_x_idx]}')
        button_x.on_clicked(change_x_feature)

        button_ax_y = plt.axes([0.35, 0.10, 0.18, 0.05])
        button_y = Button(button_ax_y, f'Y: {FEATURE_NAMES[current_y_idx]}')
        button_y.on_clicked(change_y_feature)

        button_ax_remove = plt.axes([0.12, 0.03, 0.18, 0.05])
        button_remove = Button(button_ax_remove, 'Remove Outliers')
        button_remove.on_clicked(remove_outliers)

        button_ax_restore = plt.axes([0.35, 0.03, 0.18, 0.05])
        button_restore = Button(button_ax_restore, 'Restore Outliers')
        button_restore.on_clicked(restore_outliers)

        # initial plot
        plot_scatter()
        plt.show(block=True)

        print("\nVisualization completed")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nProgram terminated.")
        exit(1)
    except ValueError as e:
        print(f"\nError reading CSV data: {e}")
        print("Please check that diabetes.csv is properly formatted.")
        exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        print("Program terminated.")
        exit(1)
 """
 Sources:
 - https://www.sciencedirect.com/science/article/abs/pii/S0091743523001998
 - https://www.researchgate.net/publication/371928243_Type_2_Diabetes_Prediction_using_K-Nearest_Neighbor_Algorithm
 - https://www.kaggle.com/code/shrutimechlearn/step-by-step-diabetes-classification?scriptVersionId=200050106
 - https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
 """
