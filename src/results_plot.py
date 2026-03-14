import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_results():
    df = pd.read_csv('.\\datasets\\submission.csv')

    y_true = df['Actual_Target']
    y_pred = df['Predicted_Target']

    labels = sorted(list(set(y_true)))

    cm = confusion_matrix(y_true , y_pred , labels=labels)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm , annot=True , fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Student ')
    plt.xlabel("Predicted Label")
    plt.ylabel('True Label')

    os.makedirs('plots',exist_ok=True)
    
    plt.savefig('.\\plots\\confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    plot_results()