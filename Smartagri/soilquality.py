import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

def check_soil_quality(dataset_path='Smartagri/datasets/Crop_recommendation.csv'):
    df = pd.read_csv(dataset_path)
    
    def label_soil(row):
        if row['N'] > 60 and row['P'] > 40 and row['K'] > 40 and 5.5 <= row['ph'] <= 7.5:
            return 'Good'
        elif row['N'] > 30 and row['P'] > 20 and row['K'] > 20:
            return 'Moderate'
        else:
            return 'Poor'
    
    df['soil_quality'] = df.apply(label_soil, axis=1)
    
    X = df[['N', 'P', 'K', 'ph', 'humidity']]
    y = df['soil_quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("\n=== SOIL QUALITY CHECKER ===")
    print(f"Model Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    quality_counts = df['soil_quality'].value_counts()
    plt.figure(figsize=(6,4))
    plt.bar(quality_counts.index, quality_counts.values, color=['green','orange','red'])
    plt.title('Soil Quality Distribution')
    plt.xlabel('Soil Quality')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/soil_quality_chart.png')
    plt.show()
    print("Chart saved to outputs/soil_quality_chart.png")
    
    sample = pd.DataFrame([[90, 42, 43, 6.5, 70]], columns=['N','P','K','ph','humidity'])
    result = model.predict(sample)[0]
    print(f"\nSample Prediction (N=90, P=42, K=43, pH=6.5, Humidity=70%): {result}")
    
    return model, acc
