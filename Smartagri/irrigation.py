import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def analyze_irrigation(dataset_path='datasets/Crop_recommendation.csv'):
    df = pd.read_csv(dataset_path)
    
    def irrigation_need(row):
        if row['rainfall'] < 60 and row['humidity'] < 60:
            return 'High'
        elif row['rainfall'] < 100 and row['humidity'] < 75:
            return 'Moderate'
        else:
            return 'Low'
    
    df['irrigation_need'] = df.apply(irrigation_need, axis=1)
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['irrigation_need']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("\n=== IRRIGATION NEED ANALYSIS ===")
    print(f"Model Accuracy: {acc*100:.2f}%")
    
    irrigation_counts = df['irrigation_need'].value_counts()
    plt.figure(figsize=(6,4))
    plt.pie(irrigation_counts.values, labels=irrigation_counts.index,
            autopct='%1.1f%%', colors=['red','orange','blue'])
    plt.title('Irrigation Need Distribution')
    plt.tight_layout()
    plt.savefig('outputs/irrigation_chart.png')
    plt.show()
    print("Chart saved to outputs/irrigation_chart.png")
    
    sample = pd.DataFrame([[40, 20, 20, 25, 55, 6.0, 50]],
                          columns=['N','P','K','temperature','humidity','ph','rainfall'])
    result = model.predict(sample)[0]
    print(f"\nSample Prediction (Rainfall=50mm, Humidity=55%): Irrigation Need = {result}")
    
    return model, acc