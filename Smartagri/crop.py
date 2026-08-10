import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def analyze_crop_production(dataset_path='datasets/Crop_recommendation.csv'):
    df = pd.read_csv(dataset_path)
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("\n=== CROP PRODUCTION ANALYSIS ===")
    print(f"Model Accuracy: {acc*100:.2f}%")
    
    top_crops = df['label'].value_counts().head(10)
    plt.figure(figsize=(10,5))
    plt.bar(top_crops.index, top_crops.values, color='green')
    plt.title('Top 10 Crops by Frequency in Dataset')
    plt.xlabel('Crop')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/crop_production_chart.png')
    plt.show()
    print("Chart saved to outputs/crop_production_chart.png")
    
    sample = pd.DataFrame([[90, 42, 43, 20.5, 82, 6.5, 200]],
                          columns=['N','P','K','temperature','humidity','ph','rainfall'])
    result = model.predict(sample)[0]
    print(f"\nRecommended Crop for given conditions: {result}")
    
    return model, acc
    