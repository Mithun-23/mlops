import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=5000):
    np.random.seed(42)
    
    # Generate realistic ranges for each feature
    age = np.random.randint(18, 70, size=num_samples)
    gender = np.random.choice(['male', 'female'], size=num_samples)
    
    # Men tend to be taller/heavier, adjust ranges slightly based on gender for realism
    height = np.where(gender == 'male', 
                      np.random.normal(175, 8, num_samples), 
                      np.random.normal(162, 7, num_samples))
    
    weight = np.where(gender == 'male', 
                      np.random.normal(82, 12, num_samples), 
                      np.random.normal(65, 10, num_samples))
    
    duration = np.random.randint(5, 120, size=num_samples)  # 5 to 120 mins
    
    # Heart rate and temp correlate with duration
    heart_rate = np.random.normal(90 + (duration * 0.5), 10, num_samples)
    heart_rate = np.clip(heart_rate, 70, 190)
    
    body_temp = np.random.normal(37 + (duration * 0.02), 0.5, num_samples)
    body_temp = np.clip(body_temp, 36.5, 41.0)
    
    # Calories formulated based roughly on MET (Metabolic Equivalent of Task)
    # Calories burned = duration * (weight) * multiplier + noise
    calories = (duration * (weight / 60)) * (heart_rate / 100) * 4.5
    # Add random noise
    calories += np.random.normal(0, 15, num_samples)
    calories = np.clip(calories, 10, 2000)
    
    df = pd.DataFrame({
        'Age': age.astype(int),
        'Gender': gender,
        'Height': np.round(height, 1),
        'Weight': np.round(weight, 1),
        'Duration': duration.astype(int),
        'Heart_Rate': np.round(heart_rate).astype(int),
        'Body_Temp': np.round(body_temp, 1),
        'Calories': np.round(calories, 1)
    })
    
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/calories.csv', index=False)
    print(f"Generated {num_samples} samples of synthetic data at data/raw/calories.csv")

if __name__ == "__main__":
    generate_synthetic_data()
