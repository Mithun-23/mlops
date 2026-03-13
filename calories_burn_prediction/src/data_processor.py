import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(file_path: str = 'data/raw/calories.csv'):
    """Load the raw dataset."""
    df = pd.read_csv(file_path)
    # Basic cleaning if needed
    return df

def get_preprocessor():
    """
    Returns a compiled Scikit-learn ColumnTransformer pipeline 
    that handles both numerical scaling and categorical encoding.
    """
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    categorical_features = ['Gender']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into predictors (X) and target (y), then into 
    train and test sets.
    """
    X = df.drop(columns=['Calories'])
    y = df['Calories']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
