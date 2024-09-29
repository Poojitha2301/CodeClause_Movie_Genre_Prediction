import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset here
def load_data():
    data = {
        'plot_summary': [
            'A young wizard discovers his magical heritage.',
            'A thrilling chase through space.',
            'A romantic drama about love and loss.',
            # Add more samples...
        ],
        'genre': [
            'Fantasy',
            'Sci-Fi',
            'Romance',
            # Add corresponding genres...
        ]
    }
    df = pd.DataFrame(data)
    return df

def train_model():
    df = load_data()
    X = df['plot_summary']
    y = df['genre']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train_vectorized, y_train)

    # Save the model and vectorizer
    joblib.dump(model, 'movie_genre_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    train_model()
