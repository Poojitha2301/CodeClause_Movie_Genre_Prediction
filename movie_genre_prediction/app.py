from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('movie_genre_model.pkl')  # Ensure your model file is correctly named and located

def predict_genre(plot_summary):
    # Preprocess the plot summary and make the prediction
    # Modify this part based on your model's requirements
    processed_data = [plot_summary]  # Placeholder for preprocessing
    prediction = model.predict(processed_data)  # Ensure this matches your model's expected input
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    genre = None
    if request.method == 'POST':
        plot_summary = request.form.get('plot_summary')
        genre = predict_genre(plot_summary)
    return render_template('index.html', genre=genre)

if __name__ == '__main__':
    app.run(debug=True)
