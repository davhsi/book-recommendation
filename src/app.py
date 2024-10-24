from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set Flask configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'hsihsi123142$^%')

# Load model and data
artifacts_dir = os.path.join(os.path.dirname(__file__), '../artifacts')
model = pickle.load(open(os.path.join(artifacts_dir, 'model.pkl'), 'rb'))
book_names = pickle.load(open(os.path.join(artifacts_dir, 'book_names.pkl'), 'rb'))
final_rating = pickle.load(open(os.path.join(artifacts_dir, 'final_rating.pkl'), 'rb'))
book_pivot = pickle.load(open(os.path.join(artifacts_dir, 'book_pivot.pkl'), 'rb'))

# Fetch poster URLs based on recommendations
def fetch_posters(suggestions):
    book_titles = []
    poster_urls = []

    # Get book titles from suggestions
    for book_id in suggestions[0]:
        book_titles.append(book_pivot.index[book_id])

    # Fetch poster URLs for each recommended book
    for title in book_titles:
        try:
            book_idx = final_rating[final_rating['title'] == title].index[0]
            poster_urls.append(final_rating.iloc[book_idx]['image_url'])
        except IndexError:
            poster_urls.append('')  # Add a default URL or leave blank if not found

    return poster_urls

# Recommend books
def recommend_book(book_name):
    try:
        # Find the index of the selected book
        book_idx = np.where(book_pivot.index == book_name)[0][0]

        # Get the nearest neighbors (recommended books)
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6
        )

        # Fetch poster URLs for the recommended books
        poster_urls = fetch_posters(suggestions)

        # Get the list of recommended books
        recommended_books = [book_pivot.index[suggestion] for suggestion in suggestions[0]]

        return recommended_books, poster_urls

    except IndexError:
        return [], []  # Return empty lists if the book is not found

# Main route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_books = []
    poster_urls = []
    selected_book = None

    if request.method == 'POST':
        selected_book = request.form.get('book_name')
        recommended_books, poster_urls = recommend_book(selected_book)

    return render_template(
        'index.html',
        book_names=book_names,
        recommended_books=recommended_books,
        poster_urls=poster_urls,  # Correct variable name
        selected_book=selected_book
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
