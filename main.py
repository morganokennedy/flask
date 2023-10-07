from flask import Flask, request, jsonify, render_template

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pickle

import pandas as pd

# Load the dataset
movies_metadata = pd.read_csv('movies_metadata.csv')

# Create a dictionary to quickly map movie titles to their poster paths
movie_to_backdrop = dict(zip(movies_metadata['title'], movies_metadata['poster_path']))
movie_to_overview = dict(zip(movies_metadata['title'], movies_metadata['overview']))


app = Flask(__name__)
class MovieModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):  # Reduced embedding dimension
        super(MovieModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 128)  # Simplified layers
        self.fc2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.7)
        
    def forward(self, user, movie):
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
model = MovieModel(671, 2798)  #
model.load_state_dict(torch.load('model(0,5).pth'))
model.eval() 
with open('user2user_encoded.pkl', 'rb') as f:
    user2user_encoded = pickle.load(f)

with open('movie2movie_encoded.pkl', 'rb') as f:
    movie2movie_encoded = pickle.load(f)

with open('movie_encoded2movie.pkl', 'rb') as f:
    movie_encoded2movie = pickle.load(f)

# You will load your trained model here. Ensure the model is loaded only once when the server starts.
# For demonstration purposes, I'm not loading an actual model. 
# In practice, you'd replace the next lines with actual model loading.
def recommend_movies(model, user_id, top_n=10):
    user_id = int(request.json['user_id'])

    # Convert user id to torch tensor
    if user_id not in user2user_encoded:
        print(f"User ID {user_id} not found in user2user_encoded dictionary.")
        return []
    user_tensor = torch.tensor([user2user_encoded[user_id]], dtype=torch.int64)

    # Calculate user embedding
    user_embedding = model.user_embedding(user_tensor)
    
    # Calculate score for all movies
    all_movie_ids = torch.tensor(range(len(movie2movie_encoded)), dtype=torch.int64)
    all_movie_embeddings = model.movie_embedding(all_movie_ids)
    scores = (user_embedding @ all_movie_embeddings.T).squeeze()
    
    # Rank movies based on score
    _, indices = scores.topk(top_n)
    
    # Convert back to original movie titles
    # top_movies = [movie_encoded2movie[idx.item()] for idx in indices]
    top_movies = [{"title": movie_encoded2movie[idx.item()],
               "poster": movie_to_backdrop.get(movie_encoded2movie[idx.item()], ""),
               "overview": movie_to_overview.get(movie_encoded2movie[idx.item()], "Description not available.")} for idx in indices]

    
    # return top_movies
    
    return top_movies

# Example: Recommend top 5 movies for user with ID 1
# print(recommend_movies(model, 136))



@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     # user_id = request.json['user_id']
#     user_id = int(request.json['user_id'])

#     # Here you would typically generate movie recommendations based on the user ID.
#     # For demonstration purposes, I'll return a static list of movies.
#     recommendations = recommend_movies(model, user_id)
#     # recommendations = ["Mossvie 1", "Movie 2", "Movie 3"]  # Dummy data
    
#     return jsonify(recommendations)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.json['user_id'])
    recommendations = recommend_movies(model, user_id)
    return jsonify(recommendations)



if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable if available, otherwise default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)
