<h1 align="center"> Spotify Recommender Engine 🎧 </h1> 
 
 ***A Spotify Music Recommender built using spotify API and NearestNeighbors algorithm and deployed using streamlit. The dataset is parsed using the spotify web API with my personal top artists and songs added to the dataset. The NearestNeighbors algorithm finds the cosine similarity between all the songs to recommend 6 similar songs.***

## Demo
 ### Try it yourself [here](https://share.streamlit.io/koushik0901/Spotify-Music-Recommender/app.py)

## Implementation Details
- The dataset is parsed using the Spotify Web API. This dataset has various features like artist name, track name, unique track id, popularity, danceability, loudness, speechiness, acousticness and more.
- These features are used to train the unsupervised NearestNeighbors algorithm with cosine similarity as the metric.
- During inference, model returns the distance and the index position on the dataframe for 6 similar songs to the given input song.