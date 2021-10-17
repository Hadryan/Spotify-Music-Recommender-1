from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import Union


client_id = "d644cc04eeca4cd1837811019588d947"
client_secret = "5a59f0ac32af44ad959094d98903f422"

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
)

with open("./models/spotify-recommender.pkl", "rb") as f:
    model = pickle.load(f)


def predict(idx: int, features_df: pd.DataFrame) -> Union[str, pd.DataFrame]:
    artists = []
    track_names = []
    track_ids = []
    try:
        data = features_df.iloc[idx, 3:].values.reshape(1, -1)
        distances, indices = model.kneighbors(data, n_neighbors=7)
    except Exception as e:
        return "The song could not be found. Please try something else"

    
    for i in indices:
        artists.extend(features_df.iloc[i, 0].to_list())
        track_names.extend(features_df.iloc[i, 1].to_list())
        track_ids.extend(features_df.iloc[i, 2].to_list())

    # print(artists, track_names, track_ids, distances.flatten())
    out_df = pd.DataFrame.from_dict(
        {
            "artist": artists,
            "track_name": track_names,
            "track_id": track_ids,
            "distance": distances.flatten(),
        }
    ).iloc[1:, :]
    return out_df.sort_values("distance")


def get_info(track_id: str) -> str:
    urn = f"spotify:track:{track_id}"
    track = sp.track(urn)
    thumbnail_url = track["album"]["images"][0]["url"]
    return thumbnail_url


def visualize_songs(out_df: pd.DataFrame) -> None:
    """
    Visualize cover art of the songs in the inputted dataframe

    Args:
        out_df (dataframe): Dataframe returned by the predict function
    """
    image_urls = []

    for idx, row in out_df.iterrows():
        image_urls.append(get_info(row["track_id"]))

    fig = plt.figure(figsize=(35, 35))
    columns = 3

    for i, url in enumerate(image_urls):
        plt.subplot(len(image_urls) // columns + 1, columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(color="w", fontsize=0.1)
        plt.yticks(color="w", fontsize=0.1)
        plt.xlabel(out_df["track_name"].values[i], fontsize=50)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    return fig


if __name__ == "__main__":
    out_df = predict(1)
    visualize_songs(out_df)
