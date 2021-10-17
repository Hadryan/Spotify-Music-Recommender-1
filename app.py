import streamlit as st
import pandas as pd
from engine import SpotifyRecommender

features_df = pd.read_csv("./data/spotify_audio_features.csv")


def ui() -> None:
    engine = SpotifyRecommender(st.secrets['client_id'], st.secrets['client_secret'])
    st.markdown("# Spotify Music Recommender")
    st.markdown(
        "#### *A Spotify Music Recommender built using spotify API and NearestNeighbors algorithm.\
        The dataset is parsed using the spotify web API with my personal top artists and songs added to the dataset. \
        The NearestNeighbors algorithm finds the cosine distance between all the songs to recommend 6 similar songs.*"
    )
    st.markdown("# Try it out:")
    song_name = st.text_input(label="Enter a song name", value="FEVER")
    song_idx = features_df[features_df.eq(song_name).any(1)].index
    output = engine.predict(song_idx, features_df)

    if isinstance(output, pd.DataFrame):
        fig = engine.visualize_songs(output)
        st.write(fig)

    elif isinstance(output, str):
        st.markdown(f"#### *{output}*")

    st.markdown("")
    st.markdown(
        """# Connect with me
  [<img height="30" src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" />][github]
  [<img height="30" src="https://img.shields.io/badge/linkedin-blue.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />][LinkedIn]
  [<img height="30" src = "https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>][instagram]
  
  [github]: https://github.com/Koushik0901
  [instagram]: https://www.instagram.com/koushik_shiv/
  [linkedin]: https://www.linkedin.com/in/koushik-sivarama-krishnan/""",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    ui()
