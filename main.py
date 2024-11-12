import spotipy
from spotipy.oauth2 import SpotifyOAuth
from transformers import pipeline

# Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='4556136bac284778808fdf213949c3da',
    client_secret='9dd6a72dbc2b45218f9ea66a3c77bed9',
    redirect_uri='http://localhost:8888/callback',
    scope='playlist-modify-private'))

# Emotion Detection Model
# Load the emotion detection model with the updated parameter
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Emotion to genre mapping
emotion_to_genre = {
    'joy': 'pop',
    'sadness': 'acoustic',
    'anger': 'rock',
    'fear': 'ambient',
    'love': 'romance',
    'surprise': 'dance'
}

def detect_primary_emotion(text):
    results = emotion_detector(text)
    primary_emotion = max(results[0], key=lambda x: x['score'])
    return primary_emotion['label']

def get_tracks_by_genre(genre, limit=10):
    results = sp.search(q=f'genre:{genre}', type='track', limit=limit)
    track_ids = [track['id'] for track in results['tracks']['items']]
    return track_ids

def create_playlist_for_emotion(emotion, track_ids):
    user_id = sp.current_user()['id']
    playlist_name = f"{emotion.capitalize()} Mood Playlist"
    playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
    sp.user_playlist_add_tracks(user_id, playlist['id'], track_ids)
    print(f"Playlist '{playlist_name}' created with {len(track_ids)} tracks!")

def main():
    user_input = input("Describe how you're feeling: ")
    emotion = detect_primary_emotion(user_input)
    print(f"Detected emotion: {emotion}")
    
    genre = emotion_to_genre.get(emotion, 'pop')
    print(f"Selected genre for playlist: {genre}")
    
    track_ids = get_tracks_by_genre(genre)
    create_playlist_for_emotion(emotion, track_ids)

if __name__ == "__main__":
    main()
