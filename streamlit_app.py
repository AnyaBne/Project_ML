import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Charger le dataset
df = pd.read_csv('song_dataset.csv')

# Configuration de Surprise
reader = Reader(rating_scale=(df['play_count'].min(), df['play_count'].max()))
data = Dataset.load_from_df(df[['user', 'song', 'play_count']], reader)

# Utilisation de SVD
algo = SVD()

# Entraînement du modèle
trainset = data.build_full_trainset()
algo.fit(trainset)

# Fonction pour obtenir des recommandations
def get_recommendations(user_id, n=10):
    listened_songs = df[df['user'] == user_id]['song'].unique()
    all_songs = df[~df['song'].isin(listened_songs)]['song'].unique()
    predictions = []
    for song in all_songs:
        predictions.append((song, algo.predict(user_id, song).est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_song_ids = [song for song, _ in predictions[:n]]
    recommended_songs = df[df['song'].isin(recommended_song_ids)][['song', 'title']].drop_duplicates().head(n)
    return recommended_songs

# Fonctions pour l'interface Streamlit
def show_home_page():
    st.title('Welcome to the Music Recommendation Engine')
    user_id = st.text_input('Enter your ID:', key="user_id_input")
    if st.button('Login', key='login_button'):
        if user_id in df['user'].unique():
            st.session_state['page'] = 'recommendation'
            st.session_state['user_id'] = user_id
        else:
            st.error('ID doesn\'t exist. Please try again.')

def show_recommendation_page():
    st.title('Music Recommendation Engine')
    st.write(f"Welcome user {st.session_state.get('user_id', '')}.")
    if 'user_id' in st.session_state and st.session_state['user_id']:
        recommended_songs = get_recommendations(st.session_state['user_id'])
        st.subheader('Recommended Songs for you:')
        for index, row in recommended_songs.iterrows():
            st.write(f"{row['song']} - {row['title']}")
    if st.button('Back to Home', key='back_home_button'):
        st.session_state['page'] = 'home'

# Affichage des pages en fonction de l'état de la session
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

if st.session_state['page'] == 'home':
    show_home_page()
elif st.session_state['page'] == 'recommendation':
    show_recommendation_page()
