import os
import re
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# 1. SETUP & COMPATIBILITY
os.environ['TF_USE_LEGACY_KERAS'] = '1'
app = Flask(__name__)
app.secret_key = 'ipu_project_secret_key'
TMDB_API_KEY = 'da26246f79a9f2c547c2ada13e5b39a5' 

def get_tmdb_results(query):
    # 1. Pehle check karo ki ye koi Actor (Person) toh nahi hai
    person_url = f"https://api.themoviedb.org/3/search/person?api_key={TMDB_API_KEY}&query={query}"
    try:
        p_res = requests.get(person_url, timeout=3).json()
        external_results = []
        
        # Agar koi actor mil gaya (Jaise Ranbir, Salman, Akshay)
        if p_res.get('results'):
            person_id = p_res['results'][0]['id']
            # Uss actor ki saari movies nikaalo
            movie_url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?api_key={TMDB_API_KEY}"
            m_res = requests.get(movie_url, timeout=3).json()
            
            for item in m_res.get('cast', [])[:20]: # Top 20 movies
                path = item.get('poster_path')
                poster = f"https://image.tmdb.org/t/p/w500{path}" if path else "https://via.placeholder.com/300x450"
                external_results.append({
                    'movieId': item['id'], 
                    'title': item.get('title'),
                    'poster': poster,
                    'year': item.get('release_date', 'N/A')[:4],
                    'rating': item.get('vote_average', 'N/A'),
                    'mood': 'Actor Hit'
                })
            return external_results

        # 2. Agar actor nahi mila, toh normal Movie search karo
        movie_search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
        m_res = requests.get(movie_search_url, timeout=3).json()
        for item in m_res.get('results', []):
            path = item.get('poster_path')
            poster = f"https://image.tmdb.org/t/p/w500{path}" if path else "https://via.placeholder.com/300x450"
            external_results.append({
                'movieId': item['id'], 'title': item.get('title'), 'poster': poster,
                'year': item.get('release_date', 'N/A')[:4], 'rating': item.get('vote_average', 'N/A'),
                'mood': 'New Discovery'
            })
        return external_results
    except:
        return []
# 2. LOAD DATA
CSV_PATH = 'movies_with_popularity.csv'
df = pd.read_csv(CSV_PATH).fillna('')

def extract_year(title):
    match = re.search(r'\((\d{4})\)', str(title))
    return match.group(1) if match else "N/A"

df['year'] = df['title'].apply(extract_year)

# 3. AI MODEL ARCHITECTURE
unique_movie_titles = df['title'].unique()
unique_cast = np.unique("|".join(df['cast'].astype(str)).split("|"))
unique_keywords = np.unique("|".join(df['keywords'].astype(str)).split("|"))
unique_moods = df['mood'].unique()

class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.title_lookup = tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None)
        self.title_emb = tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        self.cast_lookup = tf.keras.layers.StringLookup(vocabulary=unique_cast, mask_token=None)
        self.cast_emb = tf.keras.layers.Embedding(len(unique_cast) + 1, 16)
        self.key_lookup = tf.keras.layers.StringLookup(vocabulary=unique_keywords, mask_token=None)
        self.key_emb = tf.keras.layers.Embedding(len(unique_keywords) + 1, 16)
        self.mood_lookup = tf.keras.layers.StringLookup(vocabulary=unique_moods, mask_token=None)
        self.mood_emb = tf.keras.layers.Embedding(len(unique_moods) + 1, 32)

    def call(self, inputs):
        t_v = self.title_emb(self.title_lookup(inputs["title"]))
        c_ids = self.cast_lookup(tf.strings.split(inputs["cast"], "|"))
        c_v = tf.reduce_mean(self.cast_emb(c_ids), axis=1)
        k_ids = self.key_lookup(tf.strings.split(inputs["keywords"], "|"))
        k_v = tf.reduce_mean(self.key_emb(k_ids), axis=1)
        m_v = self.mood_emb(self.mood_lookup(inputs["mood"]))
        return tf.concat([t_v, c_v, k_v, m_v], axis=-1)

model = MovieModel()
_ = model({"title": tf.constant([str(df['title'].iloc[0])]), "cast": tf.constant([str(df['cast'].iloc[0])]), 
           "keywords": tf.constant([str(df['keywords'].iloc[0])]), "mood": tf.constant([str(df['mood'].iloc[0])])})

if os.path.exists('enhanced_movie_weights.weights.h5'):
    model.load_weights('enhanced_movie_weights.weights.h5', by_name=True, skip_mismatch=True)

# 4. OMDB HELPER
def get_omdb(title):
    clean = re.sub(r'\s*\(\d{4}\)', '', title).strip()
    try:
        r = requests.get(f"http://www.omdbapi.com/?t={clean}&apikey=58664459", timeout=2).json()
        if r.get('Response') == 'True':
            return {'poster': r.get('Poster'), 'plot': r.get('Plot'), 'rating': r.get('imdbRating')}
    except: pass
    return {'poster': 'https://via.placeholder.com/300x450', 'plot': 'No plot available.', 'rating': 'N/A'}

def prepare_section(data_df):
    results = []
    for _, row in data_df.iterrows():
        meta = get_omdb(row['title'])
        results.append({
            'title': row['title'], 'poster': meta['poster'], 
            'rating': row['rating'] if row['rating'] != '' else meta['rating'],
            'year': row['year']
        })
    return results
def get_enriched_list(dataframe):
    enriched = []
    for _, row in dataframe.iterrows():
        try:
            # 1. CSV se Poster uthao
            poster_from_csv = str(row.get('poster', '')).strip()
            
            # 2. Agar TMDB ka aadha link hai (/abc.jpg), toh use pura karo
            if poster_from_csv.startswith('/') and not poster_from_csv.startswith('http'):
                final_poster = f"https://image.tmdb.org/t/p/w500{poster_from_csv}"
            
            # 3. Agar CSV mein link hai hi nahi, toh OMDB se mango
            elif not poster_from_csv or poster_from_csv == 'nan' or poster_from_csv == 'N/A':
                meta = get_omdb(row['title'])
                final_poster = meta['poster'] if meta['poster'] != 'N/A' else 'https://via.placeholder.com/300x450?text=No+Poster'
            
            # 4. Agar CSV mein pura link hai, toh wahi use karo
            else:
                final_poster = poster_from_csv

            enriched.append({
                'movieId': row['movieId'],
                'title': row['title'],
                'year': row['year'],
                'mood': row['mood'],
                'Industry': row.get('Industry', 'Global'),
                'cast': row['cast'],
                'poster': final_poster,  # Final sanitized poster
                'rating': row['rating'] if row['rating'] != '' else '7.5'
            })
        except Exception as e:
            print(f"Error for {row['title']}: {e}")
            continue
    return enriched

@app.route('/')
def home():
    
    # 1. Sabse popular movie ko Banner ke liye select karein (Dynamic)
    trending_all = df.sort_values(by='popularity', ascending=False)
    
    # get_enriched_list ko 1 movie bhej rahe hain banner ke liye
    banner_data = get_enriched_list(trending_all.head(1))
    banner_movie = banner_data[0] if banner_data else None
    
    trending_list = get_enriched_list(trending_all.head(12))
    # Filter check: Ensure 'BOLLYWOOD' is capital in your CSV
    bollywood_df = df[df['Industry'].str.upper() == 'BOLLYWOOD'].sort_values(by='year', ascending=False).head(12)
    
    # Baki industries
    hollywood_df = df[df['Industry'].str.upper() == 'HOLLYWOOD'].head(12)
    chinese_df = df[df['Industry'].str.upper() == 'CHINESE'].head(12)
    korean_df = df[df['Industry'].str.upper() == 'KOREAN'].head(12)
    animation_df = df[df['Industry'].str.upper() == 'ANIME'].head(12)

    return render_template('index.html', 
                           banner=banner_movie,
                           bollywood=get_enriched_list(bollywood_df),
                           hollywood=get_enriched_list(hollywood_df),
                           chinese=get_enriched_list(chinese_df),
                           korean=get_enriched_list(korean_df),
                           animation=get_enriched_list(animation_df),
                           trending=get_enriched_list(df.sort_values(by='popularity', ascending=False).head(6)))


@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    if not query:
        return redirect(url_for('home'))

    # 1. TMDB API se data mangwana (Actors ke liye best hai)
    api_results = get_tmdb_results(query)

    # 2. Local CSV mein dhoondna (Current setup ke liye)
    local_results_df = df[
        df['title'].str.contains(query, case=False, na=False) | 
        df['cast'].str.contains(query, case=False, na=False)
    ].copy()
    local_results = get_enriched_list(local_results_df)

    # 3. SMART MERGE:
    # Agar API mein results mile hain (jaise actor search par milte hain), toh unhe dikhao
    # Par hum duplicates hatayenge taaki results clean rahein
    final_results = api_results 
    existing_titles = [r['title'].lower() for r in final_results]
    
    for movie in local_results:
        if movie['title'].lower() not in existing_titles:
            final_results.append(movie)

    # Sirf search.html bhej rahe hain, koi aur page edit nahi hoga
    return render_template('search.html', query=query, results=final_results)


@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form.get('movie_name')
    # Use exact match first, then fallback to contains
    target = df[df['title'].str.lower() == query.lower()].iloc[:1]
    if target.empty:
        target = df[df['title'].str.contains(query, case=False, na=False)].iloc[:1]
    
    if target.empty:
        return render_template('index.html', error="Movie not found!")

    # Vector calculation (Keeping your AI logic)
    query_input = {
        "title": tf.constant([str(target['title'].iloc[0])]),
        "cast": tf.constant([str(target['cast'].iloc[0])]),
        "keywords": tf.constant([str(target['keywords'].iloc[0])]),
        "mood": tf.constant([str(target['mood'].iloc[0])])
    }
    
    # Pre-calculating all vectors is better for performance
    all_inputs = {
        "title": tf.constant(df['title'].values.astype(str)),
        "cast": tf.constant(df['cast'].values.astype(str)),
        "keywords": tf.constant(df['keywords'].values.astype(str)),
        "mood": tf.constant(df['mood'].values.astype(str))
    }
    
    query_vec = model(query_input)
    all_vecs = model(all_inputs)
    
    # Dot Product Similarity
    scores = tf.matmul(query_vec, all_vecs, transpose_b=True).numpy()[0]
    
    # Boosting logic (Bhai, ye logic Viva ke liye best hai)
    target_ind = target['Industry'].iloc[0]
    target_mood = target['mood'].iloc[0]
    target_cast = set(str(target['cast'].iloc[0]).split("|"))

    for i in range(len(df)):
        # Region Match Boost
        if df.iloc[i]['Industry'] == target_ind: scores[i] *= 1.2
        # Actor Match Boost (High Importance)
        curr_cast = set(str(df.iloc[i]['cast']).split("|"))
        if target_cast.intersection(curr_cast): scores[i] *= 2.0
        # Mood Match Boost
        if df.iloc[i]['mood'] == target_mood: scores[i] *= 1.3

    indices = np.argsort(scores)[::-1][1:13] # Top 12 excluding itself
    
    # Enriched results for recommendation page
    results = get_enriched_list(df.iloc[indices])
    
    return render_template('index.html', results=results, query=query)

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    # 1. Current movie ka data nikalna
    target_movie = df[df['movieId'] == movie_id].iloc[:1]
    if target_movie.empty:
        return redirect(url_for('home'))
    
    # 2. AI Recommendation Logic (TensorFlow Model)
    query_input = {
        "title": tf.constant([str(target_movie['title'].iloc[0])]),
        "cast": tf.constant([str(target_movie['cast'].iloc[0])]),
        "keywords": tf.constant([str(target_movie['keywords'].iloc[0])]),
        "mood": tf.constant([str(target_movie['mood'].iloc[0])])
    }
    
    all_inputs = {
        "title": tf.constant(df['title'].values.astype(str)),
        "cast": tf.constant(df['cast'].values.astype(str)),
        "keywords": tf.constant(df['keywords'].values.astype(str)),
        "mood": tf.constant(df['mood'].values.astype(str))
    }
    
    # Model se vectors generate karna
    query_vec = model(query_input)
    all_vecs = model(all_inputs)
    
    # Similarity Scores (Dot Product)
    scores = tf.matmul(query_vec, all_vecs, transpose_b=True).numpy()[0]
    
    # --- 🔥 BALANCED BOOSTING LOGIC ---
    target_ind = str(target_movie['Industry'].iloc[0]).upper()
    target_mood = str(target_movie['mood'].iloc[0]).lower()
    target_cast = set(str(target_movie['cast'].iloc[0]).split("|"))
    
    for i in range(len(df)):
        # Same Industry (Bollywood/Hollywood) Match -> 2.5x Boost
        if str(df.iloc[i]['Industry']).upper() == target_ind:
            scores[i] *= 2.5
            
        # Mood Match (Epic/Action/Romantic) -> 3.0x Boost (Accuracy ke liye sabse imp)
        if str(df.iloc[i]['mood']).lower() == target_mood:
            scores[i] *= 3.0
            
        # Cast Match (Actors) -> 1.2x Boost (Ise kam rakha hai taaki genre na bigde)
        curr_cast = set(str(df.iloc[i]['cast']).split("|"))
        if target_cast.intersection(curr_cast):
            scores[i] *= 1.2
            
        # Same Movie: Khud ko list se hatao
        if df.iloc[i]['movieId'] == movie_id:
            scores[i] = 0
    # ----------------------------------

    # Top 12 relevant movies nikalna
    indices = np.argsort(scores)[::-1][:12]
    related_movies = get_enriched_list(df.iloc[indices])

    # 3. Movie Page Information
    movie_info = get_enriched_list(target_movie)[0]
    
    # YouTube Trailer Search Link
    clean_title = re.sub(r'\s*\(\d{4}\)', '', movie_info['title']).strip()
    search_query = f"{clean_title} official trailer".replace(" ", "+")
    movie_info['trailer_url'] = f"https://www.youtube.com/results?search_query={search_query}"
    
    return render_template('movie_page.html', movie=movie_info, related=related_movies)

if __name__ == '__main__':
    app.run(debug=True,port=5000)
