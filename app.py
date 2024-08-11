import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import numpy as np
from datetime import datetime
import requests
from PIL import Image
import streamlit as st
from io import BytesIO

st.header('Premier League matches prediction')
image_url = "https://static.dezeen.com/uploads/2016/08/designstudiopremier-league-rebrand-relaunch-logo-design-barclays-football_dezeen_slideshow-a-852x609.jpg"

# Download the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image in Streamlit
st.image(image, caption='Premier League logo', use_column_width=True)
st.write('''
- This app predicts the result of a Premier League match.
- The model is trained using data from the 2018/2019 to 2023/2024 season.
- The model uses the Logistic Regression algorithm.
\nPlease add the following features to predict if the team will win or lose the match :
''')


# Define a function to calculate additional features
def calculate_features(data, historical_data):
    """
    Calculate additional features based on user input and historical data.
    """
    # Convert date and time to datetime and extract additional features
    match_datetime = datetime.combine(data['date'], data['time'])
    data['day_of_week'] = match_datetime.weekday()

    # Calculate time condition
    hour = match_datetime.hour
    if hour < 12:
        data['time_condition'] = 'early'
    elif hour < 17:
        data['time_condition'] = 'afternoon'
    else:
        data['time_condition'] = 'evening'

    # Calculate days since last match
    last_match_date = historical_data[historical_data['team']
                                      == data['team']]['date'].max()
    data['days_since_last_match'] = (
        match_datetime.date() - pd.to_datetime(last_match_date).date()).days

    # Calculate rolling averages and other metrics from historical data
    recent_matches = historical_data[historical_data['team']
                                     == data['team']].sort_values('date').tail(5)
    data['rolling_xg'] = recent_matches['xg'].mean()
    data['rolling_xga'] = recent_matches['xga'].mean()
    data['rolling_poss'] = recent_matches['poss'].mean()
    data['rolling_sh'] = recent_matches['sh'].mean()
    data['rolling_sot'] = recent_matches['sot'].mean()
    data['rolling_dist'] = recent_matches['dist'].mean()
    data['rolling_goal_diff'] = (
        recent_matches['gf'] - recent_matches['ga']).mean()

    # Calculate form
    recent_results = recent_matches['result'].map({'W': 3, 'D': 1, 'L': 0})
    data['form'] = recent_results.mean()

    # Calculate head-to-head record
    h2h_matches = historical_data[
        ((historical_data['team'] == data['team']) & (historical_data['opponent'] == data['opponent'])) |
        ((historical_data['team'] == data['opponent']) &
         (historical_data['opponent'] == data['team']))
    ].sort_values('date').tail(5)
    data['h2h_record'] = h2h_matches[h2h_matches['team'] == data['team']]['result'].map(
        {'W': 1, 'D': 0.5, 'L': 0}).mean()

    # Calculate ratios from historical data
    season_data = historical_data[historical_data['team'] == data['team']]
    data['fk_ratio'] = season_data['fk'].sum() / season_data['sh'].sum()
    data['pk_per_shot'] = season_data['pkatt'].sum() / season_data['sh'].sum()
    data['fk_percentage'] = data['fk_ratio'] * 100
    data['pk_per_shot_percentage'] = data['pk_per_shot'] * 100

    return data


input_data = {}
input_data['team'] = st.selectbox('Home Team',
                                  ('Manchester City', 'Manchester United', 'Liverpool', 'Chelsea',
                                   'Leicester City', 'West Ham United', 'Tottenham Hotspur', 'Arsenal',
                                   'Leeds United', 'Everton', 'Aston Villa', 'Newcastle United',
                                   'Wolverhampton Wanderers',
                                   'Crystal Palace', 'Southampton', 'Brighton and Hove Albion', 'Burnley',
                                   'Fulham', 'West Bromwich Albion', 'Sheffield United', 'Bournemouth',
                                   'Brentford', 'Nottingham Forest', 'Luton Town', 'Watford', 'Norwich City'))

input_data['opponent'] = st.selectbox('Opponent Team',
                                      ('Wolves', 'Leicester City', 'Leeds United', 'Arsenal', 'West Ham',
                                       'Sheffield Utd', 'Liverpool', 'Tottenham', 'Burnley', 'Fulham',
                                       'Manchester Utd', 'West Brom', 'Southampton', 'Newcastle Utd',
                                       'Chelsea', 'Brighton', 'Crystal Palace', 'Aston Villa', 'Everton',
                                       'Manchester City', 'Norwich City', 'Bournemouth', 'Watford',
                                       "Nott'ham Forest", 'Luton Town', 'Brentford'))

input_data['date'] = st.date_input('Match Date')
input_data['time'] = st.time_input("Kick-off time")
input_data['venue'] = st.selectbox('Home or Away', ('Home', 'Away'))
input_data['formation'] = st.selectbox('Expected home team formation',
                                       ('4-2-3-1', '4-3-3', '3-1-4-2', '3-4-3', '4-4-1-1', '4-1-2-1-2',
                                        '4-3-1-2', '4-4-2', '4-2-2-2', '3-4-1-2', '3-5-2', '4-1-4-1',
                                        '5-4-1', '3-3-3-1', '4-3-2-1', '5-3-2', '3-5-1-1', '4-1-3-2',
                                        '4-5-1', '3-2-4-1', '4-2-4'))
captains = pd.read_csv("captains.csv")
input_data['captain'] = st.selectbox(
    'Home Team Captain', captains[captains['team'] == input_data['team']]['captain'].values)

old_data = pd.read_csv("matches_final.csv")

# Calculate additional features
result = calculate_features(input_data, old_data)

input_data = pd.DataFrame(input_data, index=[0])
result = pd.DataFrame(result, index=[0])
features = pd.concat([input_data, result], axis=0)

with open('grad_boost.pkl', 'rb') as f:
    model = pickle.load(f)

    pred_prob = model.predict_proba(features)
    pred = model.predict(features)

    st.subheader('Prediction')
    if pred[0] == 1:
        st.write('The home team will win the match.')
    else:
        st.write('The home team will lose the match.')

    # Ensure probabilities sum to 1
    st.write('Probability of home team winning: {:.2f}%'.format(
        pred_prob[0][1] * 100))
    st.write('Probability of home team losing: {:.2f}%'.format(
        pred_prob[0][0] * 100))
