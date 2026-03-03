import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load datasets
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# Keep only relevant seasons (optional but recommended)
matches = matches[matches['dl_applied'] == 0]

# Merge datasets
df = deliveries.merge(matches[['id','city','winner']], 
                      left_on='match_id', 
                      right_on='id')

# total runs per ball (already exists in deliveries.csv)
df['total_runs'] = df['total_runs']

# Get 2nd innings only
df = df[df['inning'] == 2]

# Calculate cumulative score
df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()

# Get total runs of 1st innings
total_runs_df = df.groupby('match_id')['total_runs'].sum().reset_index()
total_runs_df.rename(columns={'total_runs':'target'}, inplace=True)

df = df.merge(total_runs_df, on='match_id')

# Feature Engineering
df['runs_left'] = df['target'] - df['current_score']
df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
df['wickets_left'] = 10 - df.groupby('match_id')['player_dismissed'].cumcount()
df['crr'] = df['current_score'] / (df['over'] + 1)
df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

# Drop invalid rows
df = df[df['balls_left'] > 0]
df = df[df['runs_left'] >= 0]

# Result column
df['result'] = df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

# Select required columns
final_df = df[['batting_team','bowling_team','city',
               'current_score','wickets_left',
               'balls_left','runs_left',
               'crr','rrr','result']]

# Train-test split
X = final_df.drop('result', axis=1)
y = final_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column separation
categorical_cols = ['batting_team','bowling_team','city']
numerical_cols = ['current_score','wickets_left',
                  'balls_left','runs_left',
                  'crr','rrr']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
pickle.dump(pipeline, open("model.pkl","wb"))

# Save teams & cities
pickle.dump(sorted(final_df['batting_team'].unique()), open("team.pkl","wb"))
pickle.dump(sorted(final_df['city'].dropna().unique()), open("city.pkl","wb"))

print("Model trained and saved successfully.")