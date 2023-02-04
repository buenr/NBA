import catboost as cb
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.endpoints import cumestatsplayergames
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
import timeit
 
starttime = timeit.default_timer()
print("The start time is :",starttime)

html = requests.get("https://www.fantasypros.com/daily-fantasy/nba/fanduel-salary-changes.php").content
soup = BeautifulSoup(html, "html.parser")
table = soup.find("table")
prices_df = pd.read_html(str(table))[0]

prices_df['Name'] = prices_df['Player'].str.split('(').str[0].str[0:-1]
prices_df['Name'] = prices_df['Name'].str.replace('O.G.','OG')
prices_df['Name'] = prices_df['Name'].str.replace('R.J.','RJ')
prices_df['Name'] = prices_df['Name'].str.replace('A.J.','AJ')
prices_df['Name'] = prices_df['Name'].str.replace(' III','')
prices_df['Name'] = prices_df['Name'].str.replace(' II','')
prices_df['Name'] = prices_df['Name'].str.replace(' Jr.','')
prices_df['Name'] = prices_df['Name'].str.replace(' Sr.','')

prices_df['Team'] = prices_df['Player'].apply(lambda st: st[st.find("(")+1:st.find(" - ")])
prices_df['Team'] = prices_df['Team'].str.replace('UTH', 'UTA')
prices_df['Team'] = prices_df['Team'].str.replace('NOR', 'NOP')

prices_df['Pos'] = prices_df['Player'].apply(lambda st: st[st.find(" - ")+3:st.find(")")])
prices_df['Key'] = prices_df['Name'] + '-' + prices_df['Team']

prices_df['Price'] = prices_df['Today'].replace("[$,]", "", regex=True).astype(int)
prices_df.to_csv('Fanduel_Player_Prices_Clean.csv', index=False)
prices_df = prices_df[['Key','Price','Pos']]

# Get the player game logs for the current season
gamelogs = playergamelogs.PlayerGameLogs(season_nullable=Season.current_season,
                                         season_type_nullable=SeasonType.regular)

# Get the player game logs for the current season
cumestats = cumestatsplayergames.CumeStatsPlayerGames()

# Get the list of game IDs from the schedule
gamelogs = gamelogs.get_data_frames()[0]

gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace('O.G.','OG')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace('R.J.','RJ')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace('A.J.','AJ')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' III','')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' II','')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' Jr.','')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' Sr.','')

gamelogs['Home'] = gamelogs['MATCHUP'].str.find('vs.')
gamelogs['Home'] = np.where(gamelogs['Home'] > 0, 1, 0)

gamelogs['Opp'] = gamelogs['MATCHUP'].str[-3:]
gamelogs.to_csv('test.csv', index=False)

df = gamelogs[['PLAYER_ID','PLAYER_NAME','NBA_FANTASY_PTS','GAME_DATE','MIN','TEAM_ABBREVIATION','Opp','Home']].sort_values(by='GAME_DATE')

df['FANTASY_ROLLING2'] = df.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].rolling(window=2, min_periods=1, closed='left').mean().reset_index(0,drop=True)
df['MINS_ROLLING2'] = df.groupby('PLAYER_ID')['MIN'].rolling(window=2, min_periods=1, closed='left').mean().reset_index(0,drop=True)
df['FANTASY_ROLLING15'] = df.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].rolling(window=15, min_periods=1, closed='left').mean().reset_index(0,drop=True)
df['MINS_ROLLING15'] = df.groupby('PLAYER_ID')['MIN'].rolling(window=15, min_periods=1, closed='left').mean().reset_index(0,drop=True)
#df['FANTASY_PRED'] = df.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].shift(1)
df['Key'] = df['PLAYER_NAME'] + '-' + df['TEAM_ABBREVIATION']

df = df.dropna()

df2 = pd.merge(df, prices_df, how='left', on='Key')

feature_names = ['FANTASY_ROLLING2', 'MINS_ROLLING2','FANTASY_ROLLING15', 'MINS_ROLLING15', 'Home']

# Define the features and target
X = df2[feature_names]
y = df2['NBA_FANTASY_PTS']

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the catboost regressor
#reg = cb.CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', cat_features=['TEAM_ABBREVIATION','Opp','Pos'])
reg = cb.CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

# Fit the model to the training data
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Sort the dataframe by 'id' and 'date' in descending order
df2 = df2.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=[True, False])

# Keep only the first record for each 'id'
df_latest = df2.drop_duplicates(subset='PLAYER_ID', keep='first')

# Define the features you want to use for prediction
X_predict = df_latest[feature_names]

# Use the predict method of the xgboost model to make predictions
y_predict = reg.predict(X_predict)

# Add the predictions to the dataframe
df_latest['FANTASY_POINTS_PRED'] = y_predict
df_latest['Rate'] = 1000*df_latest['FANTASY_POINTS_PRED']/df_latest['Price']
df_latest.to_csv('test2.csv', index=False)

# Evaluate the model
from sklearn.metrics import mean_squared_error
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

# get feature importances
feature_importances = reg.get_feature_importance(type='PredictionValuesChange')

# print feature importances
for feature, importance in zip(feature_names, feature_importances):
    print(feature, ':', importance)

print("The time difference is :", timeit.default_timer() - starttime)