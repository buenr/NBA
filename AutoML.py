import autosklearn.regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
import timeit
import joblib

#automl = joblib.load('automl_model.pkl')

starttime = timeit.default_timer()
print("The start time is :",starttime)

# Get the player game logs for the current season
gamelogs = playergamelogs.PlayerGameLogs(season_nullable=Season.current_season,
                                         season_type_nullable=SeasonType.regular)

# Get the list of game IDs from the schedule
gamelogs = gamelogs.get_data_frames()[0]

gamelogs = gamelogs.sort_values(by='GAME_DATE',ascending=True)
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace('O.G.','OG')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace('R.J.','RJ')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace('A.J.','AJ')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' III','')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' II','')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' Jr.','')
gamelogs['PLAYER_NAME'] = gamelogs['PLAYER_NAME'].str.replace(' Sr.','')
#gamelogs['Home'] = gamelogs['MATCHUP'].str.find('vs.')
#gamelogs['Home'] = np.where(gamelogs['Home'] > 0,1,0)
#gamelogs['Opp'] = gamelogs['MATCHUP'].str[-3:]
gamelogs['Shots'] = gamelogs['FGA'] + gamelogs['FTA']/2
gamelogs['cume_FPTS'] = gamelogs.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].cumsum()
gamelogs['cume_MINS'] = gamelogs.groupby('PLAYER_ID')['MIN'].cumsum()
gamelogs['cume_FPTSPERMIN'] = gamelogs['cume_FPTS']/gamelogs['cume_MINS']

#df_test = gamelogs[gamelogs['PLAYER_NAME'] == 'Ben Simmons']

df = gamelogs[['PLAYER_ID','PLAYER_NAME','NBA_FANTASY_PTS','GAME_DATE','MIN','PTS','REB','AST','Shots','TEAM_ABBREVIATION','cume_FPTSPERMIN']]

df['FANTASY_ROLLING1'] = df.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].rolling(window=1,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['MINS_ROLLING1'] = df.groupby('PLAYER_ID')['MIN'].rolling(window=1,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['SHOTS_ROLLING1'] = df.groupby('PLAYER_ID')['Shots'].rolling(window=1,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['PTS_ROLLING1'] = df.groupby('PLAYER_ID')['PTS'].rolling(window=1,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['REB_ROLLING1'] = df.groupby('PLAYER_ID')['REB'].rolling(window=1,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['AST_ROLLING1'] = df.groupby('PLAYER_ID')['AST'].rolling(window=1,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['FPTSPERMIN1'] = df['cume_FPTSPERMIN']*df['MINS_ROLLING1']

df['FANTASY_ROLLING2'] = df.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].rolling(window=2,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['MINS_ROLLING2'] = df.groupby('PLAYER_ID')['MIN'].rolling(window=2,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['SHOTS_ROLLING2'] = df.groupby('PLAYER_ID')['Shots'].rolling(window=2,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['PTS_ROLLING2'] = df.groupby('PLAYER_ID')['PTS'].rolling(window=2,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['REB_ROLLING2'] = df.groupby('PLAYER_ID')['REB'].rolling(window=2,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['AST_ROLLING2'] = df.groupby('PLAYER_ID')['AST'].rolling(window=2,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['FPTSPERMIN2'] = df['cume_FPTSPERMIN']*df['MINS_ROLLING2']

df['FANTASY_ROLLING15'] = df.groupby('PLAYER_ID')['NBA_FANTASY_PTS'].rolling(window=15,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['MINS_ROLLING15'] = df.groupby('PLAYER_ID')['MIN'].rolling(window=15,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['SHOTS_ROLLING15'] = df.groupby('PLAYER_ID')['Shots'].rolling(window=15,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['PTS_ROLLING15'] = df.groupby('PLAYER_ID')['PTS'].rolling(window=15,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['REB_ROLLING15'] = df.groupby('PLAYER_ID')['REB'].rolling(window=15,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['AST_ROLLING15'] = df.groupby('PLAYER_ID')['AST'].rolling(window=15,min_periods=1,closed='left').mean().reset_index(0,drop=True)
df['FPTSPERMIN15'] = df['cume_FPTSPERMIN']*df['MINS_ROLLING15']

df['Key'] = df['PLAYER_NAME'] + '-' + df['TEAM_ABBREVIATION']
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
#df_test = df[df['PLAYER_NAME'] == 'Ben Simmons']

df = df.dropna()

df_prop = pd.read_csv('odds_data.csv')
df_prop['run_dt'] = pd.to_datetime(df_prop['run_ts']).dt.date

df2 = pd.merge(df,df_prop,how='inner',left_on=['Key','GAME_DATE'],right_on=['Key','run_dt'])

feature_names = ['SHOTS_ROLLING1','FANTASY_ROLLING1','MINS_ROLLING1','FPTSPERMIN1','SHOTS_ROLLING2','FANTASY_ROLLING2','MINS_ROLLING2','FPTSPERMIN2','SHOTS_ROLLING15','FANTASY_ROLLING15','MINS_ROLLING15','FPTSPERMIN15','Prop_pts']
#,'TEAM_ABBREVIATION','Opp'

# Define the features and target
X = df2[feature_names]
y = df2['NBA_FANTASY_PTS']

# Split data into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Step 2: Instantiate an AutoSklearnRegressor
automl = autosklearn.regression.AutoSklearnRegressor()
#automl = autosklearn.regression.AutoSklearnRegressor(resampling_strategy='cv', ensemble_size=1,time_left_for_this_task = 60,per_run_time_limit=30)

# Step 3: Fit the model to the training data
automl.fit(X_train,y_train)
#automl.leaderboard()
#models_with_weights = automl.get_models_with_weights()
#models_with_weights.sort(key=lambda x: x[1], reverse=True)
#joblib.dump(models_with_weights[0][0], 'best_model.pkl')

# Step 4: Predict fantasy points using the test data
y_pred = automl.predict(X_test)

# Step 5: Evaluate the performance of the model
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)

max_dt = df_prop['run_dt'].max()
df_prop_latest = df_prop[df_prop['run_dt']==max_dt]

# Keep only the first record for each 'id'
df_latest = df.sort_values(by=['PLAYER_ID','GAME_DATE'],ascending=[True,False])
df_latest = df_latest.drop_duplicates(subset='PLAYER_ID',keep='first')
df_latest = pd.merge(df_latest,df_prop_latest,how='inner',left_on=['Key'],right_on=['Key'])

# Define the features you want to use for prediction
X_predict = df_latest[feature_names]

# Use the predict method of the xgboost model to make predictions
y_predict = automl.predict(X_predict)

# Add the predictions to the dataframe
df_latest['FANTASY_POINTS_PRED'] = y_predict

#Price data
df_prices = pd.read_csv('prices_data.csv')
df_prices = df_prices.sort_values(by=['run_ts'],ascending=[False])
df_prices = df_prices.drop_duplicates(subset='Key',keep='first')
df_latest2 = pd.merge(df_latest,df_prices,how='left',on='Key')

df_latest2['Rate'] = 1000*df_latest2['FANTASY_POINTS_PRED']/df_latest2['Price']
df_latest2.to_csv('predictions.csv',index=False)

print("The time difference is :",timeit.default_timer() - starttime)

# Save the model to a file
joblib.dump(automl, 'automl_model.pkl')