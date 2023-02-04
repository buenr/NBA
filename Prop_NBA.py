#Get FantasyPros rankings and load to df
import pandas as pd
import glob
import os 
import numpy as np
from pulp import lpSum, LpProblem, LpMaximize, LpBinary, LpInteger, LpVariable, LpStatus, value
from datetime import datetime

# Use glob to create a list of file paths for all .csv files in the specified folder
path = 'C:/Users/RNA/Downloads/*.csv'
csv_files = glob.glob(path)
df2 = pd.DataFrame()

for file in csv_files:
    if 'nba-player-props-rotowire' in file:
        
        # Read in the file
        df_temp = pd.read_csv(file, skiprows=1)
        stat = df_temp.columns[3]
        
        bet_const = 115
        bet_div = 30
        
        if stat in ['Rebounds','Assists']:
            bet_div = 45
        
        if stat in ['Turnovers']:
            bet_div = 60
        
        if stat in ['Steals','Blocks']:
            bet_div = 90
        
        print(stat)
        
        columns_to_check = ['Over', 'Under']
        df_temp['Greater'] = df_temp[columns_to_check].abs().idxmax(axis=1)
        df_temp['Greater_Value'] = df_temp[columns_to_check].abs().max(axis=1)
        df_temp['Change'] = np.where(df_temp['Greater']=='Over',1,-1)
        df_temp[stat] = df_temp[stat] + (df_temp['Change']*(df_temp['Greater_Value'].abs()-bet_const)/bet_div)
        
        columns_to_check = ['Over.1', 'Under.1']
        df_temp['Greater.1'] = df_temp[columns_to_check].abs().idxmax(axis=1)
        df_temp['Greater_Value.1'] = df_temp[columns_to_check].abs().max(axis=1)
        df_temp['Change.1'] = np.where(df_temp['Greater.1']=='Over.1',1,-1)
        df_temp[stat+'.1'] = df_temp[stat+'.1'] + (df_temp['Change.1']*(df_temp['Greater_Value.1'].abs()-bet_const)/bet_div)
        
        columns_to_check = ['Over.2', 'Under.2']
        df_temp['Greater.2'] = df_temp[columns_to_check].abs().idxmax(axis=1)
        df_temp['Greater_Value.2'] = df_temp[columns_to_check].abs().max(axis=1)
        df_temp['Change.2'] = np.where(df_temp['Greater.2']=='Over.2',1,-1)
        df_temp[stat+'.2'] = df_temp[stat+'.2'] + (df_temp['Change.2']*(df_temp['Greater_Value.2'].abs()-bet_const)/bet_div)
        
        columns_to_check = ['Over.3', 'Under.3']
        df_temp['Greater.3'] = df_temp[columns_to_check].abs().idxmax(axis=1)
        df_temp['Greater_Value.3'] = df_temp[columns_to_check].abs().max(axis=1)
        df_temp['Change.3'] = np.where(df_temp['Greater.3']=='Over.3',1,-1)
        df_temp[stat+'.3'] = df_temp[stat+'.3'] + (df_temp['Change.3']*(df_temp['Greater_Value.3'].abs()-bet_const)/bet_div)

        #df_temp.columns = ['Player','Team','Opp',str(stat),'Over1','Under1','Remove1','Over2','Under2','Remove2','Over2','Under2','Remove','Over3','Under3']
        
        # Concatenate the dataframes
        df2 = pd.concat([df2, df_temp])
        os.remove(file)

df3 = df2[['Player','Team','Opp','Rebounds','Rebounds.1','Rebounds.2','Rebounds.3','Points','Points.1','Points.2','Points.3','Assists','Assists.1','Assists.2','Assists.3','Turnovers','Turnovers.1','Turnovers.2','Turnovers.3','Steals','Steals.1','Steals.2','Steals.3','Blocks','Blocks.1','Blocks.2','Blocks.3']]
df3[['Rebounds','Rebounds.1','Rebounds.2','Rebounds.3','Points','Points.1','Points.2','Points.3','Assists','Assists.1','Assists.2','Assists.3','Turnovers','Turnovers.1','Turnovers.2','Turnovers.3','Steals','Steals.1','Steals.2','Steals.3','Blocks','Blocks.1','Blocks.2','Blocks.3']] = df3[['Rebounds','Rebounds.1','Rebounds.2','Rebounds.3','Points','Points.1','Points.2','Points.3','Assists','Assists.1','Assists.2','Assists.3','Turnovers','Turnovers.1','Turnovers.2','Turnovers.3','Steals','Steals.1','Steals.2','Steals.3','Blocks','Blocks.1','Blocks.2','Blocks.3']].clip(lower=0)

df3 = df3.groupby(['Player','Team','Opp']).min().reset_index()

df3['Rebounds'] = df3[['Rebounds','Rebounds.1','Rebounds.2','Rebounds.3']].mean(axis=1)
df3['Points'] = df3[['Points','Points.1','Points.2','Points.3']].mean(axis=1)
df3['Assists'] = df3[['Assists','Assists.1','Assists.2','Assists.3']].mean(axis=1)
df3['Blocks'] = df3[['Blocks','Blocks.1','Blocks.2','Blocks.3']].mean(axis=1)
df3['Steals'] = df3[['Steals','Steals.1','Steals.2','Steals.3']].mean(axis=1)
df3['Turnovers'] = df3[['Turnovers','Turnovers.1','Turnovers.2','Turnovers.3']].mean(axis=1)

df3['Rebounds_pts'] = df3['Rebounds']*1.2
df3['Points_pts'] = df3['Points']
df3['Assists_pts'] = df3['Assists']*1.5
df3['Blocks_pts'] = df3['Blocks']*3
df3['Steals_pts'] = df3['Steals']*3
df3['Turnovers_pts'] = df3['Turnovers']

df3['Prop_pts'] = df3['Rebounds_pts'].fillna(0)+df3['Points_pts'].fillna(0)+df3['Assists_pts'].fillna(0)+df3['Blocks_pts'].fillna(0)+df3['Steals_pts'].fillna(0)-df3['Turnovers_pts'].fillna(0)
df3['Key'] = df3['Player'] + '-' + df3['Team']
df3['run_ts'] = datetime.now()
#df3.to_csv('odds_data.csv', index=0)
df3.to_csv('odds_data.csv', mode='a', index=0, header=False)

import requests
from bs4 import BeautifulSoup

html = requests.get("https://www.fantasypros.com/daily-fantasy/nba/fanduel-salary-changes.php").content
soup = BeautifulSoup(html, "html.parser")
table = soup.find("table")
prices_df = pd.read_html(str(table))[0]

prices_df['Name'] = prices_df['Player'].str.split('(').str[0].str[0:-1]
prices_df['Name'] = prices_df['Name'].str.replace(' Jr.','')
prices_df['Name'] = prices_df['Name'].str.replace(' Sr.','')
prices_df['Name'] = prices_df['Name'].str.replace(' III','')

prices_df['Team'] = prices_df['Player'].apply(lambda st: st[st.find("(")+1:st.find(" - ")])
prices_df['Team'] = prices_df['Team'].str.replace('UTH', 'UTA')
prices_df['Team'] = prices_df['Team'].str.replace('NOR', 'NOP')
prices_df['Team'] = prices_df['Team'].str.replace('PHO', 'PHX')

prices_df['Pos'] = prices_df['Player'].apply(lambda st: st[st.find(" - ")+3:st.find(")")])
prices_df['Key'] = prices_df['Name'] + '-' + prices_df['Team']

prices_df['Price'] = prices_df['Today'].replace("[$,]", "", regex=True).astype(int)

#prices_df.to_csv('Fanduel_Player_Prices_Clean.csv', index=False)
prices_df['run_ts'] = datetime.now()
#prices_df.to_csv('prices_data.csv', index=0)
prices_df.to_csv('prices_data.csv', mode='a', index=False, header=False)

df4 = pd.merge(df3, prices_df, how='left', on='Key')
df4['Value'] = df4['Prop_pts']/df4['Price']*1000 

players = df4[['Key','Price','Pos','Prop_pts']]
players[['Pos 1', 'Pos 2']] = players['Pos'].str.split("/", expand=True)
players = players.drop(columns=['Pos'])
players['player'] = players.index
players = players[players['Price'] < 9900]

# create the optimization model
model = LpProblem("Team Optimization", LpMaximize)

# create decision variables for each player
player_vars = {player: LpVariable(f"{player}_selected", 0, 1, LpBinary) for player in players['player']}

# create decision variables for each position
pg_var = LpVariable("PG_selected", 2, 2, LpInteger)
sg_var = LpVariable("SG_selected", 2, 2, LpInteger)
sf_var = LpVariable("SF_selected", 2, 2, LpInteger)
pf_var = LpVariable("PF_selected", 2, 2, LpInteger)
c_var = LpVariable("C_selected", 1, 1, LpInteger)

# add the salary cap constraint
model += lpSum([players.loc[i, 'Price'] * player_vars[i] for i in players.index]) <= 60000

# add the position constraint
model += lpSum([player_vars[i] for i in players.index if players.loc[i, 'Pos 1'] == 'PG']) == pg_var
model += lpSum([player_vars[i] for i in players.index if players.loc[i, 'Pos 1'] == 'SG']) == sg_var
model += lpSum([player_vars[i] for i in players.index if players.loc[i, 'Pos 1'] == 'SF']) == sf_var
model += lpSum([player_vars[i] for i in players.index if players.loc[i, 'Pos 1'] == 'PF']) == pf_var
model += lpSum([player_vars[i] for i in players.index if players.loc[i, 'Pos 1'] == 'C']) == c_var

# set the objective function
model += lpSum([players.loc[i, 'Prop_pts'] * player_vars[i] for i in players.index]) + lpSum([pg_var, sg_var, sf_var, pf_var, c_var]), "Total Points"

#solve the model
model.solve()

#print the results
print("Status:", LpStatus[model.status])
for player, var in player_vars.items():
    if var.varValue == 1:
        print("Player selected:", player)

# print the results
print("Number of PGs selected:", pg_var.varValue)
print("Number of SGs selected:", sg_var.varValue)
print("Number of SFs selected:", sf_var.varValue)
print("Number of PFs selected:", pf_var.varValue)
print("Number of Cs selected:", c_var.varValue)

# print the total salary and points of the selected team
print("Total Salary:", value(lpSum([players.loc[i, 'Price'] * player_vars[i] for i in players.index])))
print("Total Points:", value(lpSum([players.loc[i, 'Prop_pts'] * player_vars[i] for i in players.index])))

player_list = []

# Print the selected players
for player in players.player:
    if player_vars[player].varValue == 1.0:
        player_list.append(player)
        #print(player)

print(players.loc[player_list])