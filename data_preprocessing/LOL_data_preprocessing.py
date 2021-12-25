#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:49:23 2021

@author: hong
"""

import pandas

## Import Dataset
players = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/players_NA_S8.csv')
stats = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/stats_NA_S8.csv')

id_to_match = []

for i in range(0,len(players.id)):
    if players.id[i] == stats.id[i]:
        id_to_match.append(players.matchid[i])
        continue
    
id_to_team = []

for i in range(0,len(players.id)):
    if players.player[i] <= 5:
        id_to_team.append(100)
        continue
    else:
        id_to_team.append(200)
        continue
        

stats["matchid"] = id_to_match 
stats["teamid"] = id_to_team
stats = stats.drop(columns = 'Unnamed: 0')
stats.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/stats_NA_S8.csv',index = False)

df = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/stats_NA_S8.csv')
kda = []
for i in range(0,len(df.id)):
    kda.append((df.win[i]+df.assists[i])/(df.kills[i]+1)) ## add one to avoid divide by 0 issue

df['kda'] = kda

## drop irrelevant columns
## only want certain columns
## 'id', 'win','kills', 'deaths', 'assists','totdmgdealt','totdmgtochamp','totdmgtaken',
## 'goldearned','goldspent','totminionskilled','neutralminionskilled','champlvl','matchid'

df = df[['id', 'matchid','teamid','win','kda','totdmgdealt','totdmgtochamp','totdmgtaken','goldearned','goldspent','totminionskilled','neutralminionskilled','champlvl']]
  
df.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/stats_short_NA_S8.csv',index = False)

## calculate avg team stats in a match 

avg_dat = df.groupby(["matchid","teamid"]).mean()
avg_dat = avg_dat.drop(columns = 'id')
avg_dat.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/matches_avg_stats_NA_S8.csv')

df = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/final_match.csv')
matches = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/matches_NA_S8.csv')

match_to_duration = []
for i in range(0,len(df.matchid)):
    x =  matches.index[matches["id"] == df.matchid[i]]
    match_to_duration.append(matches.loc[x,"duration"].values[0])
        
df["duration"] = match_to_duration
df.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/final_match.csv',index = False)

df['totdmgdealt'] = df['totdmgdealt']/df["duration"]

df['totdmgtochamp'] = df['totdmgtochamp']/df["duration"]
df['totdmgtaken'] = df['totdmgtaken']/df["duration"]
df['goldearned'] = df['goldearned']/df["duration"]
df['goldspent'] = df['goldspent']/df["duration"]
df['totminionskilled'] = df['totminionskilled']/df["duration"]
df['neutralminionskilled'] = df['neutralminionskilled']/df["duration"]
df['champlvl'] = df['champlvl']/df["duration"]

df['towerkills'] = df['towerkills']/df["duration"]
df['inhibkills'] = df['inhibkills']/df["duration"]
df['baronkills'] = df['baronkills']/df["duration"]
df['dragonkills'] = df['dragonkills']/df["duration"]
df['harrykills'] = df['harrykills']/df["duration"]

df.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/final_match_normalized.csv',index = False)
