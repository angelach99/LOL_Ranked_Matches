#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:55:08 2021

@author: hong
"""


# Load Libraries
import pandas
import numpy as np


########################
##### Import data ######
########################

## Import matches info
matches = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/matches.csv')

## Narrow down the scope
## scope: NA game, season 8
matches = matches.loc[matches['platformid'] == "NA1"]
matches = matches.loc[matches["seasonid"] == 8]

match_id = matches.id.values.tolist() ## list of season 8 NA games

## Import participations info
players = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/participants.csv')

player_index = []

for i in range(0,len(players.matchid)):
    if players.matchid[i] in match_id:
        player_index.append(i)

players = players.loc[player_index]

## import team stats info
teamstats = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/teamstats.csv')
team_index = []

for i in range(0,len(teamstats.matchid)):
    if teamstats.matchid[i] in match_id:
        team_index.append(i)

teamstats = teamstats.loc[team_index]

## import stats info
stats = pandas.read_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/stats2.csv')
## stats data is sort by id, so no need to import stat1 
## because id in player data set is also sorted by id
players.id.min() == players.id.iloc[0] ## return ture
index = players.id.iloc[0]

xx = stats.index[stats['id'] == index]
stats = stats[stats.index >= xx[0]] 

index2 = players.id.iloc[16616-1]
yy = stats.index[stats['id'] == index2]
stats = stats[stats.index <= yy[0]]

player_lis = players.id.values.tolist()
stats_index = []

for i in range(0,len(stats.id)):
    if stats.id.iloc[i] in player_lis:
        stats_index.append(i)
stats = stats.iloc[stats_index]



## export
matches.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/matches_NA_S8.csv')
players.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/players_NA_S8.csv')
teamstats.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/teamstats_NA_S8.csv')
stats.to_csv('/Users/hong/Desktop/McGill/MGSC661/LOL_Dat/stats_NA_S8.csv')