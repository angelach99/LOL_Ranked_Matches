## LOL Ranked Matches Analysis Report
MGSC661 Final Report <br>
Fall 2021 @McGill

## Abstract
The League of Legends (LOL) is a multiplayer online battle arena game developed by Riot
in 2009. In each match, 10 players are divided into two teams, red and blue. In its most played
mode, the Summoner’s Rift, each player needs to control a character known as champion to
defend their team’s base and invade the other half of the map. A team wins by destroying the
Nexus in front of the enemy team’s base.

League of Legends has been the most played game in the world since 2012. Each year it
will hold a ranked season where players play with or against others that share a similar level of
gaming abilities. The annually ranked seasons first started in July 2010 and right now LOL is
approaching the end of season 11. During a ranked season, the points will be added or deducted
if a player succeeds or loses in one rank game. The final points will decide the annual ranking
of a player. Currently, the League of Legends ranking system has 9 tires and 4 divisions within
each tire. However, before season 9 there were only 7 tires and 5 divisions in each tire.

In this project, we used the League of Legend Ranked Matches dataset to explore the
hidden information behind ranked games. Using classification models, we predicted the
outcome of a game using game statistics such as the kill death assist ratio. We also conducted
cluster analysis on the matches. By mapping the clustering result to the tires based on clusters’
characteristics and size, we then analyzed the similarity and differences among tires.

Since the origin dataset comprised 184,070 ranked solo games across various seasons and
platforms, we restrict our scope of work to analyzing North American platform Season 8
matches. However, the idea behind this project can be applied to the whole dataset.

## Data
The dataset was downloaded from [LOL Ranked Matches from Kaggle](https://www.kaggle.com/paololol/league-of-legends-ranked-matches). 

The original dataset comprised of 7 tables, but the datasets were manipulated into two
final matches statistics tables. <br>
[Final Dataset](https://github.com/angelach99/LOL_Ranked_Matches/blob/master/data_preprocessing/final_match.csv) <br>
[Final Dataset Normalized](https://github.com/angelach99/LOL_Ranked_Matches/blob/master/data_preprocessing/final_match_normalized.csv)

## Overview
[Data Preprocessing 1](https://github.com/angelach99/LOL_Ranked_Matches/blob/master/data_preprocessing/LOL_data_preprocessing.py)

[Data Preprocessing 2 (Extraction)](https://github.com/angelach99/LOL_Ranked_Matches/blob/master/data_preprocessing/lol_data_extraction.py)   

[The League of Legends Ranked Matches Analysis Report](https://github.com/angelach99/LOL_Ranked_Matches/blob/master/MGSC661%20Final%20Project%20--%20The%20League%20of%20Legends%20Ranked%20Matches%20Analysis%20Report.pdf)

[Project analysis coding](https://github.com/angelach99/LOL_Ranked_Matches/blob/master/MGSC661%20Final%20Project.R)

