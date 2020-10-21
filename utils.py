# imports
import requests
from pickle import load, dump
from pathlib import Path
from heapq import nlargest, nsmallest
import matplotlib.pyplot as plt
import numpy as np

# global variables - if you edit these, you need to delete players_data.p and let program regenerate
global_start_year = 2000
global_end_year = 2019

class Player:
    def __init__(self, player_name, player_position):
        self.player_name = player_name
        self.player_position = player_position
        self.scores = dict() # maps year -> (week, score that week)

    def playedInYear(self, year):
        return year in self.scores

    def add_score(self, year, week, score):
        if year not in self.scores:
            self.scores[year] = []
        
        self.scores[year].append((week, score))
    
    def get_score(self, year, week):
        if year not in self.scores or week not in range(1, 17):
            return -1
        
        return [tup[1] for tup in self.scores[year] if tup[0] == week][0]
    def get_mean_score_for_year(self, year):
        if year not in self.scores or len(self.scores[year]) == 0: 
            return -1
        return np.mean(self.scores[year], axis=0)[-1]
    
    def get_stddev_for_year(self, year):
        if year not in self.scores or len(self.scores[year]) == 0: 
            return -1
        return np.std(self.scores[year], axis=0)[-1]

    def get_total_score_for_year(self, year):
        if year not in self.scores or len(self.scores[year]) == 0: 
            return -1
        return np.sum(self.scores[year], axis=0)[-1]

    def get_mean_score_for_career(self):
        return np.mean([x for item in list(self.score.values()) for x in item], axis=0)[-1]
    
    def get_stddev_for_career(self):
        return np.std([x for item in list(self.score.values()) for x in item], axis=0)[-1]

    def get_total_score_for_career(self):
        return np.sum([x for item in list(self.score.values()) for x in item], axis=0)[-1]

    @staticmethod
    def score(fumbles_lost, passing_ints, passing_tds, passing_yds, receiving_tds, receiving_yds, receptions, rushing_tds, rushing_yds):
        total_points = 0.0
    
        # TDs
        total_points += 6 * (rushing_tds + receiving_tds)
        total_points += 4 * passing_tds

        # Yards + Receptions
        total_points += passing_yds / 25.0
        total_points += (rushing_yds + receiving_yds) / 10.0
        total_points += receptions / 2.0
        
        # Fumbles + Interceptions
        total_points -= 2 * fumbles_lost
        total_points -= passing_ints

        return round(total_points, 3)


def get_average_points_per_position(position, players, topn, plot_output=False, print_output=False):
    """
    See if the AVT theory holds true - calculates means/std dev of the 1-topn best players at each position
    """
    # populate a list of values for the top N players at that position to perform statistics on later 
    top_n_players = [[] for _ in range(topn)]
    for year in range(global_start_year, global_end_year):
        total_scores_for_year = [(players[player].get_total_score_for_year(year), player, year) for player in players if players[player].player_position == position]
        top_n_pos = nlargest(topn, total_scores_for_year)
        for (ind, elem) in enumerate(top_n_pos):
            top_n_players[ind].append((elem[1], elem[2]))

    # calculate statistics
    x = []
    y = []
    e_season = []
    e_game = []
    for (ind, top_n_scores) in enumerate(top_n_players):
        # model = each season is a normally distributed variable, with each game as a sample
        # mean = get mean of all the total scores per player (storing player and year for top n scores)
        # std dev = (assumption is all seasons are independent) sum variances and square root sum
        # std dev per game = random sample, with mean = mean/16 and std dev = season std dev / sqrt(16)
        mean_season = round(np.mean([players[player].get_total_score_for_year(year) for (player, year) in top_n_scores]), 3)
        mean_game = round(mean_season / 16.0, 3)
        stddev_season = round(np.sum([players[player].get_stddev_for_year(year) ** 2 for (player, year) in top_n_scores]) ** 0.5, 3)
        stddev_game = round(stddev_season / 4.0, 3) # divide by sqrt(n = 16 games) = 4
        if print_output:
            print(f"For {position}{ind+1}:")
            print(f"Mean: {mean_season} ({mean_game} points per game)")
            print(f"Std. Dev: {stddev_season} ({stddev_game} points per game)")
        x.append(ind + 1)
        y.append(mean_season)
        e_season.append(stddev_season)
        e_game.append(stddev_game)
    
    if plot_output:
        plt.errorbar(np.array(x), np.array(y), np.array(e_season), linestyle='None', marker='o', capsize=2, color="blue", ecolor="red")
        plt.title(f"Average Points Scored by the Top {topn} {position}s Per Season ")
        plt.xlabel(f"{position} position at finish")
        xticks_arr = [str(num+1) for num in range(topn)]
        xticks_arr.insert(0, "")
        plt.xticks(np.arange(topn+1), xticks_arr)
        plt.ylabel("Points / Season")
        plt.show()

        plt.errorbar(np.array(x), np.array(y)/16, np.array(e_game), linestyle='None', marker='o', capsize=2, color="blue", ecolor="red")
        plt.title(f"Average Points Scored by the Top {topn} {position}s Per Game")
        plt.xlabel(f"{position} position at finish")
        xticks_arr = [str(num+1) for num in range(topn)]
        xticks_arr.insert(0, "")
        plt.xticks(np.arange(topn+1), xticks_arr)
        plt.ylabel("Points / Game")
        plt.show()
    
    return np.array(y), np.array(e_season)

def closestNRanks(player, players, year_to_predict, nRanks = 5, topn=50):
    if not players[player].playedInYear(year_to_predict - 1):
        return -1
    mean_points, standard_devations = get_average_points_per_position(players[player].player_position, players, topn, plot_output=False, print_output=False)
    meanPlayer = players[player].get_total_score_for_year(year_to_predict - 1)
    stdPlayer = players[player].get_stddev_for_year(year_to_predict - 1)
    print(f"{player}'s season distribution: ")
    print("Total Season Score: " + str(meanPlayer))
    print("Standard Deviation: " + str(stdPlayer))
    print("Closest Ranks: ")
    dists = nsmallest(nRanks, [(abs(mean_points[i] - meanPlayer), i) for i in range(len(mean_points))])
    for rank in dists:
        print(players[player].player_position + str(rank[1] + 1))


def load_data(filepath, start_year = 2000, end_year = 2019):  
    global_start_year = start_year
    global_end_year = end_year
    # check if data has already been pulled
    if Path(filepath).exists():
        print("Data already saved! Loading now...")
        players = load(open(filepath, "rb"))
    else:
        print("Data not saved! Parsing now...")
        players = dict()
        # parse data
        for year in range(global_start_year, global_end_year):
            for week in range(1, 18): 
                json_resp = requests.get(f"https://www.fantasyfootballdatapros.com/api/players/{year}/{week}").json()
                # currently only extracting fantasy scores - TODO for future: store targets, etc
                for player in json_resp:
                    if player['player_name'] not in players:
                        players[player['player_name']] = Player(player['player_name'], player['position'])
                    players[player['player_name']].add_score(year, week, player['fantasy_points']['half_ppr'])
            print(f"Finished with year {year}!")
        dump(players, open(filepath, "wb"))
    print("Data loaded!")
    return players

if __name__ == "__main__":
    players = load_data("data_save/players_data.p")
    #mean_points, standard_devations = get_average_points_per_position("WR", players, 30, plot_output=False, print_output=True)
    for player in players:
        closestNRanks(player, players, 2019, nRanks=1)