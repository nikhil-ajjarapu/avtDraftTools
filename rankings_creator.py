from utils import *

# use elo to help user generate rankings from pairwise comparisons
# 1) Rank everyone using weighted (closestNRanks + ADP from popular websites)
# 2) Ask user to compare everyone with the same predicted rank to fix tie breakers
#    (maybe use other ranks as extra tiebreakers or show info on screen)
# 3) Show user information about last years performance (+ machine learning model to predict next year's performance?) and let them decide
# 4) Show final ranks at end and let them adjust

# TODO: use global variables that every function updates to store tier lists, use flask syntax and every pairwise comparison should redirect to a function that takes in 2 players and displays them to user. After the user chooses, redirect again until comparisons are done.

def generateTiers
    players = load_data("data_save/players_data.p", start_year=1999, end_year=2019)
    nRanks = 30
    position = "QB"
    mean_points, standard_devations = get_average_points_per_position(position, players, nRanks, plot_output=False, print_output=False)