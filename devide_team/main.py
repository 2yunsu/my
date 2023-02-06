#!/bin/python3

from random import choice

# create a list of players from a file
players = []
file = open('players.txt', 'r', encoding="UTF-8")
players = file.read().splitlines()

# create empty team lists
teamA = []
teamB = []
teamC = []
teamD = []

# loop until there are no players left
while len(players) > 0:

    # choose a random player for team A
    playerA = choice(players)
    teamA.append(playerA)
    # remove the player from the players list
    players.remove(playerA)

    # break out of the loop if there are no players left
    if players == []:
        break

    # choose a random player for team B
    playerB = choice(players)
    teamB.append(playerB)
    # remove the player from the players list
    players.remove(playerB)

    if players == []:
        break

    playerC = choice(players)
    teamC.append(playerC)
    # remove the player from the players list
    players.remove(playerC)

    if players == []:
        break

    playerD = choice(players)
    teamD.append(playerD)
    # remove the player from the players list
    players.remove(playerD)

# print the teams
print("홍팀:", teamA)
print("청팀:", teamB)
print("백팀:", teamC)
print("황팀:", teamD)