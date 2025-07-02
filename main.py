from uno_package import player, game, state

def main():
    print(state.cardMatrix)
    for i in range(0,20):
        me = player.HumanPlayer('Zach')
        players = [me, player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
        game_instance = game.Game(players, True)
        game_instance.shuffle_players()
        game_instance.play()

if __name__ == "__main__":
    main()