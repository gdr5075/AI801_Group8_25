from uno_package import player, game

def main():
    me = player.HumanPlayer('Zach')
    players = [me, player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
    for i in range(0,20):
        game.Game(players).play()

if __name__ == "__main__":
    main()