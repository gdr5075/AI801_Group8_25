from uno_package import player, game

def main():
    for i in range(0,20):
        me = player.HumanPlayer('Zach')
        players = [me, player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
        game.Game(players).play()

if __name__ == "__main__":
    main()