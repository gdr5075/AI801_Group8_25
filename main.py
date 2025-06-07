from uno_package import player, game

def main():
    players = [player.Player('Gandalf'), player.Player('Frodo'), player.Player('Sauron'), player.Player('Gollum')]
    game.Game(players)
        
if __name__ == "__main__":
    main()