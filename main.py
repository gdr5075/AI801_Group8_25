from uno_package import card, deck

def main():
    mainDeck = deck.UnoMainDeck()
    discardDeck = deck.Deck()
    mainDeck.print_all()
        
if __name__ == "__main__":
    main()