## AlphaZero Algorithm

AlphaZero is an algorithm developed by Google's DeepMind team to train a neural network to play any 2 play perfect information game whose game states can be mapped out
in a decision tree. The *nac.py* file is my own implementation of this algorithm for playing Naughts and Crosses.

### How does AlphaZero Play?

In order to decide which move AlphaZero thinks should be played in a given position it first assigns all legal moves a number between 0 and 1 which represents the 
probability it thinks it will win if it plays that move. It then selects the highest scoring node and expands it. \
This process is repeated until either the whole decision tree is mapped out or it has expaneded a specified number of nodes. \
Then AlphaZero scores each node that has not been expanded between -1 and 1 where -1 means it thinks it will definitely lose, and 1 it will definietly win. These 
values are then backed up the decision tree and the move with the highest score is selected to be played.

### How is AlphaZero Trained?

The first step is to generate training data for AlphaZero. This is done by getting AlphaZero to play 1000s of games against itself. \
In order to ensure AlphaZero explores a wide variety of moves, each value assigned by AlphaZero is weighted by a function that increases the less a move has been 
played from that position. The formulae used by the DeepMind team for training is U + sqrt(m)/(1+n) where U is the probability assigned by AlphaZero, m is the number 
of times the parent node has been played and n is the number of times the child node has been played. \
The highest score is selected and that is the move that is played. \
Once the game has finished for each node that was played whether or not AlphaZero went on to win/draw/lose from that position is stored in the node.

In order to train AlphaZero for each node we get the probability of winning from each move found in the training games and the average match result from that node
and train AlphaZero's outputs to match that from the training data. This process is then repeated. Each time AlphaZero is training on games from a higher skilled 
player meaning it continually improves itself. It took 18 training loops with 100,000 games per loop for AlphaGo (an early version of AlphaZero) to be better than 
any human at Go.

### Neural Network Arcitecture

AlphaZero took the current game state and the previous 63 game states as inputs. This is then passed through convolution layers until the inputs are have a shape of 
(1, 1, n) then this node on the neural network combined with the inputs is passed through some fully connected layers to the output. This neural network arcitecture is 
inspired by computer vision. The AlphaZero algorithm will work with any neural network with the appropiate input and output layers. However a large neural network is 
needed to see results for most games.

## Draughts
Draughts game made using Python. I made it to test algorithms for playing turn based games. \
**To Add:**
* Finish Draw Logic
* Visuals in pygame
* minimax algorithm

## Four Way Countdown

Four way countdown (FWC) is a four player game where the goal is to get rid of all your tiles by combining numbers from rolled dice. The game involves a lot of luck from 
the dice roll. I simulated it in Python to test out different strategies.
