

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
