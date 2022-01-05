Author: Siddharth Shah 

"python xword3.py 10x10 20 dct20k.txt"
The above run command will print output into the terminal detailing
its execution. The python file generates a crossword using dimensions
10x10 (height by width), number of blocks (20), and the input dictionary
of possible insert words (dct20k.txt). The solver then proceeds to create
constraints for every word and checks if input words are valid when performing
a DFS solve with CACHEING to increase runtime efficiency.