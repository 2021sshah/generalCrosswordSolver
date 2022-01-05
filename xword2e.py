# Siddharth Shah Gabor Pd. 2
import sys, time, random

# Universal Methods
def placeSym(mainBoard, row, col, sym):
    board = copyBoard(mainBoard)
    board[row][col] = sym
    if sym == blockCHAR: board[HEIGHT-1-row][WIDTH-1-col] = sym
    return board 

def convertBoardSegment(mainBoard, idxList, sym):
    board = copyBoard(mainBoard)
    for row, col in idxList:
        board = placeSym(board, row, col, sym)
    return board

def placeWordInBoard(mainBoard, idxList, symStr):
    board = copyBoard(mainBoard)
    for idx, tup in enumerate(idxList): # Array indicies, (row, col)
        board = placeSym(board, tup[0], tup[1], symStr[idx])
    return board

def copyBoard(board):
    return [[board[r][c] for c in range(WIDTH)] for r in range(HEIGHT)]

def copyDct(dct):
    return {key:val for key, val in dct.items()}

def convertSegmentToValues(board, struct):
    return [board[row][col] for row, col in struct]

# Input Specific
def manageInputs(sysList):
    splitIdx = sysList[0].lower().index("x")
    height, width = int(sysList[0][:splitIdx]), int(sysList[0][splitIdx+1:])
    numBlocks = int(sysList[1])
    dictName = sysList[2]
    return height, width, numBlocks, dictName

def managePreconditions(seeds):
    seedStrings = set()
    digits = "1234567890"
    for seed in seeds:
        orientation = seed[0].upper()   # H or V
        splitIdx = seed.lower().index("x")
        finalIdx = len(seed) # Assume no chars at end
        for idx, char in enumerate(seed[splitIdx+1:]):
            if char in digits: continue
            finalIdx = idx + splitIdx + 1 # Found char -> update finalIdx
            break # Leave once updated
        down, right = int(seed[1:splitIdx]), int(seed[splitIdx+1:finalIdx])
        remainingChar = seed[finalIdx:].upper()
        if not remainingChar: remainingChar = blockCHAR
        seedStrings.add((orientation, down, right, remainingChar))
    return seedStrings

def insertSeedStrings(board, seedStrings):
    for orientation, row, col, charsToPlace in seedStrings:
        for idx, char in enumerate(charsToPlace):
            board = placeSym(board, row, col+idx, char) if orientation == "H" else placeSym(board, row+idx, col, char)
    return board

def boundRowsAndCols():
    rows = [[(r, c) for c in range(WIDTH)] for r in range(HEIGHT)]
    cols = [[(r, c) for r in range(HEIGHT)] for c in range(WIDTH)]
    return rows + cols

# Helper Methods for Validation
def findASpace(board):
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if board[r][c] != blockCHAR: return (r,c)
    return (0,0)

spaceSet = set()
def spaceConnections(board, idx):
    if idx in spaceSet: return # No Duplicates
    if idx[0] < 0 or idx[0] >= HEIGHT or idx[1] < 0 or idx[1] >= WIDTH: return # Out of Range
    if board[idx[0]][idx[1]] == blockCHAR: return # Cannot Access Idx
    spaceSet.add(idx)
    spaceConnections(board, (idx[0]-1, idx[1])) # Up
    spaceConnections(board, (idx[0]+1, idx[1])) # Down
    spaceConnections(board, (idx[0], idx[1]-1)) # Left
    spaceConnections(board, (idx[0], idx[1]+1)) # Right

def convertConstToValues(board, const): # Indices to Values
    return [board[row][col] for row, col in const]

# Board Validity
def symmetric180(board):
    halfway = (HEIGHT*WIDTH+1)//2
    for row in range((HEIGHT-1)//2):
        for col in range(WIDTH):
            if (row*HEIGHT+col+1) > halfway: continue # Already gone halfway
            if board[row][col] == board[HEIGHT-1-row][WIDTH-1-col]: continue
            if board[row][col] == blockCHAR or board[HEIGHT-1-row][WIDTH-1-col] == blockCHAR: return False
    return True

def spacesConnected(board):
    spaceSet.clear()
    idx = findASpace(board)
    spaceConnections(board, idx)
    if len(spaceSet) == (HEIGHT*WIDTH-sum([1 for r in range(HEIGHT) for c in range(WIDTH) if board[r][c] == blockCHAR])): 
        return True # Total Connected Spaces == Total Open Spaces
    return False

def minWordLengthSatisfied(board):
    for const in rowsAndCols: # Go Through Every Constraint Set
        values = [blockCHAR] + convertConstToValues(board, const) + [blockCHAR]
        prevIdx = 0
        for valIdx in range(1, len(values)):
            if values[valIdx] != blockCHAR: continue
            if valIdx - prevIdx > 1 and valIdx - prevIdx < 4: return False # One Or Two Letters in Between Spaces
            prevIdx = valIdx
    return True

def isValid(board):
    return symmetric180(board) and spacesConnected(board) and minWordLengthSatisfied(board)

# Helper Methods for Solve
def removeTooShort(board):
    for const in rowsAndCols: # Same Structure as minWordLength
        values = [blockCHAR] + convertConstToValues(board, const) + [blockCHAR]
        prevIdx = 0
        for valIdx in range(1, len(values)):
            if values[valIdx] != blockCHAR: continue
            if valIdx - prevIdx > 1 and valIdx - prevIdx < 4: 
                board = convertBoardSegment(board, const[prevIdx:valIdx-1], blockCHAR)
            prevIdx = valIdx
    return board    

def makeBoardValid(board):
    board = removeTooShort(board)
    repeat = not minWordLengthSatisfied(board)
    while(repeat): # Necessary
        board = removeTooShort(board)
        if minWordLengthSatisfied(board): repeat = False
    return board

# Blocks Routine
def placeBlocks(board, numToPlace):
    if not spacesConnected(board): return "" # Invalid
    if not numToPlace: return board # Solved
    idxInfoLst = possibleChoices(board) # Ordered By Disruptive (Check Return)
    for idxWeight, flippedList in idxInfoLst: # Idx for testing
        if len(flippedList) > numToPlace: continue
        newBoard = convertBoardSegment(board, flippedList, blockCHAR)
        psblSol = placeBlocks(newBoard, numToPlace-len(flippedList))
        if psblSol: return psblSol
    return ""

def possibleChoices(mainBoard): # Brunt Work of Ordering Moves
    halfway = HEIGHT*WIDTH//2
    idxToChangedSet, idxToMoveWeight = {}, {}
    for row in range((HEIGHT-1)//2):
        for col in range(WIDTH):
            if row*WIDTH+col+1 > halfway: continue # Already Done
            if mainBoard[row][col] != openCHAR or mainBoard[HEIGHT-1-row][WIDTH-1-col] != openCHAR: continue
            board = copyBoard(mainBoard)
            tempBoard = makeBoardValid(placeSym(board, row, col, blockCHAR))
            valid, changedSet = compileChangedAndValid(mainBoard, tempBoard)
            if not valid: continue
            idxToChangedSet[(row, col)] = changedSet
            idxToMoveWeight[(row, col)] = moveWeight(tempBoard, row, col, changedSet)
    # First Is Heuristic
    toRet = sorted([(idxToMoveWeight[idx], idxToChangedSet[idx]) for idx in idxToChangedSet if len(idxToChangedSet[idx])])
    return toRet

def moveWeight(board, row, col, flippedList): # Smaller is Better
    total = 0 # Distance To Block In 4 Directions
    for c in range(col-1, -1, -1): # Left
        if board[row][c] == blockCHAR: break
        total += 1
    for c in range(col+1, WIDTH): # Right
        if board[row][c] == blockCHAR: break
        total += 1
    for r in range(row-1, -1, -1): # Up
        if board[r][col] == blockCHAR: break
        total += 1
    for r in range(row+1, HEIGHT): # Down
        if board[r][col] == blockCHAR: break
        total += 1
    centerDist = (HEIGHT-row)*row + abs(WIDTH-col)*col
    return len(flippedList)*100 - total - centerDist/((WIDTH+HEIGHT)/6.5)

def compileChangedAndValid(board, newBoard):
    changedSet = set()
    for row in range(HEIGHT): # Isn't More Efficient To Do Half
        for col in range(WIDTH):
            if board[row][col] != blockCHAR and newBoard[row][col] == blockCHAR: # Block Was Placed
                if board[row][col] != openCHAR: 
                    return False, {1} # Invalid
                changedSet.add((row, col))
    return True, changedSet

# Reading the Word Dictionary
def readPossibleWords(fileName):
    wordList = open(fileName, "r").read().splitlines()
    psblWords, lenToWords = set(), {}
    letToCount = {}
    for word in wordList:
        word = word.upper()
        psblWords.add(word)
        if not len(word) in lenToWords: lenToWords[len(word)] = set()
        lenToWords[len(word)].add(word)
        for let in word:
            if not let in letToCount: letToCount[let] = 0
            letToCount[let] += 1
    wordToScore = {}
    for word in psblWords:
        wordToScore[word] = sum([letToCount[ch] for ch in word])//len(word)
    return psblWords, lenToWords, wordToScore

def determineWordsToAdd(board):
    jointIndicies = set()
    for const in rowsAndCols: # Vertical and Horizonal
        startIdx = 0
        for upToIdx, posTup in enumerate(const): # array indicies, (row, col)
            if board[posTup[0]][posTup[1]] != blockCHAR: continue
            jointIndicies.add(tuple(const[startIdx:upToIdx]))
            startIdx = upToIdx + 1
        if startIdx != len(const): jointIndicies.add(tuple(const[startIdx:]))
    return {wordIndicies for wordIndicies in jointIndicies if len(wordIndicies)} # Removes Empty

# Placing Words in Crosswords
CACHE = {} # Global
goBack, stallCounter = 0, 0
stuckWord, stuckPsbl = (), set()
bestLevel = 0
def placeAllWords(board, wordsToAdd, usedWords, level): # WordsToAdd must be local
    global goBack, stallCounter, stuckWord, stuckPsbl, bestLevel
    global totalWords
    if not len(wordsToAdd):
        print("here")
        if repeatingWords(board): return "" # Invalid
        return board
    if goBack: # Print for testing
        print("going")
        showBoardwIndent(board,2)
        if goBack == 1: bestLevel == level
        print()
    if goBack: 
        goBack -= 1
        return "" # Getting Out
    if len(wordsToAdd) < 5:
        print("stall")
        stallCounter += 1
        if stallCounter > 25: # Time to Leave
            stallCounter == 0
            goBack = totalWords*30
            stuckWord = wordsToAdd.copy()
            stuckWord = stuckWord.pop()
            stuckPsbl = possibleWords(board, stuckWord)
            return "" # Go Back
    if level > bestLevel: # Print Intermediary Steps
        showBoard(board)
        print()
        bestLevel = level
    leastWord, leastPsblWords = (), set()  # Begin From Here
    for wordIndicies in wordsToAdd: # Choose Most Constrained Word
        wordVals = "".join(convertSegmentToValues(board, wordIndicies))
        if wordVals in CACHE: pWords = CACHE[wordVals]
        else:
            pWords = possibleWords(board, wordIndicies)
            CACHE[wordVals] = pWords
        pWords -= usedWords
        if not len(pWords): return "" # Invalid
        if not leastWord or len(pWords) < len(leastPsblWords): 
            leastWord = wordIndicies 
            leastPsblWords = pWords
    bestToWorst = determineBestWords(leastPsblWords) # Choose Best Words
    for score, word in bestToWorst:
        newBoard = placeWordInBoard(board, leastWord, word)
        invalid, filledIndicies, filledWords = extraFilled(newBoard, wordsToAdd)
        if invalid: return "" # Newly Filled Not Word
        psblSol = placeAllWords(newBoard, wordsToAdd-{leastWord}-filledIndicies, usedWords|filledWords, level+1) # usedWords|{word}|filledWords
        if psblSol: return psblSol
    return "" 

def repeatingWords(board):
    words = ["".join(convertSegmentToValues(board, word)) for word in wordsToAdd] # Global wordsToAdd
    return len(words) != len(set(words))

def possibleWords(board, idxList):
    psblWords = lenToWords[len(idxList)]
    for pos, idxTup in enumerate(idxList): # Array indicies, (row, col)
        if board[idxTup[0]][idxTup[1]] == openCHAR: continue
        psblWords = psblWords & findWordSet(len(idxList), pos, board[idxTup[0]][idxTup[1]])
    return psblWords

def findWordSet(wordLen, letIdx, symAtIdx):
    fittingWords = set()
    for word in lenToWords[wordLen]:
        if word[letIdx] == symAtIdx: fittingWords.add(word)
    return fittingWords

def determineBestWords(wordList):
    return sorted({(wordToScore[word], word) for word in wordList})[::-1] # Highest to Lowest Score

def extraFilled(board, wordsToAdd):
    removeIdx, removeWord = set(), set()
    for wordIndicies in wordsToAdd:
        valueLst = convertSegmentToValues(board, wordIndicies)
        if openCHAR in valueLst: continue
        prosWord = "".join(valueLst)
        if not prosWord in psblWords or prosWord in removeWord: return (True, set(), set())
        removeIdx.add(wordIndicies)
        removeWord.add(prosWord)
    return (False, removeIdx, removeWord)

# Printing and Testing
def showBoard(board):
    for row in range(len(board)):
        print(" ".join(board[row]))

def showBoardwIndent(board, indent):
    for row in range(len(board)):
        print("\t"*indent + " ".join(board[row]))

def placeWord(board, wordsToAdd):
    wordsToAdd.pop()
    return board

startTime = time.time()
# Constraints
blockCHAR = "#"
openCHAR = "-"
# Inputs
HEIGHT, WIDTH, numBlocks, fileName = manageInputs(sys.argv[1:4])
seedStrings = managePreconditions(sys.argv[4:])
# Low-hanging Fruit
if numBlocks == HEIGHT*WIDTH:
    showBoard([[blockCHAR for c in range(WIDTH)] for r in range(HEIGHT)])
    exit(0)
# Initializations
board = [[openCHAR for c in range(WIDTH)] for r in range(HEIGHT)] # Matrix [row][col]
if numBlocks%2: board[HEIGHT//2][WIDTH//2] = blockCHAR # Base Case
board = insertSeedStrings(board, seedStrings)
rowsAndCols = boundRowsAndCols()
# Place Blocks
board = makeBoardValid(board)
blockCount = len([1 for r in range(HEIGHT) for c in range(WIDTH) if board[r][c] == blockCHAR])
showBoard(board) # Prints Original
print()
board = placeBlocks(board, numBlocks-blockCount)
if len(board): 
    showBoard(board)
    print()
# Reading Dictionary + Determining Words
psblWords, lenToWords, wordToScore = readPossibleWords(fileName)
wordsToAdd = determineWordsToAdd(board)
totalWords = len(wordsToAdd)
# Place Words in Board
board = placeAllWords(board, wordsToAdd.copy(), set(), 0)
showBoard(board)
print("Total Time: {}s".format(round(time.time()-startTime, 3)))