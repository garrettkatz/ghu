import numpy as np
def make_dsst_grid(letters, rows, cols):

    grid = [["+" for c in range(cols+2)] for r in range(2*rows+2)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c < len(letters):
                grid[2*r+1][c+1] = letters[c]
                grid[2*r+2][c+1] = str(c)
            else:
                grid[2*r+1][c+1] = letters[np.random.randint(len(letters))]
                grid[2*r+2][c+1] = "_"
    return grid

if __name__=="__main__":

    letters = list('abcd')
    # explicitly convert to list for python3
    digits = list(map(str,range(len(letters))))
    grid = make_dsst_grid(letters, 3, len(letters))

    print("\n".join([" ".join(g) for g in grid]))