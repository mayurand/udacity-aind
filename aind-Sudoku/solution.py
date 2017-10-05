assignments = []

rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

boxes = cross(rows,cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

# To include the diagonal sudoku problem, simply add a new constraint as a unit to unitlist
diagonal_units1 = [rows[i]+cols[i] for i in range(len(rows))]
diagonal_units2 = [rows[i]+cols[8-i] for i in range(len(rows))]
diagonal_units = [diagonal_units1,diagonal_units2]

unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """
    
    lenAssign = len(assignments)
    values[box] = value
    
    if len(value) == 1:
        if lenAssign>0:
            temp = assignments[lenAssign-1]
            if temp[box] != value:
                temp[box] = value
                assignments.append(temp.copy())
        else:
            assignments.append(values.copy())
        
    return values

def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    assert len(grid) == 81, "Input grid must be a string of length 81 (9x9)"
    return dict(zip(boxes, grid))


def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    Returns: None
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    ## Find all instances of naked twins
    for unit in unitlist:
        # Take a dictionary of a unit 
        unit_dict = dict()
        
        for box in unit:
            box_val = values[box]
            if len(box_val)==2: # For pair of values
                if box_val not in unit_dict.keys():
                    unit_dict[box_val]=[]
                unit_dict[box_val].append(box)

        if bool(unit_dict): # Check if dictinary empty
            for key, value in unit_dict.items(): 
                # If actually a twin pair then remove these values in twins from that unit except at these two locations
                # Eliminate the naked twins as possibilities from that unit
                if len(value)>1: 
                    for box_ in unit:
                        if box_ not in value:
                            values[box_] = values[box_].replace(key[0],"") 
                            values[box_] = values[box_].replace(key[1],"") 
                        
   
    return values


def eliminate(values):
    """Eliminate the numbers that are not possible in the solution
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    
    for box in values.keys():
        digit = values[box]
        if len(digit)==1:
            for peer in peers[box]:
                # values[peer] = values[peer].replace(digit,'') 
                values = assign_value(values, peer, values[peer].replace(digit,'') )
                
    return values

def only_choice(values):
    """Select the only choice element
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    
    """
    num_places = []
    for unit in unitlist:
        for digit in '123456789':
            for box in unit:
                if digit in values[box]:
                    num_places.append(box)
            if len(num_places) ==1:
                values = assign_value(values, num_places[0], digit)
                # values[num_places[0]]=digit
            num_places = []    
            
    return values


def reduce_puzzle(values):
    """Reduce the puzzle using constraint propagation: elimination, only choice and naked twins
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        
        # Apply one by one each constraint
        values = eliminate(values)
        values = only_choice(values)
        values = naked_twins(values)
        
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    """Using depth-first search and propagation, create a search tree and solve the sudoku.
    
        Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    
    ## Try reduce puzzle to find a solution using constraint propagation and if doesn't work then try search
    values = reduce_puzzle(values)
    if values is False:
        return False            
    if all(len(values[s]) == 1 for s in boxes):
        return values

    # Choose one of the unfilled squares with the fewest possibilities
    box,box_selected = min((len(values[box_selected]), box_selected) for box_selected in boxes if len(values[box_selected]) > 1)

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[box_selected]:
        new_sudoku = values.copy()
        new_sudoku[box_selected] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt
        

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    ## Generate the sudoku grid
    values = grid_values(grid)
     
    ## Record the complete dictionary first
    assign_value(values, boxes[0], values[boxes[0]])
     
    ## Generate the dictionary for values possible for every box  
    for box, value in values.items():
        if value == '.':
            values = assign_value(values, box, cols)
 
    values = search(values)


    return values
    

if __name__ == '__main__':
    # diag_sudoku_grid = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))
 
    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)
 
    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
