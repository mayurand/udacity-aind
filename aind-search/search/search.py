# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

from util import *

class SearchProblem:
	"""
	This class outlines the structure of a search problem, but doesn't implement
	any of the methods (in object-oriented terminology: an abstract class).
	
	You do not need to change anything in this class, ever.
	"""

	def getStartState(self):
		"""
		Returns the start state for the search problem 
		"""
		util.raiseNotDefined()
		
	def isGoalState(self, state):
		"""
			 state: Search state
		
		 Returns True if and only if the state is a valid goal state
		"""
		util.raiseNotDefined()

	def getSuccessors(self, state):
		"""
			 state: Search state
		 
		 For a given state, this should return a list of triples, 
		 (successor, action, stepCost), where 'successor' is a 
		 successor to the current state, 'action' is the action
		 required to get there, and 'stepCost' is the incremental 
		 cost of expanding to that successor
		"""
		util.raiseNotDefined()

	def getCostOfActions(self, actions):
		"""
			actions: A list of actions to take
 
		 This method returns the total cost of a particular sequence of actions.	The sequence must
		 be composed of legal moves
		"""
		util.raiseNotDefined()
					 

def tinyMazeSearch(problem):
	"""
	Returns a sequence of moves that solves tinyMaze.	For any other
	maze, the sequence of moves will be incorrect, so only use this for tinyMaze
	"""
	from game import Directions
	s = Directions.SOUTH
	w = Directions.WEST
	return	[s, s, w, s, w, w, s, w]

class NodeSearch:
	def __init__(self, value=None, parent=None, action=None, heuristic=0, pathCost=0):
		self.value = value
		self.parent = parent
		self.heuristic = heuristic
		self.pathCost = pathCost
		self.action = action
		
	def getValue(self):
		return self.value

	def getActionVal(self):
		return self.action
	
	def getCost(self):
		return self.pathCost + self.heuristic
	
	def getParent(self):
		return self.parent
	
	
def depthFirstSearch(problem):
	"""
	Search the deepest nodes in the search tree first
	[2nd Edition: p 75, 3rd Edition: p 87]
	
	Your search algorithm needs to return a list of actions that reaches
	the goal.	Make sure to implement a graph search algorithm 
	[2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
	
	To get started, you might want to try some of these simple commands to
	understand the search problem that is being passed in:
	
	print "Start:", problem.getStartState()
	print "Is the start a goal?", problem.isGoalState(problem.getStartState())
	print "Start's successors:", problem.getSuccessors(problem.getStartState())
	"""
	startNode = NodeSearch(problem.getStartState())
	fringe = Stack()  # # FIFO for dfs
	fringe.push(startNode)
	exploredSet = []

	## Loop till there is nothing in the fringe
	while (not fringe.isEmpty()):
		popNode = fringe.pop()
		
		if (problem.isGoalState(popNode.getValue())):
			actionList = []
			# Find the actions that led to this goal state and return
			while(popNode.getActionVal() != None):
				
				actionList.append(popNode.getActionVal())
				popNode = popNode.getParent()
			return actionList[::-1]
		
		# If not goal state then expand further
		for nextState, nextAction, cost in problem.getSuccessors(popNode.getValue()):
			
			if nextState not in exploredSet:	 
				newNode = NodeSearch(nextState, popNode, nextAction, nullHeuristic(nextState), cost+1)
				fringe.push(newNode)
				exploredSet.append(nextState)

def breadthFirstSearch(problem):
	"""
	Search the shallowest nodes in the search tree first.
	[2nd Edition: p 73, 3rd Edition: p 82]
	"""
	startNode = NodeSearch(problem.getStartState())
	fringe = Queue()  # # FIFO for dfs
	fringe.push(startNode)
	exploredSet = []

	## Loop till there is nothing in the fringe
	while (not fringe.isEmpty()):
		popNode = fringe.pop()
		
		if (problem.isGoalState(popNode.getValue())):
			actionList = []
			# Find the actions that led to this goal state and return
			while(popNode.getActionVal() != None):
				
				actionList.append(popNode.getActionVal())
				popNode = popNode.getParent()
				
			print actionList[::-1]	
			return actionList[::-1]
		
		# If not goal state then expand further
		for nextState, nextAction, cost in problem.getSuccessors(popNode.getValue()):
			if nextState not in exploredSet:
				newNode = NodeSearch(nextState, popNode, nextAction, nullHeuristic(nextState), cost+1)
				fringe.push(newNode)
				exploredSet.append(nextState)
			
def uniformCostSearch(problem):
	"Search the node of least total cost first. "
	startNode = NodeSearch(problem.getStartState())
	fringe = PriorityQueue()  # # FIFO for dfs
	fringe.push(startNode,0)
	exploredSet = []

	## Loop till there is nothing in the fringe
	while (not fringe.isEmpty()):
		popNode = fringe.pop()
		
		if (problem.isGoalState(popNode.getValue())):
			actionList = []
			# Find the actions that led to this goal state and return
			while(popNode.getActionVal() != None):
				
				actionList.append(popNode.getActionVal())
				popNode = popNode.getParent()
			return actionList[::-1]
		
		# If not goal state then expand further
		for nextState, nextAction, nodeCost in problem.getSuccessors(popNode.getValue()):
						
			if nextState not in exploredSet:
				if nextState in fringe.getList(): ## This one is wrong as get list doesnt work 

				####### Problem updating the parent node priority and parent for a given node 

					new_g = popNode.G +1
					if nullHeuristic(nextState,problem) > new_g:
						newNode = NodeSearch(nextState, popNode, nextAction, nullHeuristic(nextState,problem), nodeCost+1)	
						fringe.push(newNode,popNode.getCost())
						exploredSet.append(nextState)
				else:
					newNode = NodeSearch(nextState, popNode, nextAction, nullHeuristic(nextState,problem), nodeCost+1)	
					fringe.push(newNode,popNode.getCost())
					exploredSet.append(nextState)
		

def nullHeuristic(state, problem=None):
	"""
	A heuristic function estimates the cost from the current state to the nearest
	goal in the provided SearchProblem.	This heuristic is trivial.
	"""
	return 0

def aStarSearch(problem, heuristic=nullHeuristic):
	"Search the node that has the lowest combined cost and heuristic first."
	startNode = NodeSearch(problem.getStartState())
	fringe = PriorityQueue()  # # FIFO for dfs
	fringe.push(startNode,0)
	exploredSet = []

	## Loop till there is nothing in the fringe
	while (not fringe.isEmpty()):
		popNode = fringe.pop()
		
		if (problem.isGoalState(popNode.getValue())):
			actionList = []
			# Find the actions that led to this goal state and return
			while(popNode.getActionVal() != None):
				
				actionList.append(popNode.getActionVal())
				popNode = popNode.getParent()
			return actionList[::-1]
		
		# If not goal state then expand further
		for nextState, nextAction, nodeCost in problem.getSuccessors(popNode.getValue()):
			
			if nextState not in exploredSet:
				
				
				####### Problem updating the parent node priority and parent for a given node 
				if nextState in fringe.getList(): ## This one is wrong as get list doesnt work 
					new_g = popNode.G +1
					if heuristic(nextState,problem) > new_g:
						newNode = NodeSearch(nextState, popNode, nextAction, heuristic(nextState[0],problem), nodeCost+1)	
						fringe.push(newNode,popNode.getCost())
						exploredSet.append(nextState)
				else:
					newNode = NodeSearch(nextState, popNode, nextAction, heuristic(nextState[0],problem), nodeCost+1)	
					fringe.push(newNode,popNode.getCost())
					exploredSet.append(nextState)
		
	
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
