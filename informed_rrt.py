"""
Path Planning Sample Code with RRT*

author: AtsushiSakai(@Atsushi_twi)

"""

import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from data_loader import load_test_dataset
import pickle 
import pygame
show_animation = True

obc,obstacles, paths, path_lengths= load_test_dataset() 

'''XDIM = 40
YDIM = 40
windowSize = [XDIM, YDIM]
pygame.init()
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode(windowSize)
white = 255, 255, 255
black = 0, 0, 0
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255
cyan = 0,180,105
dark_green = 0, 102, 0'''


class RRT():
	"""
	Class for RRT Planning
	"""

	def __init__(self, start, goal, obstacleList, randArea,
		         expandDis=0.6, goalSampleRate=0, maxIter=2000):
		"""
		Setting Parameter

		start:Start Position [x,y]
		goal:Goal Position [x,y]
		obstacleList:obstacle Positions [[x,y,size],...]
		randArea:Ramdom Samping Area [min,max]

		"""
		self.start = Node(start[0], start[1])
		self.end = Node(goal[0], goal[1])
		self.minrand = randArea[0]
		self.maxrand = randArea[1]
		self.expandDis = expandDis
		self.goalSampleRate = goalSampleRate
		self.maxIter = maxIter
		self.obstacleList = obstacleList

	def Planning(self, animation=False):
		"""
		Pathplanning

		animation: flag for animation on or off
		"""

		self.nodeList = [self.start]
		for i in range(self.maxIter):
			rnd = self.get_centroid_point()
			#rnd=self.get_random_point()
			nind = self.GetNearestListIndex(self.nodeList, rnd)

			newNode = self.steer(rnd, nind)
			#  print(newNode.cost)

			if self.__CollisionCheck(newNode, self.obstacleList):
				nearinds = self.find_near_nodes(newNode)
				newNode = self.choose_parent(newNode, nearinds)
				self.nodeList.append(newNode)
				self.rewire(newNode, nearinds)

			if animation:
				self.DrawGraph(rnd)
		# generate coruse
		#print "I am here" 
			lastIndex = self.get_best_last_index()
			if lastIndex !=None:
				print "iterations: "+str(i)
				break
		if lastIndex !=None:
			path = self.gen_final_course(lastIndex)
			return path, self.nodeList[lastIndex].cost
		else:
			return 0,0

	def choose_parent(self, newNode, nearinds):
		if len(nearinds) == 0:
			return newNode

		dlist = []
		for i in nearinds:
			dx = newNode.x - self.nodeList[i].x
			dy= newNode.y - self.nodeList[i].y	
			theta = math.atan2(dy,dx)
			d = math.sqrt(dx ** 2 + dy** 2)
			if self.check_collision_extend(self.nodeList[i], theta, d):
				dlist.append(self.nodeList[i].cost + d)
			else:
				dlist.append(float("inf"))

			mincost = min(dlist)
			minind = nearinds[dlist.index(mincost)]

		if mincost == float("inf"):
			print("mincost is inf")
			return newNode

		newNode.cost = mincost
		newNode.parent = minind

		return newNode

	def steer(self, rnd, nind):

		# expand tree
		nearestNode = self.nodeList[nind]
		theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
		newNode = copy.deepcopy(nearestNode)
		newNode.x += self.expandDis * math.cos(theta)
		newNode.y += self.expandDis * math.sin(theta)

		newNode.cost += self.expandDis
		newNode.parent = nind
		return newNode

	def get_random_point(self):

		if random.randint(0, 100) > self.goalSampleRate:
			rnd = [random.uniform(self.minrand, self.maxrand),
			random.uniform(self.minrand, self.maxrand)]
		else:  # goal point sampling
			rnd = [self.end.x, self.end.y]

		return rnd

	def get_centroid_point(self):

		if random.randint(0, 100) > self.goalSampleRate:
			rnd = [random.uniform(self.minrand, self.maxrand),random.uniform(self.minrand, self.maxrand)]

			dsg= math.sqrt((self.start.x-self.end.x)** 2 + (self.start.y-self.end.y)** 2)
			drg= math.sqrt((rnd[0]-self.end.x)** 2 + (rnd[1]-self.end.y)** 2)
			drs= math.sqrt((rnd[0]-self.start.x)** 2 + (rnd[1]-self.start.y)** 2)
			rnd[0]=(rnd[0]*dsg+self.start.x*drg+self.end.x*drs)/(dsg+drg+drs)
			rnd[1]=(rnd[1]*dsg+self.start.y*drg+self.end.y*drs)/(dsg+drg+drs)
		else:  # goal point sampling
			rnd = [self.end.x, self.end.y]

		return rnd



	def get_best_last_index(self):
		#print "I am here 1"
		disglist = [self.calc_dist_to_goal(node.x, node.y) for node in self.nodeList]
		#print "I am here 2"
		goalinds = [disglist.index(i) for i in disglist if i <= self.expandDis]
		#print(goalinds)
		if not goalinds:
			#print "Empty"
			return None
		#print "I am here 3"
		mincost = min([self.nodeList[i].cost for i in goalinds])
		#print "I am here 4"
		for i in goalinds:
			if self.nodeList[i].cost == mincost:
				return i

		return None

	def gen_final_course(self, goalind):
		path = [[self.end.x, self.end.y]]
		while self.nodeList[goalind].parent is not None:
			node = self.nodeList[goalind]
			path.append([node.x, node.y])
			goalind = node.parent
		path.append([self.start.x, self.start.y])
		return path

	def calc_dist_to_goal(self, x, y):
		return np.linalg.norm([x - self.end.x, y - self.end.y])

	def find_near_nodes(self, newNode):
		nnode = len(self.nodeList)
		r = 40.0 * math.sqrt((math.log(nnode) / nnode))
		#  r = self.expandDis * 5.0
		dlist = [(node.x - newNode.x) ** 2 +(node.y - newNode.y) ** 2 for node in self.nodeList]
		nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
		return nearinds

	def rewire(self, newNode, nearinds):
		nnode = len(self.nodeList)
		for i in nearinds:
			nearNode = self.nodeList[i]

			dx = newNode.x - nearNode.x
			dy = newNode.y - nearNode.y
			d = math.sqrt(dx ** 2 + dy ** 2)

			scost = newNode.cost + d

			if nearNode.cost > scost:
				theta = math.atan2(dy, dx)
				if self.check_collision_extend(nearNode, theta, d):
					nearNode.parent = nnode - 1
					nearNode.cost = scost

	def check_collision_extend(self, nearNode, theta, d):

		tmpNode = copy.deepcopy(nearNode)

		for i in range(int(d / self.expandDis)):
			tmpNode.x += self.expandDis * math.cos(theta)
			tmpNode.y += self.expandDis * math.sin(theta)
			if not self.__CollisionCheck(tmpNode, self.obstacleList):
				return False

		return True

	def DrawGraph(self, rnd=None):
		u"""
		Draw Graph
		"""
		plt.clf()
		if rnd is not None:
			plt.plot(rnd[0], rnd[1], "^k")
		for node in self.nodeList:
			if node.parent is not None:
				plt.plot([node.x, self.nodeList[node.parent].x], [node.y, self.nodeList[node.parent].y], "-g")

		for (ox, oy, sizex,sizey) in self.obstacleList:
			plt.plot(ox, oy, "sk", ms=7.2*sizex)


		'''fig3 = plt.figure()
		ax3 = fig3.add_subplot(111, aspect='equal')
		for (ox, oy, size) in self.obstacleList:
		p=patches.Rectangle((ox,oy),size,size)
		ax3.add_patch(p)
		#input()'''


		plt.plot(self.start.x, self.start.y, "xr")
		plt.plot(self.end.x, self.end.y, "xr")
		plt.axis([-20, 20, -20, 20])
		plt.grid(True)
		plt.pause(0.01)

	def GetNearestListIndex(self, nodeList, rnd):
		dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])** 2 for node in nodeList]
		minind = dlist.index(min(dlist))

		return minind


	def __CollisionCheck(self, node, obstacleList):

		for (ox, oy, sizex,sizey) in obstacleList:
			dx = abs(ox - node.x)
			dy = abs(oy - node.y)
			CF = False
			if dx > sizex/2.0 or dy > sizey/2.0:            
				CF= True #safe

			if CF==False:            
				return False  # collision

		return True  # safe'''

class Node():
	"""
	RRT Node
	"""

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.cost = 0.0
		self.parent = None

'''def init_obstacles(obstacleList):  #initialized the obstacle
	rectObs = []
	for (ox,oy,sizex,sizey) in obstacleList:
		obs=np.zeros(2,dtype=np.float32)
		X=np.zeros(2,dtype=np.float32)
		Y=np.zeros(2,dtype=np.float32)
		X[0]=1.0*sizex/2.0
		X[1]=0.0
		Y[0]=0.0
		Y[1]=1.0*sizey/2.0
		obs[0]=ox-X[0]+Y[0]
		obs[1]=oy-X[1]+Y[1]
		rectObs.append(pygame.Rect((obs[0],obs[1]),(sizex,sizey)))
			
	for rect in rectObs:
		pygame.draw.rect(screen, black, rect)
	fpsClock.tick(10)'''

def main():
	print("Start rrt planning")
	min_t=1000.0
	max_t=0.0
	m_time=0.0
	count=0.0
	e=[]
	ce=[]
	for i in range(0,10):
		print "i="+str(i)
		tenv=[]
		cenv=[]
		for j in range(0,20):
			print "i= "+str(i)+" j= "+str(j)
			if path_lengths[i][j]>0:
				# ====Search Path with RRT====
				obstacleList = [
				(obc[i][0][0], obc[i][0][1], 5.0, 5.0),
				(obc[i][1][0], obc[i][1][1], 5.0, 5.0),
				(obc[i][2][0], obc[i][2][1], 5.0, 5.0),
				(obc[i][3][0], obc[i][3][1], 5.0, 5.0),
				(obc[i][4][0], obc[i][4][1], 5.0, 5.0),
				(obc[i][5][0], obc[i][5][1], 5.0, 5.0),
				(obc[i][6][0], obc[i][6][1], 5.0, 5.0)
				]  # [x,y,size(radius)]

				# Set Initial parameters
				#init_obstacles(obstacleList)
				tt=0.0
				for s in range(0,1):
					rrt = RRT(start=[paths[i][j][0][0], paths[i][j][0][1]], goal=[paths[i][j][path_lengths[i][j]-1][0], paths[i][j][path_lengths[i][j]-1][1]],
					  randArea=[-20, 20], obstacleList=obstacleList)
					tic = time.clock()
					path, cost = rrt.Planning(animation=False)
					toc = time.clock()
					t=toc-tic
					tt=tt+t
				m_time=(tt/float(1))
				tenv.append(m_time)
				cenv.append(cost)
				if m_time>max_t:
					max_t=m_time
				if m_time<min_t:
					min_t=m_time
				count=count+1
		e.append(tenv)
		ce.append(cenv)
	pickle.dump(e, open("time_s2D_unseen_irrt.p", "wb" ))
	pickle.dump(ce, open("cost_s2D_unseen_irrt.p", "wb" ))
				
	print "mean time: "+ str(m_time/float(count))
	print "max time: "+ str(max_t)
	print "min time: "+ str(min_t)
	#print path
	print "cost" + str(cost)
	# Draw final path
	if show_animation:
		rrt.DrawGraph()
		plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
		plt.grid(True)
		plt.pause(0.01)  # Need for Mac
		plt.show()


if __name__ == '__main__':
	main()
