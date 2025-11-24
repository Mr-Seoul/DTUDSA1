import math
import heapq
from collections import deque

def gcd(a,b):
    if b == 0:
        return a
    return gcd(b,a%b)

def lcm(a,b):
    return int((a*b)/(gcd(a,b)))

class SortingArray:
    def __init__(self,arr):
        self.arr = arr
    
    def append(self,val):
        self.arr.append(val)

    def insertSort(self):
        arr = self.arr
        sortedNum = 0

        def swap(indexA,indexB):
            temp = arr[indexA]
            arr[indexA] = arr[indexB]
            arr[indexB] = temp
        
        while sortedNum < len(arr):
            min = float('inf')
            minIndex = -1
            for index in range(sortedNum,len(arr)):
                if arr[index] < min:
                    min = arr[index]
                    minIndex = index
            swap(sortedNum,minIndex)
            sortedNum += 1
        
        return arr

    def mergeSort(self):
        n = 1
        arr = self.arr
        inversions = 0

        def mergeSortedArrays(start,A,B):
            indexA = 0
            indexB = 0
            nonlocal inversions
            while indexA + indexB < len(A) + len(B):
                    if indexA < len(A) and indexB < len(B):
                        if A[indexA] < B[indexB]:
                            self.arr[start + indexA + indexB] = A[indexA]
                            indexA += 1
                        else:
                            self.arr[start + indexA + indexB] = B[indexB]
                            indexB += 1
                            inversions += len(A) - indexA
                    elif indexA < len(A):
                        while indexA < len(A):
                            self.arr[start + indexA + indexB] = A[indexA]
                            indexA += 1
                    elif indexB < len(B):
                        while indexB < len(B):
                            self.arr[start + indexA + indexB] = B[indexB]
                            indexB += 1

        while n < len(arr):
            for seg in range(0,math.ceil(len(arr)/(2*n))):
                min = 2*n*seg
                leftarr = arr[min:min+n]
                rightarr = arr[min+n:min+2*n]
                mergeSortedArrays(min,leftarr,rightarr)
            n *= 2
        
        return [arr,inversions]

    def linSearch(self,val):
        for i in range(0,len(self.arr)):
            if self.arr[i] == val:
                return i
    
    def binarySearch(self,val):
        min = 0
        max = len(self.arr) - 1
        mid = (min+max)/2
        arr = self.arr
        while min < max:
            #Check if list is sorted
            if not (arr[min] <= arr[mid] <= arr[max]):
                print("List not sorted")
                return -1

            #Binary search algorithm
            if arr[mid] == val:
                return mid
            elif arr[mid] < val:
                min = mid + 1
            elif arr[mid] > val:
                max = mid - 1
        print("Value not in array")
        return -1
        
    def printContents(self):
        print(self.arr)
    
class Graph:
    class dirEdge:
        def __init__(self,origin,to,weight=1):
            self.to = to
            self.weight = weight
            self.origin = origin
    
    def __init__(self,n):
        self.adjarr = [[] for i in range(0,n)]
        self.size = n
        self.directed = False
    
    def insertNodes(self,n):
        self.size += n
        for i in range(0,n):
            self.adjarr.append([])
            self.unionarr.append(len(self.unionarr))
    
    #If a target is given, it will return the shortest path from v0 to target, assuming weights are constant. Otherwise, it will return the bfs path 
    def BFS(self,v0,target = -1):
        Q = deque()
        visited = [False for i in range(0,self.size)]
        visited[v0] = True

        parent = [-1 for i in range(0,self.size)]
        parent[v0] = v0

        visitpath = []

        def path(checkingNode):
            path = []
            while checkingNode != parent[checkingNode]:
                path.append(checkingNode)
                checkingNode = parent[checkingNode]
            path.append(checkingNode)
            return path[::-1]

        Q.append(v0)
        while(len(Q) > 0):
            curNode = Q.popleft()
            #Check if target
            if curNode == target:
                return path(curNode)
            for neighbour in self.adjarr[curNode]:
                if not visited[neighbour.to]:
                    Q.append(neighbour.to)
                    parent[neighbour.to] = curNode
                    visited[neighbour.to] = True
            visitpath.append(curNode)
        return (visitpath)
    
    #Returns DFS path
    def DFS(self,v0):
        S = []
        visited = [False for i in range(0,self.size)]
        visited[v0] = True
        visitpath = []
        postpath = []
        S.append((v0,False))
        while(len(S) > 0):
            curNode, visit = S.pop()
            if visit:
                postpath.append(curNode)
            else:
                S.append((curNode,True))
                visitpath.append(curNode)
                for neighbour in self.adjarr[curNode]:
                    if not visited[neighbour.to]:
                        S.append((neighbour.to,False))
                        visited[neighbour.to] = True
        return [visitpath, postpath]
        
    def Hierholzer(self):

        def degreeList():
            degreeList = [{"din":0,"dout":0} for i in range(0,self.size)]
            for vertexList in self.adjarr:
                for edge in vertexList:
                    degreeList[edge.origin]["dout"] += 1 
                    degreeList[edge.to]["din"] += 1 
            return degreeList

        def filledUsedEdges(truthValue):
            usedEdges = [[] for i in range(0,self.size)]
            for vertexList in self.adjarr:
                for edge in vertexList:
                    usedEdges[edge.origin].append(truthValue)
            return usedEdges
        
        def curEdgesUsed(arr):
            return arr == [True for i in range(0,len(arr))]
        
        def removeEdge(i,j):
            if self.directed:
                #Remove from I to J
                for index in range(0,len(self.adjarr[i])):
                    if self.adjarr[i][index].to == j and not usedEdges[i][index]:
                        usedEdges[i][index] = True
                        break
            else:
                #Remove from I to J
                for index in range(0,len(self.adjarr[i])):
                    if self.adjarr[i][index].to == j and not usedEdges[i][index]:
                        usedEdges[i][index] = True
                        break
                #Remove from J to I
                for index in range(0,len(self.adjarr[j])):
                    if self.adjarr[j][index].to == i and not usedEdges[j][index]:
                        usedEdges[j][index] = True
                        break
        
        def allNodesUsed():
            return usedEdges == filledUsedEdges(True)
        
        def checkDegreeConditions():
            v0 = 0
            if self.directed:
                vStart = -1
                vEnd = -1
                dList = degreeList()
                for index in range(0,len(dList)):
                    degrees = dList[index]
                    if degrees["din"] != degrees["dout"]:
                        if vStart != -1 and vEnd != -1:
                            print("Degree edges wrong")
                            return []
                        elif degrees["dout"] == degrees["din"] + 1 and vStart == -1:
                            vStart = index
                            v0 = vStart
                        elif degrees["din"] == degrees["dout"] + 1 and vEnd == -1:
                            vEnd = index
                        else:
                            print("Degree edges wrong")
                            return []
                if vStart != -1:
                    print("Euler Path")
                else:
                    print("Euler Cycle")
            else:
                oddNum = 0
                dList = degreeList()
                for index in range(0,len(dList)):
                    degrees = dList[index]
                    if degrees["din"] != degrees["dout"]:
                        print("Not undirected Graph")
                        return []
                    else:
                        if degrees["din"] % 2 == 1:
                            oddNum += 1
                            v0 = index
                if not (oddNum == 0 or oddNum == 2):
                    print("Degree edges wrong")
                    return []
                elif oddNum == 0:
                    print("Euler cycle")
                elif oddNum == 2:
                    print("Euler path")
            return v0

        #Check Connectivity
        if len(self.allCC()) != 1:
            print("Graph not connected")
            return []

        #Check Degrees and find starting node
        v0 = checkDegreeConditions()
        if v0 == []:
            return
                    
        usedEdges = filledUsedEdges(False)
        S = []
        curNode = v0

        S.append(curNode)
        path = []
        while not allNodesUsed():
            curNode = S[-1]
            #Follow random path
            followed = False

            if curEdgesUsed(usedEdges[curNode]):
                path.append(S.pop())

            else:
                for markedindex in range(0,len(usedEdges[curNode])):
                    if not followed:
                        if not usedEdges[curNode][markedindex]:
                            prev = curNode
                            toNode = self.adjarr[curNode][markedindex].to

                            removeEdge(prev,toNode)
                            S.append(toNode)

                            curNode = toNode
                            followed = True
        while len(S) > 0:
            path.append(S.pop())
        return path

    def printAdjList(self):
        for vertix in self.adjarr:
            printarr = []
            for edge in vertix:
                printarr.append((edge.to,edge.weight))
            print(printarr)
        
class UnDirectedGraph(Graph):
    def __init__(self,n):
        super().__init__(n)
        self.unionarr = [i for i in range(0,n)]
        self.unionrank = [0 for i in range(0,n)]
    
    def totalWeight(undirGraph):
            totWeight = 0
            for vertexlist in undirGraph.adjarr:
                for edge in vertexlist:
                    totWeight += edge.weight/2
            return totWeight
    
    def find(self,index):
        if index == self.unionarr[index]:
            return index
        else: 
            root = self.find(self.unionarr[index])
            #path compression
            self.unionarr[index] = root
            return root
    
    def union(self,indexA,indexB):
        r1 = self.find(indexA)
        r2 = self.find(indexB)

        if r1 != r2:
            if self.unionrank[r1] < self.unionrank[r2]:
                self.unionarr[r1] = r2
            elif self.unionrank[r1] > self.unionrank[r2]:
                self.unionarr[r2] = r1
            else:
                self.unionarr[r2] = r1
                self.unionrank[r1] += 1
    
    def isConnected(self,indexA,indexB):
        return self.find(indexA) == self.find(indexB)

    def printUnionContents(self):
        print(self.unionarr)
    
    def insertUndirectedEdge(self,vFrom,vTo,weight=1):
        self.adjarr[vFrom].append(Graph.dirEdge(vFrom,vTo,weight))
        self.adjarr[vTo].append(Graph.dirEdge(vTo,vFrom,weight))
        self.union(vFrom,vTo)
    
    def allCC(self):
        result = [[] for i in range(self.size)]
        for node in range(self.size):
            result[self.find(node)].append(node)
        result = [result[i] for i in range(self.size) if result[i] != []]
        
        return result

    def MST(self,AssumeConnectivity=True):
        #Prim's algorith.

        #Check if connected(Only if specified for performance reasons)
        if len(self.allCC()) != 1 and not AssumeConnectivity:
            return UnDirectedGraph(0)
    
        #Like dijkstra's, but with neighbour.weight, not cur[key] + neightbour.weight
        P = []
        MSTGraph = UnDirectedGraph(self.size)
        inMST = dict()
        for i in range(self.size):
            inMST.update({i:False})

        #Initialize Priority queue
        for node in range(0,self.size):
            neighbourlist = self.adjarr[node]
            for neighbour in neighbourlist:
                heapq.heappush(P, [neighbour.weight ,neighbour.origin, neighbour.to])
        
        #Keep on checking if the cheapest edge fits, and if it does do so.
        while len(P) > 0 and len(inMST) > 0:
            weight, origin, to = heapq.heappop(P)
            #Check if the two edges are connected
            if MSTGraph.find(origin) != MSTGraph.find(to):
                #Connect vertices in MST
                MSTGraph.insertUndirectedEdge(origin,to,weight)

                #Remove used edges
                inMST.pop(origin,False)
                inMST.pop(to,False)
        
        return (MSTGraph.totalWeight(),MSTGraph)
    
    def isBipartite(self):
        Q = deque()
        visited = [False for i in range(0,self.size)]
        color = [-1 for i in range(0,self.size)]

        Q.append(0)
        color[0] = 0
        visited[0] = True

        while len(Q) > 0:
            curNode = Q.popleft()
    
            for neighbour in self.adjarr[curNode]:
                if visited[neighbour.to] and color[neighbour.to] == color[curNode]:
                    return False
                if not visited[neighbour.to]:
                    Q.append(neighbour.to)
                    visited[neighbour.to] = True
                    color[neighbour.to] = not color[curNode]
        return True
    
class DirectedGraph(Graph):
    def __init__(self,n):
        super().__init__(n)
        self.directed = True

    def reversedGraph(self):
        newGraph = DirectedGraph(self.size)
        for vertexList in self.adjarr:
            for edge in vertexList:
                newGraph.insertDirectedEdge(edge.to,edge.origin,edge.weight)
        return newGraph
    
    def insertDirectedEdge(self,vFrom,vTo,weight=1):
        self.adjarr[vFrom].append(Graph.dirEdge(vFrom,vTo,weight))
    
    def TopologicalSort(self):
        degreeList = [0 for _ in range(0,self.size)]
        ZeroQueue = deque()
        path = []

        for vertexlist in self.adjarr:
            for edge in vertexlist:
                degreeList[edge.to] += 1
        
        def addZeroNodes():
            for vertex in range(0, self.size):
                if degreeList[vertex] == 0:
                    ZeroQueue.append(vertex)

        addZeroNodes()
        while len(ZeroQueue) > 0:
            curNode = ZeroQueue.popleft()
            path.append(curNode)
            for neighbour in self.adjarr[curNode]:
                degreeList[neighbour.to] -= 1
                if degreeList[neighbour.to] == 0:
                    ZeroQueue.append(neighbour.to)

        if len(path) == len(self.adjarr):
            return path
        else:
            return []
    
    def isDAG(self):
        return not self.TopologicalSort == []
    
    def TopologicalShortestPath(self,v0,target = -1):
        topoSort = self.TopologicalSort()
        #See if graph isn't DAG
        if topoSort == []:
            return []
        
        def path(curNode):
            path = []
            while curNode != parent[curNode]:
                path.append(curNode)
                curNode = parent[curNode]
            path.append(curNode)
            return path[::-1]

        minDist = [float('inf') for _ in range(0,len(self.adjarr))]
        minDist[topoSort[topoSort.index(v0)]] = 0
        parent = [i for i in range(0,len(self.adjarr))]

        for node in topoSort[topoSort.index(v0):len(topoSort)]:
            if node == target:
                    return (minDist[node],path(node))
            for neighbour in self.adjarr[node]:
                if minDist[neighbour.to] > minDist[node] + neighbour.weight or minDist[neighbour.to] == float('inf'):
                    parent[neighbour.to] = node
                    minDist[neighbour.to] = minDist[node] + neighbour.weight
        if target == -1:
            return minDist
        else:
            return [minDist[target],[]]
    
    def allSCC(self):
        #Kosaraju's algorithm
        nodes = [False for i in range(0,self.size)]
        allSCC = []
        DFS = []

        for nodeindex in range(0,len(nodes)):
            if not nodes[nodeindex]:
                curDFS = self.DFS(nodeindex)[1]
                for node in curDFS:
                    nodes[node] = True
                    DFS.append(node)
        
        #Reset nodes for algorithm
        nodes = [False for i in range(0,self.size)]

        #Reversed DFS
        reversedGraph = self.reversedGraph()
        for node in DFS[::-1]:
            if not nodes[node]:
                reversedDFS = reversedGraph.DFS(node)[1]

                reversedDFS = [i for i in reversedDFS if not nodes[i]]
                for dfsNode in range(0,len(reversedDFS)):
                    nodes[reversedDFS[dfsNode]] = True
                
                allSCC.append(reversedDFS)

        return allSCC
    
    #If no target is given, the entire dijkstra graph will be generated
    def Dijkstra(self, v0, target=-1):
        P = []  # priority queue
        visited = set()
        parent = [i for i in range(self.size)]
        minDist = [float('inf')] * self.size
        minDist[v0] = 0

        def path(curNode):
            node = curNode[1]
            p = []
            while node != parent[node]:
                p.append(node)
                node = parent[node]
            p.append(node)
            return (minDist[curNode[1]], p[::-1])

        # Initialize queue
        heapq.heappush(P, (0, v0))  # (distance, node)

        while P:
            val, key = heapq.heappop(P)
            # Skip outdated entries (lazy deletion)
            if val > minDist[key]:
                continue
            visited.add(key)
            # Target found
            if key == target:
                return path((val, key))
            # Explore neighbors
            for neighbour in self.adjarr[key]:
                if neighbour.to in visited:
                    continue
                newDist = val + neighbour.weight
                if newDist < minDist[neighbour.to]:
                    minDist[neighbour.to] = newDist
                    parent[neighbour.to] = key
                    heapq.heappush(P, (newDist, neighbour.to))
        if target == -1:
            return minDist
        else:
            return []

class SegmentTree():

    def __init__(self):
        self.size = 0
        self.capacity = 1
        self.valArray = []
        self.sumTree = [0]
        self.minTree = [float('inf')]
        self.maxTree = [float("-inf")]
    
    def buildTree(self,arr,type):
        n = len(arr)
        tree = [0] * (int(2 **math.ceil(math.log2(n)+1)))
        tree[0] = float('nan')
        size = 2 ** math.ceil(math.log2(n))

        identity = -1
        
        def minCond(a,b):
            return min(a,b)
        
        def maxCond(a,b):
            return max(a,b)
        
        def sumCond(a,b):
            return a+b

        condition = -1
        match type:
            case "min":
                condition = minCond
                identity = float("inf")
            case "max":
                condition = maxCond
                identity = float("-inf")
            case "sum":
                condition = sumCond
                identity = 0
            case _:
                print("Wrong condition")
                return
        
        padded_arr = arr + [identity] * (size - n)
        n = len(padded_arr)

        def build(node, l, r,cond):
            if l == r:
                tree[node] = padded_arr[l]
            else:
                mid = (l + r) // 2
                build(2 * node, l, mid,cond)
                build(2 * node + 1, mid + 1, r,cond)
                tree[node] = cond(tree[2 * node] ,tree[2 * node + 1])

        build(1, 0, n - 1,condition)
        return tree
        

    def append(self,val):
        self.valArray.append(val)
        self.size += 1

        if self.size > self.capacity:
            # Need to rebuild trees with double capacity
            self.capacity *= 2
            self.sumTree = self.buildTree(self.valArray, "sum")
            self.minTree = self.buildTree(self.valArray, "min")
            self.maxTree = self.buildTree(self.valArray, "max")
        else:
            self.update(self.size - 1, val, "sum")
            self.update(self.size - 1, val, "min")
            self.update(self.size - 1, val, "max")
    
    def pop(self):
        val = self.valArray.pop()
        self.size -= 1

        self.update(self.size - 1, 0, "sum")
        self.update(self.size - 1, float('inf'), "min")
        self.update(self.size - 1, float('-inf'), "max")

        if self.size < self.capacity // 2 and self.capacity > 1:
            self.capacity = self.capacity//2
            self.sumTree = self.buildTree(self.valArray, "sum")
            self.minTree = self.buildTree(self.valArray, "min")
            self.maxTree = self.buildTree(self.valArray, "max")
            
        return val

    def update(self, index, val, type):

        def minCond(a, b):
            return min(a, b)

        def maxCond(a, b):
            return max(a, b)

        def sumCond(a, b):
            return a + b

        condition = -1
        match type:
            case "min":
                condition = minCond
                tree = self.minTree
            case "max":
                condition = maxCond
                tree = self.maxTree
            case "sum":
                condition = sumCond
                tree = self.sumTree
            case _:
                print("Wrong condition")
                return
        
        size = len(tree)//2
        i = size + index
        tree[i] = val

        # Update upwards
        i = i//2
        while i >= 1:
            tree[i] = condition(tree[2 * i], tree[2 * i + 1])
            i = i//2
    
    def range(self,i,j,type):

        def minCond(a,b):
            return min(a,b)
        
        def maxCond(a,b):
            return max(a,b)
        
        def sumCond(a,b):
            return a+b
        
        if i >= self.size or j >= self.size:
            print("Incorrect indices")
            return 

        condition = -1
        tree = []
        match type:
            case "min":
                condition = minCond
                tree = self.minTree
                val = float("inf")
            case "max":
                condition = maxCond
                tree = self.maxTree
                val = float("-inf")
            case "sum":
                condition = sumCond
                tree = self.sumTree
                val = 0
            case _:
                print("Wrong condition")
                return

        i += self.capacity
        j += self.capacity

        while i <= j:
            if (i % 2 == 1):
                val = condition(val,tree[i])
                i += 1
            if (j % 2 == 0):
                val = condition(val,tree[j])
                j -= 1
            i = i//2
            j = j//2
        return val

    def printContents(self):
        print("ValArray",self.valArray)
        print("MinTreeArray",self.minTree)
        print("maxTreeArray",self.maxTree)
        print("sumTreeArray",self.sumTree)

class BinarySearchTree():

    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    def __init__(self):
        self.root = None

    def insert(self, key):
        def _insert(node, key):
            if node is None:
                return BinarySearchTree.Node(key)
            if key < node.key:
                node.left = _insert(node.left, key)
            elif key > node.key:
                node.right = _insert(node.right, key)
            # Ignore duplicates
            return node

        self.root = _insert(self.root, key)

    def search(self, key):
        def _search(node, key):
            if node is None:
                return False
            if key == node.key:
                return True
            elif key < node.key:
                return _search(node.left, key)
            else:
                return _search(node.right, key)

        return _search(self.root, key)

    def delete(self, key):
        def _min_value_node(node):
            current = node
            while current.left is not None:
                current = current.left
            return current

        def _delete(node, key):
            if node is None:
                return None
            if key < node.key:
                node.left = _delete(node.left, key)
            elif key > node.key:
                node.right = _delete(node.right, key)
            else:
                if node.left is None:
                    return node.right
                elif node.right is None:
                    return node.left
                temp = _min_value_node(node.right)
                node.key = temp.key
                node.right = _delete(node.right, temp.key)
            return node

        self.root = _delete(self.root, key)

    def inorder_traversal(self):
        result = []
        def _inorder(node):
            if node:
                _inorder(node.left)
                result.append(node.key)
                _inorder(node.right)
        _inorder(self.root)
        return result

    def preorder_traversal(self):
        result = []
        def _preorder(node):
            if node:
                result.append(node.key)
                _preorder(node.left)
                _preorder(node.right)
        _preorder(self.root)
        return result

    def postorder_traversal(self):
        result = []
        def _postorder(node):
            if node:
                _postorder(node.left)
                _postorder(node.right)
                result.append(node.key)
        _postorder(self.root)
        return result
    
#G = UnDirectedGraph(5)
#G.insertUndirectedEdge(0,1)
#G.insertUndirectedEdge(1,2)
#G.insertUndirectedEdge(2,3)
#G.insertUndirectedEdge(3,4)
#G.insertUndirectedEdge(1,3)
