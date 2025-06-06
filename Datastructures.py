import math
import copy

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

        def mergeSortedArrays(A,B):
            result = []
            while len(A) + len(B) > 0:
                    if len(A) > 0 and len(B) > 0:
                        if A[0] < B[0]:
                            result.append(A[0])
                            del A[0]
                        else:
                            result.append(B[0])
                            del B[0]
                    elif len(A) > 0:
                        for i in A:
                            result.append(i)
                        A = []
                    elif len(B) > 0:
                        for i in B:
                            result.append(i)
                        B = []
            return result

        while n < len(arr):
            for seg in range(0,math.ceil(len(arr)/(2*n))):
                min = 2*n*seg
                leftarr = arr[min:min+n]
                rightarr = arr[min+n:min+2*n]
                newarr = mergeSortedArrays(leftarr,rightarr)
                arr[min:min+2*n] = newarr
            n *= 2
        
        return arr

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

class Stack:
    def __init__(self):
        self.stackarr = []
    def top(self):
        if len(self.stackarr) > 0:
            return self.stackarr[-1]
    def push(self,val):
        self.stackarr.append(val)
    def pop(self):
        if not self.isEmpty():
            val =  self.stackarr[-1]
            self.stackarr = self.stackarr[0:len(self.stackarr)-1]
            return val
    def isEmpty(self):
        return len(self.stackarr) == 0
    def printContents(self):
        print(self.stackarr)

class Queue:
    def __init__(self):
        self.stackarr = []
    def enqueue(self,val):
        self.stackarr.append(val)
    def dequeue(self):
        if len(self.stackarr) > 0:
            val =  self.stackarr[0]
            self.stackarr = self.stackarr[1:len(self.stackarr)]
            return val
    def isEmpty(self):
        return len(self.stackarr) == 0
    def printContents(self):
        print(self.stackarr)

class PriorityQueue:
    class heapNode:
        def __init__(self,key,val):
            self.key = key
            self.val = val

    def __init__(self,condition):
        self.heap = [PriorityQueue.heapNode(float('nan'),float('nan'))]
        self.keymap = {}
        self.size = 0
        #max = "max", min = "min"
        self.condition = condition
    
    def swapNode(self, i, j):
        self.keymap[self.heap[i].key], self.keymap[self.heap[j].key] = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def extreme(self):
        return self.heap[1]
    
    def extractExtreme(self):
        rootNode = self.heap[1]
        #Remove the first value from the heap
        del self.keymap[rootNode.key]
        #now copy the last value to the first and bubble down
        if self.size > 1:
            last = self.size
            self.heap[1] = self.heap[last]
            self.keymap[self.heap[1].key] = 1
            self.heap.pop()
            self.bubbleDown(1)
        elif self.size == 1:
            #Since the root node has been deleted, the heap is now empty.
            self.heap.pop()
        self.size -= 1
        return {"key":rootNode.key,"val":rootNode.val}
    
    def heapCondition(self,parentNode, childNode):
        if self.condition == "min":
            if parentNode.val > childNode.val:
                return False
        elif self.condition == "max":
            if parentNode.val < childNode.val:
                return False
        return True

    def bubbleDown(self,index):
        #Check if this has left leaf node
        leftchild = 2*index
        if self.size > leftchild:
            if not self.heapCondition(self.heap[index], self.heap[leftchild]):
                #First switch node values
                self.swapNode(index,leftchild)
            
                self.bubbleDown(leftchild)
        #Check if this has right leaf node
        rightchild = 2*index + 1
        if self.size > rightchild:
            if not self.heapCondition(self.heap[index], self.heap[rightchild]):
                #First switch node values
                self.swapNode(index,rightchild)
                
                self.bubbleDown(rightchild)
    
    def bubbleUp(self,index):
        #Check if this is root node
        if index == 1:
            return
        parent = (index) // 2
        if not self.heapCondition(self.heap[parent], self.heap[index]):
            #First switch keymap values
            temp = self.keymap[self.heap[index].key]
            self.keymap[self.heap[index].key] = self.keymap[self.heap[parent].key]
            self.keymap[self.heap[parent].key] = temp
            #Now switch values in heap
            temp = self.heap[index]
            self.heap[index] = self.heap[parent]
            self.heap[parent] = temp
            self.bubbleUp(parent)

    def insert(self, keypair):
        #The keypair needs to be in the format: {"key":key,"val":value}
        if keypair['key'] not in self.keymap:
            self.heap.append(PriorityQueue.heapNode(keypair['key'],keypair['val']))
            self.size+=1
            self.keymap.update({keypair['key']:self.size})

            self.bubbleUp(self.keymap[keypair['key']])   
    
    def updateKey(self, keypair):
        if keypair['key'] in self.keymap:
            targetNode = self.heap[self.keymap[keypair['key']]]
            if targetNode.val > keypair['val'] and self.condition == "min":
                targetNode.val = keypair['val']
                self.bubbleUp(self.keymap[keypair['key']])
                return True
            elif targetNode.val < keypair['val'] and self.condition == "max":
                targetNode.val = keypair['val']
                self.bubbleDown(self.keymap[keypair['key']])
                return True   
        return False

    def isEmpty(self):
        return self.size == 0

    def printContents(self):
        print([(node.key,node.val) for node in self.heap])
    
    def hasKey(self,key):
        return key in self.keymap
    
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
        S = Queue()
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

        S.enqueue((v0,False))
        while(not S.isEmpty()):
            curNode = S.dequeue()
            #Check if target
            if curNode == target:
                return path(curNode)
            for neighbour in self.adjarr[curNode]:
                if not visited[neighbour.to]:
                    S.enqueue(neighbour.to)
                    parent[neighbour.to] = curNode
                    visited[neighbour.to] = True
            visitpath.append(curNode)
        return (visitpath)
    
    #Returns DFS path
    def DFS(self,v0):
        S = Stack()
        visited = [False for i in range(0,self.size)]
        visited[v0] = True
        visitpath = []
        postpath = []
        S.push((v0,False))
        while(not S.isEmpty()):
            curNode, visit = S.pop()
            if visit:
                postpath.append(curNode)
            else:
                S.push((curNode,True))
                visitpath.append(curNode)
                for neighbour in self.adjarr[curNode]:
                    if not visited[neighbour.to]:
                        S.push((neighbour.to,False))
                        visited[neighbour.to] = True
        return [visitpath, postpath]
    
    def Dijkstra(self,v0,target = -1):
        P = PriorityQueue("min")        
        #Initialize priority queue, visited and minDist
        visited = []
        parent = [i for i in range(self.size)]
        minDist = [float('inf') for i in range(0,self.size)]
        minDist[v0] = 0

        def path(curNode):
            path = []
            checkingNode = curNode['key']
            while checkingNode != parent[checkingNode]:
                path.append(checkingNode)
                checkingNode = parent[checkingNode]
            path.append(checkingNode)
            return (minDist[curNode['key']],path[::-1])
        #Algorithm
        P.insert({"key":v0,"val":0})

        while not P.isEmpty():
            curNode = P.extractExtreme()
            visited.append(curNode['key'])
            #Check if target
            if curNode['key'] == target:
                return path(curNode)
            for neighbour in self.adjarr[curNode['key']]:
                if neighbour.to not in visited:
                    #RelaxEdges
                    if not P.hasKey(neighbour.to):
                        P.insert({"key":neighbour.to,"val":curNode['val']+neighbour.weight})
                        minDist[neighbour.to] = curNode['val']+neighbour.weight
                        parent[neighbour.to] = curNode['key']
                    else:
                        pathAdded = P.updateKey({"key":neighbour.to,"val":curNode['val']+neighbour.weight})
                        if pathAdded:
                            parent[neighbour.to] = curNode['key']
                            minDist[neighbour.to] = curNode['val']+neighbour.weight
        if target == -1:
            return minDist
        else:
            return []
        
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
                        elif degrees["out"] == degrees["din"] + 1 and vStart == -1:
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
                    
        usedEdges = filledUsedEdges(False)
        S = Stack()
        curNode = v0

        S.push(curNode)
        path = []
        while not allNodesUsed():
            #Follow random path
            followed = False

            if curEdgesUsed(usedEdges[curNode]):
                curNode = S.pop()
                path.append(curNode)

            else:
                for markedindex in range(0,len(usedEdges[curNode])):
                    if not followed:
                        if not usedEdges[curNode][markedindex]:
                            prev = curNode
                            toNode = self.adjarr[curNode][markedindex].to

                            removeEdge(prev,toNode)
                            S.push(toNode)

                            curNode = toNode
                            followed = True
        while not S.isEmpty():
            path.append(S.pop())
        return path
    
    def allCC(self):
        nodes = [i for i in range(0,self.size)]
        result = []
        while len(nodes) > 0:
            curSCC = self.DFS(nodes[0])[0]
            for i in curSCC:
                del nodes[nodes.index(i)]
            result.append(curSCC)
        return result

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
            self.unionarr[indexA] = indexB
    
    def isConnected(self,indexA,indexB):
        return self.find(indexA) == self.find(indexB)

    def printUnionContents(self):
        print(self.unionarr)
    
    def insertUndirectedEdge(self,vFrom,vTo,weight=1):
        self.adjarr[vFrom].append(Graph.dirEdge(vFrom,vTo,weight))
        self.adjarr[vTo].append(Graph.dirEdge(vTo,vFrom,weight))
        self.union(vFrom,vTo)
    
    def Kruskal(self):
        #Check if connected
        if len(self.allSCC()) != 1:
            return UnDirectedGraph(0)

        #add all edges to priority queue, remove all duplicate edges (same origin,to), keep the lowest weight
        P = PriorityQueue("min")
        for vertexlist in self.adjarr:
            for edge in vertexlist:
                sortedVertix = sorted([edge.to,edge.origin])
                key = str(sortedVertix[0]) + ":" + str(sortedVertix[1])
                if not P.hasKey(key):
                    P.insert({"key":key,"val":edge.weight})
                else:
                    P.updateKey({"key":key,"val":edge.weight})
        
        def graphEdgesNum(undirGraph):
            num = 0
            for vertexlist in undirGraph.adjarr:
                for edge in vertexlist:
                    num += 0.5
            return num

        #Algorithm
        MST = UnDirectedGraph(self.size)
        while graphEdgesNum(MST) != self.size - 1:
            curEdge = P.extractExtreme()
            edgeFrom = int(curEdge['key'].split(":")[0])
            edgeTo = int(curEdge['key'].split(":")[1])
            if MST.find(edgeFrom) != MST.find(edgeTo):
                MST.insertUndirectedEdge(edgeFrom,edgeTo,curEdge['val'])

        return (MST.totalWeight(),MST)

    def Prim(self,v0):
        #Check if connected
        if len(self.allSCC()) != 1:
            return UnDirectedGraph(0)
    
        #Like dijkstra's, but with neighbour.weight, not cur[key] + neightbour.weight
        P = PriorityQueue("min")
        MST = UnDirectedGraph(self.size)
        inMST = [False for i in range(0,self.size)]

        prev = v0
        curNode = v0
        inMST[v0] = True

        #Initialize Priority queue
        for neighbour in self.adjarr[v0]:
            P.insert({"key":neighbour.to,"val":neighbour.weight})
        
        while not P.isEmpty():
            curEdge = P.extractExtreme()
            MST.insertUndirectedEdge(prev,curEdge['key'],curEdge['val'])
            prev = curNode
            curNode = curEdge['key']
            inMST[curNode] = True

            for neighbour in self.adjarr[curNode]:
                if not inMST[neighbour.to]:
                    if P.hasKey(neighbour.to):
                        P.updateKey({"key":neighbour.to,"val":neighbour.weight})
                    else:
                        P.insert({"key":neighbour.to,"val":neighbour.weight})
        
        return (MST.totalWeight(),MST)
    
    def isBipartite(self):
        Q = Queue()
        visited = [False for i in range(0,self.size)]
        color = [-1 for i in range(0,self.size)]

        Q.enqueue(0)
        color[0] = 0
        visited[0] = True

        while not Q.isEmpty():
            curNode = Q.dequeue()
    
            for neighbour in self.adjarr[curNode]:
                if visited[neighbour.to] and color[neighbour.to] == color[curNode]:
                    return False
                if not visited[neighbour.to]:
                    Q.enqueue(neighbour.to)
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
        zeroStack = Stack()
        path = []

        for vertexlist in self.adjarr:
            for edge in vertexlist:
                degreeList[edge.to] += 1
        
        def addZeroNodes():
            for vertex in range(0, self.size):
                if degreeList[vertex] == 0:
                    zeroStack.push(vertex)

        addZeroNodes()
        while not zeroStack.isEmpty():
            curNode = zeroStack.pop()
            path.append(curNode)
            for neighbour in self.adjarr[curNode]:
                degreeList[neighbour.to] -= 1
                if degreeList[neighbour.to] == 0:
                    zeroStack.push(neighbour.to)

        if len(path) == len(self.adjarr):
            return path
        else:
            return []
    
    def isDAG(self):
        return not self.TopologicalSort == []
    
    def TopologicalShortestPath(self,v0,target = -1):
        topoSort = self.TopologicalSort()
        #See if graph isn''t DAG
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
            self.capacity = math.floor(self.capacity/2)
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
        
        size = math.floor(len(tree)/2)
        i = size + index
        tree[i] = val

        # Update upwards
        i = math.floor(i/2)
        while i >= 1:
            tree[i] = condition(tree[2 * i], tree[2 * i + 1])
            i = math.floor(i/2)
    
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
            i = math.floor(i/2)
            j = math.floor(j/2)
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

    def print_tree(self):
        def _print(node, prefix="", is_left=True):
            if node.right:
                _print(node.right, prefix + ("│   " if is_left else "    "), False)
            print(prefix + ("└── " if is_left else "┌── ") + str(node.key))
            if node.left:
                _print(node.left, prefix + ("    " if is_left else "│   "), True)
        if self.root:
            _print(self.root)
        else:
            print("Empty tree")

BST = BinarySearchTree()
BST.insert(6)
BST.insert(5)
BST.insert(1)
BST.insert(3)
BST.insert(2)
BST.insert(4)

BST.print_tree()