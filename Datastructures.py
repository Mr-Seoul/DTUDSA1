import copy

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
        self.heap = [PriorityQueue.heapNode(-1,-1)]
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
    
    def allCC(self):
        nodes = [i for i in range(0,self.size)]
        result = []
        while len(nodes) > 0:
            curSCC = self.DFS(nodes[0])
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

G1 = DirectedGraph(5)
G1.insertDirectedEdge(0,1)
G1.insertDirectedEdge(1,2)
G1.insertDirectedEdge(2,3)
G1.insertDirectedEdge(3,1)

print(G1.allSCC())

#Both: Hierholzer's algorithm

#Add datastructures Binary search trees, 2-3 trees, minimum subarray values

#Add merge sort, insert sort and quicksort