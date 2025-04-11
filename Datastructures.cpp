#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>

using namespace std;

//Vector printing
template <typename T>
void printVec(const vector<T>& arr) {
    for (T i: arr) {
        cout << i << " ";
    }
    cout << "\n";
}

//Vector manipulation
template <typename T>
vector<T> subVector(const vector<T>& arr, int pos,int len) {
    vector<T> result = {};
    if (pos < 0) {
        len -= abs(pos);
        pos = 0;
    }
    if (pos > arr.size()-1) {
        return result;
    }
    for (int i = pos; i < pos+len; i++) {
        result.emplace_back(arr[i]);
        if (i+1 >= arr.size()) {
            return result;
        }
    }
    return result;
}

//Add function to insert and remove vector values

//Vector sorting code
template <typename T>
void sortVecIncr(vector<T>& arr) {
    sort(arr.begin(), arr.end(), [](T a, T b) {
        return a < b; 
    });
}

template <typename T>
void sortVecDecr(vector<T>& arr) {
    sort(arr.begin(), arr.end(), [](T a, T b) {
        return a > b;
    });
}

//Searching code
template <typename T>
T linSearch(const vector<T>& arr, const T& target) {
    for (T i = 0; i < arr.size(); i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

//Iterator code
template <typename T>
T iteratorToIndex(const vector<T>& arr, typename vector<T>::iterator& iterator) {
    return iterator-arr.begin();
}

//String manipulation
string subString(const string& str, const int& pos, const int& len) {
    if (str.size()-pos < len) {
        return "";
    }
    string result = "";
    for (int i = pos; i < pos+len;i++) {
        result += str[i];
    }
    return result;
}

vector<string> stringSplit(const string& str, const string& splitString) {
    vector<string> result = {};
    string curString = "";
    for (int i = 0; i < str.size();) {
        if (subString(str,i,splitString.size()) == splitString) {
            i += splitString.size();
            if (curString != "") {
                result.emplace_back(curString);
                curString = "";
            }
        } else {
            curString += str[i];
            i++;
            if (i == str.size()) {
                result.emplace_back(curString);
            }
        }
    }
    return result;
}

string stringJoin(const vector<string>& arr, const string& joinString) {
    string result = "";
    for (int i = 0; i < arr.size()-1;i++) {
        result += arr[i];
        result += joinString;
    } 
    result += arr[arr.size()-1];
    return result;
}

class Stack {
    private:
        struct Node {
            int data;
            Node* next;
            Node(int value) : data(value), next(nullptr) {}
        };
    
        Node* top;
    
    public:
        Stack() : top(nullptr) {}
    
        // Push an element onto the stack
        void push(int value) {
            Node* newNode = new Node(value);
            newNode->next = top;
            top = newNode;
        }
    
        // Pop an element from the stack and return it
        int pop() {
            if (!isEmpty()) {
                Node* temp = top;
                int topElement = top->data;
                top = top->next;
                delete temp;
                return topElement;
            }
            throw out_of_range("Stack is empty");
        }
    
        // Check if the stack is empty
        bool isEmpty() const {
            return top == nullptr;
        }
    
        // Print the elements in the stack
        void printStack() const {
            if (isEmpty()) {
                cout << "Stack is empty." << endl;
            } else {
                Node* current = top;
                cout << "Stack elements: ";
                while (current != nullptr) {
                    cout << current->data << " ";
                    current = current->next;
                }
                cout << endl;
            }
        }
    
        // Destructor to free memory
        ~Stack() {
            while (!isEmpty()) {
                pop();
            }
        }
};

class Queue {
    private:
        struct Node {
            int data;
            Node* next;
            Node(int value) : data(value), next(nullptr) {}
        };
    
        Node* front;
        Node* back;
    
    public:
        Queue() : front(nullptr), back(nullptr) {}
    
        // Enqueue an element to the queue
        void enqueue(int value) {
            Node* newNode = new Node(value);
            if (isEmpty()) {
                front = back = newNode;
            } else {
                back->next = newNode;
                back = newNode;
            }
        }
    
        // Dequeue an element from the queue and return it
        int dequeue() {
            if (!isEmpty()) {
                Node* temp = front;
                int frontElement = front->data;
                front = front->next;
                delete temp;
                if (front == nullptr) { // If the queue is now empty, set back to nullptr
                    back = nullptr;
                }
                return frontElement;
            }
            throw out_of_range("Queue is empty");
        }
    
        // Check if the queue is empty
        bool isEmpty() const {
            return front == nullptr;
        }
    
        // Print the elements in the queue
        void printQueue() const {
            if (isEmpty()) {
                cout << "Queue is empty." << endl;
            } else {
                Node* current = front;
                cout << "Queue elements: ";
                while (current != nullptr) {
                    cout << current->data << " ";
                    current = current->next;
                }
                cout << endl;
            }
        }
    
        // Destructor to free memory
        ~Queue() {
            while (!isEmpty()) {
                dequeue();
            }
        }
};

class Graph {
    private:
        vector<vector<int>> adjList;
        int numVertices;
    
        vector<int> parent; 
        vector<int> rank;    
    
        int find(int u) {
            if (parent[u] != u) {
                parent[u] = find(parent[u]);  
            }
            return parent[u];
        }
    
        void unionSets(int u, int v) {
            int rootU = find(u);
            int rootV = find(v);
    
            if (rootU != rootV) {
                if (rank[rootU] > rank[rootV]) {
                    parent[rootV] = rootU;
                } else if (rank[rootU] < rank[rootV]) {
                    parent[rootU] = rootV;
                } else {
                    parent[rootV] = rootU;
                    rank[rootU]++;
                }
            }
        }
    
        void rebuildUnionFind() {
            parent.resize(numVertices);
            rank.resize(numVertices, 0);
            for (int i = 0; i < numVertices; ++i) {
                parent[i] = i;
            }
    
            for (int u = 0; u < numVertices; ++u) {
                for (int v : adjList[u]) {
                    unionSets(u, v);
                }
            }
        }
    
    public:
        Graph(int vertices) : numVertices(vertices), parent(vertices), rank(vertices, 0) {
            adjList.resize(vertices);

            for (int i = 0; i < vertices; ++i) {
                parent[i] = i;
            }
        }
    
        Graph(const vector<vector<int>>& adj) {
            numVertices = adj.size();
            adjList = adj;
            parent.resize(numVertices);
            rank.resize(numVertices, 0);
    
            for (int i = 0; i < numVertices; ++i) {
                parent[i] = i;
            }
    
            rebuildUnionFind();
        }
    
        void insertUndirectedEdge(int u, int v) {
            adjList[u].push_back(v);
            adjList[v].push_back(u);
            unionSets(u, v);  
        }

        
        void insertDirectedEdge(int u, int v) {
            adjList[u].push_back(v);  

            unionSets(u, v); 
        }
    
        void removeUndirectedEdge(int u, int v) {
            adjList[u].erase(remove(adjList[u].begin(), adjList[u].end(), v), adjList[u].end());
            adjList[v].erase(remove(adjList[v].begin(), adjList[v].end(), u), adjList[v].end());
    
            rebuildUnionFind();
        }

        void removeDirectedEdge(int u, int v) {
            adjList[u].erase(remove(adjList[u].begin(), adjList[u].end(), v), adjList[u].end());

            rebuildUnionFind();
        }

    
        vector<int> bfs(int start) {
            vector<bool> visited(numVertices, false);
            Queue q;
            vector<int> bfsOrder;
    
            visited[start] = true;
            q.enqueue(start);
    
            while (!q.isEmpty()) {
                int node = q.dequeue();
                bfsOrder.push_back(node);
    
                for (int neighbor : adjList[node]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.enqueue(neighbor);
                    }
                }
            }
            return bfsOrder;
        }

        vector<int> dfs(int start) {
            vector<bool> visited(numVertices, false);
            Stack s;
            vector<int> dfsOrder;
    
            s.push(start);
    
            while (!s.isEmpty()) {
                int node = s.pop();
    
                if (!visited[node]) {
                    visited[node] = true;
                    dfsOrder.push_back(node);
    
                    for (int i = adjList[node].size() - 1; i >= 0; --i) { 
                        int neighbor = adjList[node][i];
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                        }
                    }
                }
            }
            return dfsOrder;
        }
    
        vector<int> bfsPath(int start, int end) {
            vector<int> parent(numVertices, -1);
            vector<bool> visited(numVertices, false);
            Queue q;
            vector<int> path;
    
            visited[start] = true;
            q.enqueue(start);
    
            while (!q.isEmpty()) {
                int node = q.dequeue();
    
                if (node == end) {
                    while (node != -1) {
                        path.push_back(node);
                        node = parent[node];
                    }
                    reverse(path.begin(), path.end());
                    return path;
                }
    
                for (int neighbor : adjList[node]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        parent[neighbor] = node;
                        q.enqueue(neighbor);
                    }
                }
            }
            return path;  // If no path exists, return an empty vector
        }
    
        bool connected(int u, int v) {
            if (adjList.empty() || (adjList[u].empty() && adjList[v].empty())) {
                return false; 
            }
            return find(u) == find(v);
        }
    
        void printGraph() const {
            for (int i = 0; i < numVertices; ++i) {
                cout << i << ": ";
                for (int neighbor : adjList[i]) {
                    cout << neighbor << " ";
                }
                cout << endl;
            }
        }
    
        void printConnectedComponents() {
            for (int i = 0; i < numVertices; ++i) {
                cout << "Vertex " << i << " is in set " << find(i) << endl;
            }
        }
    };

class PriorityQueue {
    private:
        vector<int> heap;  
        int size;  
    
        int left(int i) {
            return 2 * i + 1;
        }
    
        int right(int i) {
            return 2 * i + 2;
        }
    
        int parent(int i) {
            return (i - 1) / 2;
        }
    
        void bubbleDown(int i) {
            int largest = i;
            int l = left(i);
            int r = right(i);
    
            if (l < size && heap[l] > heap[largest]) {
                largest = l;
            }
    
            if (r < size && heap[r] > heap[largest]) {
                largest = r;
            }
    
            if (largest != i) {
                swap(heap[i], heap[largest]);
                bubbleDown(largest);  
            }
        }
    
        void bubbleUp(int i) {
            while (i > 0 && heap[parent(i)] < heap[i]) {
                swap(heap[i], heap[parent(i)]);
                i = parent(i);
            }
        }
    
    public:
        PriorityQueue() : size(0) {}
    
        int max() {
            if (size == 0) {
                return -1; 
            }
            return heap[0];
        }
    
        int extractMax() {
            if (size == 0) {
                return -1; 
            }
    
            int maxElement = heap[0];  
            heap[0] = heap[size - 1];  
            size--;
            bubbleDown(0);  
    
            return maxElement;
        }
    
        void increaseKey(int i, int k) {
            if (k < heap[i]) {
                return;  
            }
            heap[i] = k;
            bubbleUp(i);  
        }
    
        void insert(int x) {
            if (size == heap.size()) {
                heap.push_back(x); 
            } else {
                heap[size] = x;
            }
            size++;
            bubbleUp(size - 1);  
        }
    
        void printHeap() {
            for (int i = 0; i < size; ++i) {
                cout << heap[i] << " ";
            }
            cout << endl;
        }
};

int main() {
    Graph g(5);

    // Insert directed edges
    g.insertDirectedEdge(0, 1);
    g.insertDirectedEdge(1, 2);

    // Insert undirected edges
    g.insertUndirectedEdge(3, 4);

    // Print the adjacency list
    cout << "Graph adjacency list:" << endl;
    g.printGraph();

    // Remove directed edge from 0 to 1
    g.removeDirectedEdge(0, 1);

    // Print the adjacency list after removal
    cout << "Graph adjacency list after removing directed edge (0 -> 1):" << endl;
    g.printGraph();

    // Check if vertices 0 and 2 are connected
    cout << "Are 0 and 2 connected? " << (g.connected(0, 2) ? "Yes" : "No") << endl;

    g.printConnectedComponents();

    return 0;
}

//Make 2 classes, one for directed and one for undirected. Use this to implement the different algorithms

//Both: Dijkstra's

//Undirected: Unionfind (connectivity), Kruskal's algorithm (MST's)

//Directed: Topological sorting, SCC, Topological dijkstra's