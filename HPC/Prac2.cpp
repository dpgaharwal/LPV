#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;
class Graph
{
    int V;                   // Number of vertices
    vector<vector<int>> adj; // Adjacency list
public:
    Graph(int V)
    {
        this->V = V;
        adj.resize(V);
    }
    void addEdge(int v, int w)
    {
        adj[v].push_back(w);
        adj[w].push_back(v); // Since it's an undirected graph
    }
    void parallelBFS(int start)
    {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;
        cout << "Parallel BFS Traversal: ";
        while (!q.empty())
        {
            int size = q.size();
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                int node;
#pragma omp critical
                {
                    node = q.front();
                    q.pop();
                    cout << node << " ";
                }
                for (int neighbor : adj[node])
                {
                    if (!visited[neighbor])
                    {
#pragma omp critical
                        {
                            if (!visited[neighbor])
                            { // Double-check condition inside critical
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        cout << endl;
    }
    void parallelDFS(int start)
    {
        vector<bool> visited(V, false);
        stack<int> s;
        s.push(start);
        visited[start] = true;
        cout << "Parallel DFS Traversal: ";
        while (!s.empty())
        {
            int size = s.size();
#pragma omp parallel for
            for (int i = 0; i < size; i++)
            {
                int node;
#pragma omp critical
                {
                    node = s.top();
                    s.pop();
                    cout << node << " ";
                }
                for (int neighbor : adj[node])
                {
                    if (!visited[neighbor])
                    {
#pragma omp critical
                        {
                            if (!visited[neighbor])
                            {
                                visited[neighbor] = true;
                                s.push(neighbor);
                            }
                        }
                    }
                }
            }
        }
        cout << endl;
    }
};
int main()
{
    Graph g(6);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    cout << "Starting BFS from node 0:" << endl;
    g.parallelBFS(0);
    cout << "Starting DFS from node 0:" << endl;
    g.parallelDFS(0);
    return 0;
}