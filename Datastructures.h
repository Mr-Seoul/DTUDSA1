//Datastructes.h
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

//Vector printing
template <typename T>
void printVec(const std::vector<T>& arr) {
    for (T i: arr) {
        std::cout << i << " ";
    }
    std::cout << "\n";
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
void sortVecInc(vector<T>& arr) {
    std::sort(arr.begin(), arr.end(), [](T a, T b) {
        return a < b; 
    });
}

template <typename T>
void sortVecDec(vector<T>& arr) {
    std::sort(arr.begin(), arr.end(), [](T a, T b) {
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

//Add doublylinkedlists, stacks, queues, graphs and add all relevant possible functions to each. 