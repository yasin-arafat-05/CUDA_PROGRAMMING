#include<iostream>
#include<vector>
using namespace std;

int main(){


    // <-----------------------------1D array--------------------------->

    // vector is an dynamic array.
    // vector initilization:

    // 1. only declaration:
    vector<int> vec1;

    // 2. declaration with a size(it's dynamic)
    // initialize with zero:
    vector<int> vec2(5);

    // 3. declaration + inilize:
    vector<int> vec3 = {1,2,3,4,5};


    // printing value with for loop:
    cout<<"value of vector two"<<endl;
    for(int i=0;i<5;i++){
        cout<<vec2[i]<< " ";
    }
    cout<<endl;


    // create an iterator for printing values:
    vector<int>::iterator v = vec3.begin();
    while(v != vec3.end()){
        cout<<"value of vector: "<<*v<<endl;
        v++;
    }


    // dynamic array proof:
    cout<<"size of vec2 before inserting vlaues: "<<vec2.size()<<endl;
    // assing value into vector 2: 
    for(int i=0;i<6;i++){
        // add vlaue based on index:
        vec2[i] = 4*i;

        // add value from the end:
        vec2.push_back(2*i);
    }
    vector<int>::iterator it = vec2.begin();
    while(it!=vec2.end()){
        cout<<*it<<endl;
        it++;
    }
    cout<<"size of vec2 after inserting vlaues: "<<vec2.size()<<endl;



    // <-----------------------------2D array--------------------------->

    // only declaration:
    vector<vector<int>> vec2D1;


    // declaration + initilization:
    vector<vector<int>> vec2D2 = {
        {1,2,3},
        {4,5,6},
        {8,9,1}
    };

    // iterating the matrix 1:
    vector<vector<int>>::iterator row = vec2D2.begin();
    while(row!=vec2D2.end()){
        vector<int>::iterator col = row->begin();
        while(col != row->end()){
            cout<< *col <<" ";
            col++;
        }
        cout<<endl;
        row++;
    }

}