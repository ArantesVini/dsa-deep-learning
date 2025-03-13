#include "BinarySearchTree.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int main()
{
    BinarySearchTree bst;
    bool running = true;
    char input[999];

    while (running)
    {
        cout << "\nChoose an option from the menu (Type the letter in uppercase):" << endl;
        cout << "(L)oad, (P)rint, (R)emove, (S)earch, (Q)uit" << endl;
        cin >> input;

        switch (input[0])
        {
        case 'L':
        {
            cout << "(T)erminal (F)ile" << endl;
            cin >> input;
            if (input[0] == 'T')
            {
                cout << "Type numbers separated by space, e.g.: 1 2 3 4 5" << endl;
                cin.ignore();
                string line;
                getline(cin, line);
                istringstream iss(line);
                int num;
                while (iss >> num)
                    bst.insert(num);
            }
            else if (input[0] == 'F')
            {
                cout << "Type the filename:" << endl;
                string filename;
                cin >> filename;
                ifstream file(filename);
                int num;
                while (file >> num)
                    bst.insert(num);
                cout << "File loaded into memory!" << endl;
            }
            break;
        }
        case 'P':
            bst.print();
            break;
        case 'R':
        {
            cout << "Type the value you want to remove:" << endl;
            int delVal;
            cin >> delVal;
            bst.deleteValue(delVal);
            break;
        }
        case 'S':
        {
            cout << "Type the value you want to search:" << endl;
            int searchVal;
            cin >> searchVal;
            bool found = bst.search(searchVal);
            if (found)
                cout << "SUCCESS! The value was found in the dataset!" << endl;
            else
                cout << "The value was not found in the dataset!" << endl;
            break;
        }
        case 'Q':
            running = false;
            break;
        }
    }

    return 0;
}
