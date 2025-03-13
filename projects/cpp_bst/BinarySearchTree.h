#ifndef BINARYSEARCHTREE_H
#define BINARYSEARCHTREE_H

#include <iostream>

class BinarySearchTree
{
private:
    struct Node
    {
        int value;
        Node *left;
        Node *right;

        Node(int val);
    };

    Node *root;
    Node *insert(Node *node, int value);
    void print(Node *node, int space);
    Node *search(Node *node, int value);
    Node *deleteNode(Node *node, int value);
    Node *minValueNode(Node *node);

public:
    BinarySearchTree();
    void insert(int value);
    void print();
    bool search(int value);
    void deleteValue(int value);
};

#endif
