#include "BinarySearchTree.h"
#include <iostream>

using namespace std;

BinarySearchTree::Node::Node(int val) : value(val), left(nullptr), right(nullptr) {}

BinarySearchTree::BinarySearchTree() : root(nullptr) {}

BinarySearchTree::Node *BinarySearchTree::insert(Node *node, int value)
{
    if (!node)
        return new Node(value);

    if (value < node->value)
        node->left = insert(node->left, value);
    else
        node->right = insert(node->right, value);

    return node;
}

void BinarySearchTree::insert(int value)
{
    root = insert(root, value);
}

void BinarySearchTree::print(Node *node, int space)
{
    if (!node)
        return;

    space += 10;
    print(node->right, space);
    cout << endl;

    for (int i = 10; i < space; i++)
        cout << " ";

    cout << node->value << "\n";
    print(node->left, space);
}

void BinarySearchTree::print()
{
    print(root, 0);
}

BinarySearchTree::Node *BinarySearchTree::search(Node *node, int value)
{
    if (!node || node->value == value)
        return node;

    if (value < node->value)
        return search(node->left, value);
    else
        return search(node->right, value);
}

bool BinarySearchTree::search(int value)
{
    return search(root, value) != nullptr;
}

BinarySearchTree::Node *BinarySearchTree::deleteNode(Node *node, int value)
{
    if (!node)
        return node;

    if (value < node->value)
        node->left = deleteNode(node->left, value);
    else if (value > node->value)
        node->right = deleteNode(node->right, value);
    else
    {
        if (!node->left)
        {
            Node *temp = node->right;
            delete node;
            return temp;
        }
        else if (!node->right)
        {
            Node *temp = node->left;
            delete node;
            return temp;
        }

        Node *temp = minValueNode(node->right);
        node->value = temp->value;
        node->right = deleteNode(node->right, temp->value);
    }

    return node;
}

void BinarySearchTree::deleteValue(int value)
{
    root = deleteNode(root, value);
}

BinarySearchTree::Node *BinarySearchTree::minValueNode(Node *node)
{
    Node *current = node;
    while (current && current->left)
        current = current->left;
    return current;
}
