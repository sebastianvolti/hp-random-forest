#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>


#pragma warning(disable : 4996)

using namespace std;

class FileLoader
{

public:
	FileLoader();

	// Lee el archivo y lo parsea para cargarlo a alguna matriz
	vector<vector<char>> readFile(const char* fileName);

	string writeFile(int parametro, vector<vector<int>> resultados, vector<int> valoresHiperparam, int cantIter, int duration);

	vector<int> readParameters(const char* fileName);

	// Encuentra una iteracion que aun quede pendiente
	pair<int, int> findIterationToMake(vector<vector<bool>> iteracionesTotales);

	// Devuelve el indice del primer escalvo libre, si no hay devuelve -1
	int getIndexFreeSlave(vector<bool> slaves);

	// Devuelve el indice del primer esclavo que este ocupado, si no hay devuelve -1
	int getIndexWorkingSlave(vector<bool> slaves);

	void printVectorBool(string title, vector<bool> vect);

	void printVectorInt(string title, vector<int> vect);

	void printMatrixInt(string title, vector<vector<bool>> vect);
};

