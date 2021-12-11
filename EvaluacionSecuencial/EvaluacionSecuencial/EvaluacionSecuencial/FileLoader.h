#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

class FileLoader
{

public:
	FileLoader();

	// Lee el archivo y lo parsea para cargarlo a alguna matriz
	vector<vector<char>> readFile(const char* fileName);
};

