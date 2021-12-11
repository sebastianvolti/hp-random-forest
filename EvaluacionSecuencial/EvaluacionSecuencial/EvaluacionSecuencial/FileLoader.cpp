#include "FileLoader.h"

FileLoader::FileLoader() {}

vector<vector<char>> FileLoader::readFile(const char* fileName) {
	vector<vector<char>> dataset;
	ifstream file;
	string str;
	file.open(fileName);

	while (getline(file, str)) {
		vector<char> row;
		for (const char i : str) {
			if (i != ';') {
				if (i == '0' || i == '1' || i == 'n' || i == 'p') {
					row.push_back(i);
				}
			}
		}
		// Agrego instancia a la matriz
		dataset.push_back(row);
	}
	return dataset;
}
