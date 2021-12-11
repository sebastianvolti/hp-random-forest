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

vector<int> FileLoader::readParameters(const char* fileName) {
	vector<int> params;
	ifstream file;
	string str;
	file.open(fileName);
	while (getline(file, str)) {
		params.push_back(stoi(str));
	}
	return params;
}

string FileLoader::writeFile(int parametro, vector<vector<int>> resultados, vector<int> valoresHiperparam, int cantIter, int duration) {

	time_t rawtime;
	struct tm* timeinfo;
	char buffer[80];
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	strftime(buffer, sizeof(buffer), "%Y-%m-%d %H.%M.%S", timeinfo);
	string str(buffer);
	str.append(".txt");

	ofstream MyFile(str.c_str());

	MyFile << "-- RESULTADOS --" << endl;
	MyFile << "-- Tiempo procesamiento: " << duration << "seg" << endl;

	string nombreParam;
	if (parametro == 0) nombreParam = "Cant Arboles";
	else if (parametro == 1) nombreParam = "Cant instancias";
	else if (parametro == 1) nombreParam = "Cant Atributos";
	else nombreParam = "Otro";

	MyFile << "Parametro a evaluar: " << nombreParam << endl;
	MyFile << "Valores evaluados: ";
	for (unsigned int i = 0; i < valoresHiperparam.size(); i++) {
		MyFile << valoresHiperparam[i] << " ";
	}
	MyFile << endl;

	for (unsigned int i = 1; i <= cantIter; i++) {
		if (i == 1) 
			MyFile << "Iter |" << i;
		else 
			MyFile << " - " << i;
	}

	MyFile << endl;

	// Hago los calculos finales
	for (unsigned int i = 0; i < resultados.size(); i++) {
		float promedioHiperparam;
		MyFile << "Val: " << valoresHiperparam[i]  << "   |";

		for (unsigned int j = 0; j < resultados[i].size(); j++) {
			MyFile << resultados[i][j] << " - ";
			promedioHiperparam += resultados[i][j];
		}
		promedioHiperparam = promedioHiperparam / resultados[i].size();
		MyFile << ": Promedio final = " << promedioHiperparam << endl;
	}

	// Close the file
	MyFile.close();
	return str;
}

pair<int, int> FileLoader::findIterationToMake(vector<vector<bool>> iteracionesTotales) {
	for (unsigned int i = 0; i < iteracionesTotales.size(); i ++) {
		for (unsigned int j = 0; j < iteracionesTotales[i].size(); j++) {
			if (!iteracionesTotales[i][j]) {
				return make_pair(i,j);
			}
		}
	}
	return make_pair(-1, -1);
}

int FileLoader::getIndexFreeSlave(vector<bool> slaves) {
	for (unsigned int indiceEsclavo = 0; indiceEsclavo < slaves.size(); indiceEsclavo++) {
		if (slaves[indiceEsclavo] == true) {
			return indiceEsclavo;
		}
	}
	return -1;
}

int FileLoader::getIndexWorkingSlave(vector<bool> slaves) {
	for (unsigned int indiceEsclavo = 0; indiceEsclavo < slaves.size(); indiceEsclavo++) {
		if (slaves[indiceEsclavo] == false) {
			return indiceEsclavo;
		}
	}
	return -1;
}

void FileLoader::printVectorBool(string title, vector<bool> vect) {
	cout << title;
	for (unsigned int i = 0; i < vect.size(); i++) {
		if (i == vect.size() - 1) 
			cout << vect[i] << endl;
		else 
			cout << vect[i] << " ";
	}
	cout << endl;
}

void FileLoader::printVectorInt(string title, vector<int> vect) {
	cout << title;
	for (unsigned int i = 0; i < vect.size(); i++) {
		if (i == vect.size() - 1)
			cout << vect[i] << endl;
		else
			cout << vect[i] << " ";
	}
	cout << endl;
}

void FileLoader::printMatrixInt(string title, vector<vector<bool>> vect) {
	cout << title << endl;
	for (unsigned int i = 0; i < vect.size(); i++) {
		for (unsigned int j = 0; j < vect[i].size(); j++) {
			if (j == vect[i].size() - 1)
				cout << vect[i][j] << endl;
			else if (j == 0)
				cout << " HP " << i << " "<< vect[i][j] << " ";
			else
				cout << vect[i][j] << " ";
		}
	}
	cout << endl;
}