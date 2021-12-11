// Argumentos: nombreDataset usarHilos cantArbolesDefault cantInstanciasDefault cantAttrsDefault indiceHPaValidar cantValoresParaHP val1 val2 ... valN
// Ej: ./EvaluacionSecuencial qsar_oral_toxicity.csv 1 100 2500 340 0 4 30 50 70 100

// Probar con : 30 50 70 100 arboles y 2500 instancias y 340 atributos

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include "RandomForest.h"
#include "FileLoader.h"
#include <math.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstdlib>

using namespace std;

const int rows = 8090; 		//cantidad de instancias del conjunto de entrenamiento
const int evalSize = 899; 	//cantidad de instancias del conjunto de evaluacion
const int attr = 1025;		//cantidad de atributos de cada instancia

const int partitionSizeCV = 809;  //cantidad de instancias de cada particion para validacion cruzada
const int partitionsCV = 10;		  //cantidad de particiones en validacion cruzada
const int iterationsCV = 10;       //cantidad de iteraciones en validacion cruzada

vector<vector<char>> dataset;     //dataset qsar
pair<int, int> evalDataWindowGlobal;
pair<int, int> evalDataWindowIter;

// Slave variables
map<string, vector<vector<char>>> dividedSlaveData; //dataset que un esclavo levanta
vector<vector<char>> globalEvalData;
vector<vector<char>> iterEvalData;
vector<vector<char>> iterTrainData;
char* datasetPath;

int main(int argc, char* argv[]) {
	RandomForest* rf = new RandomForest(rows, evalSize, attr, partitionSizeCV, partitionsCV, iterationsCV);
	FileLoader* fl = new FileLoader();
	int usoHilos;

	cout << "======================================== " << endl;
	cout << "Archivo dataset: " << argv[1] << endl;
	cout << "Uso hilos: " << argv[2] << endl;
	cout << "Cantidad arboles por defecto: " << argv[3] << endl;
	cout << "Cantidad de instancias por defecto: " << argv[4] << endl;
	cout << "Cantidad de atributos por defecto: " << argv[5] << endl;
	cout << "Indice del hiperparametro a validar: " << argv[6] << endl;
	cout << "Cantidad de valores para el hiperparametro a validar: " << argv[7] << endl;
	cout << "Valores a probar: ";
	for (int i = 8; i < argc; i++) {
		cout << argv[i] << " ";
	}
	cout << endl;
	cout << "======================================== " << endl;

	datasetPath = argv[1];
	usoHilos = atoi(argv[2]);

	// Valores por defecto para hiperparametros
	int treesQty = atoi(argv[3]);
	int instancesQty = atoi(argv[4]);
	int attrQty = atoi(argv[5]);

	int indiceHiperparamChequear = atoi(argv[6]);
	int valoresPosibles = atoi(argv[7]);
	vector<int> valores;

	for (int i = 0; i < valoresPosibles; i++) {
		valores.push_back(atoi(argv[i + 8]));
	}

	auto start = chrono::high_resolution_clock::now();

	// Leo el archivo linea a linea
	dataset = fl->readFile(datasetPath);

	// Determino el inicio y el fin de los datos finales de evaluacion (en este caso es el ultimo 10% de las instancias)
	evalDataWindowGlobal = rf->crossValEvalDataWindow(9, dataset.size(), iterationsCV);

	vector<float> metricasFinales;
	for (int j = 0; j < valoresPosibles; j++) {
		auto start2 = chrono::high_resolution_clock::now();
		for (int iter = 0; iter < iterationsCV; iter++) {
			cout << "HIPERPARAM: " << valores[j] << " ITERACION " << iter + 1 << endl;
			auto start3 = chrono::high_resolution_clock::now();

			// Redefino comienzo y fin de datos de evaluacion
			evalDataWindowIter = rf->crossValEvalDataWindow(iter, dataset.size() - (dataset.size() / 10), iterationsCV);

			dividedSlaveData = rf->divideDataForIteration(dataset, evalDataWindowGlobal.first, evalDataWindowGlobal.second, evalDataWindowIter.first, evalDataWindowIter.second);

			globalEvalData = dividedSlaveData.at("globalEval");
			iterEvalData = dividedSlaveData.at("iterEval");
			iterTrainData = dividedSlaveData.at("iterTrain");

			vector<struct Node*> randomForestMergeado;
			vector<vector<struct Node*>> threadsResult;

			int numeroHilos = thread::hardware_concurrency() - 1;

			if (usoHilos == 1) {
				vector<thread*> hilos;
				for (unsigned int i = 0; i < numeroHilos; i++) {
					if (indiceHiperparamChequear == 0)
						hilos.push_back(new thread(&RandomForest::randomForest, rf, ref(threadsResult), valores[j] / numeroHilos, instancesQty, attrQty, iterTrainData));
					if (indiceHiperparamChequear == 1)
						hilos.push_back(new thread(&RandomForest::randomForest, rf, ref(threadsResult), treesQty / numeroHilos, valores[j] , attrQty, iterTrainData));
					if (indiceHiperparamChequear == 2)
						hilos.push_back(new thread(&RandomForest::randomForest, rf, ref(threadsResult), treesQty / numeroHilos, instancesQty, valores[j], iterTrainData));
				}
				for (thread* hilo : hilos) {
					hilo->join();
				}
			}
			else {
				if (indiceHiperparamChequear == 0)
					rf->randomForest(threadsResult, valores[j], instancesQty, attrQty, iterTrainData);
				if (indiceHiperparamChequear == 1)
					rf->randomForest(threadsResult, treesQty, valores[j], attrQty, iterTrainData);
				if (indiceHiperparamChequear == 2)
					rf->randomForest(threadsResult, treesQty, instancesQty, valores[j], iterTrainData);
			}

			// Uno los rf de cada hilo al rf total
			for (vector<struct Node*> segmentoRf : threadsResult) {
				for (struct Node* arbol : segmentoRf) {
					randomForestMergeado.push_back(arbol);
				}
			}

			// Calculo evaluateRandomForest
			float valorMetricas = rf->evaluateRandomForest(randomForestMergeado, iterEvalData);
			cout << "  Iter metricas: " << valorMetricas * 100 << endl;
			metricasFinales.push_back(valorMetricas);
			auto stop3 = chrono::high_resolution_clock::now();
			auto duration3 = chrono::duration_cast<chrono::microseconds>(stop3 - start3);
			cout << "  Iter tiempo de computo: " << duration3.count() / 1000000 << "seg" << endl;
		}

		float aux = 0;
		for (float val : metricasFinales) {
			aux += val * 100;
		}
		auto stop2 = chrono::high_resolution_clock::now();
		auto duration2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);
		cout << "-------------------------------------------------------------------------------------------" << endl;
		cout << "=> Hiperparametro " << valores[j] << " dio: " << aux/10 << "% - Tiempo de computo: " << duration2.count()/1000000 << "seg"<< endl;
		cout << "-------------------------------------------------------------------------------------------" << endl;
		metricasFinales.clear();
	}
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
	cout << " =>FIN TIEMPO TOTAL: " << duration.count() / 1000000 << " seg"<< endl;
}
