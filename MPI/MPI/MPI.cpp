// PARA EJECUTAR: mpiexec -np 2 MPI.exe

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
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
#include <future>

using namespace std;

// Parametros ------------------ 
int imprimirEsclavo; // Imprimir o no en escalvos
int imprimirMaestro; // Imprimir o no en escalvos
int rows;  //Cantidad de instancias del conjunto de entrenamiento
int separarGlobalData;  // Separar o no separar datos de evaluacion globales
int evalSize;  //Cantidad de instancias del conjunto de evaluacion
int attr;  //Cantidad de atributos de cada instancia
int partitionsCV;  //cantidad de particiones en validacion cruzada
int iterationsCV;  //cantidad de iteraciones en validacion cruzada
int usoHilos;  // Usar hilos en los esclavos o no
int treesQty; // Cantidad de arboles a usar (por defecto)
int instancesQty; // Cantidad de instancias a usar (por defecto)
int attrQty; // Cantidad de atributos a usar (por defecto)
int indiceHiperParam;  // Indice del hiperparametro a validar
int cantValoresHiperparam; // Cantidad de valores a probar para el hiperparametro a validar
vector<int> valoresHiperparam; // Los valores a probar del hiperparametro
// Parametros ------------------ 

vector<vector<char>> dataset;
vector<int> parametrosTotales;
pair<int, int> evalDataWindowGlobal;
pair<int, int> evalDataWindowIter;
vector<pair<int, int>> evalDataWindowIterVector;

// Slave variables
map<string, vector<vector<char>>> dividedSlaveData; //dataset que un esclavo levanta
vector<vector<char>> globalEvalData;
vector<vector<char>> iterEvalData;
vector<vector<char>> iterTrainData;

void inicializarVariables(vector<int> parametros) { 
	imprimirEsclavo = parametros[0];
	imprimirMaestro = parametros[1];
	rows = parametros[2];
	separarGlobalData = parametros[3];
	evalSize = parametros[4];
	attr = parametros[5];
	iterationsCV = parametros[6];
	partitionsCV = parametros[7];
	usoHilos = parametros[8];
	treesQty = parametros[9];
	instancesQty = parametros[10];
	attrQty = parametros[11];
	indiceHiperParam = parametros[12];
	cantValoresHiperparam = parametros[13];
	for (int index = 0; index < cantValoresHiperparam; index++) {
		valoresHiperparam.push_back(parametros[index + 14]);
	}
}

int main(int argc, char* argv[]) {
	FileLoader* fl = new FileLoader();

	// Leo el archivos: dataset y parametros
	dataset = fl->readFile("dataset.csv");
	parametrosTotales = fl->readParameters("parameters.txt");
	
	// Inicializo los parametros en las variables
	inicializarVariables(parametrosTotales);

	RandomForest* rf = new RandomForest(rows, evalSize, attr, separarGlobalData == 1 ? (rows - rows/partitionsCV) / partitionsCV : rows / partitionsCV, partitionsCV, iterationsCV);

	const int masterId = 0;
	int myid, tag, numprocs;
	MPI_Status status;
	MPI_Request request;
	tag = 0;
	request = MPI_REQUEST_NULL;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// ======================================================= MAESTRO =======================================================
	if (myid == masterId) { // Maestro manda arrays para calcular a los esclavos y los recive calculados
		fl->printVectorInt("=> Parametros Leidos: ", parametrosTotales);
		fl->printVectorInt("Parametros a usar: ", valoresHiperparam);

		auto start = chrono::high_resolution_clock::now();

		// Determino el inicio y el fin de los datos finales de evaluacion (en este caso es el ultimo 10% de las instancias)
		if (separarGlobalData) {
			evalDataWindowGlobal = rf->crossValEvalDataWindow(iterationsCV - 1, dataset.size(), partitionsCV);
		}
		else {
			evalDataWindowGlobal = make_pair(-1,-1);
		}

		// Defino inicio y fin de datos de evaluacion de cada iteracion y los guardo en vector
		int cantidadDatosRestringidos = separarGlobalData ? dataset.size() / partitionsCV  : 0;
		for (int iter = 0; iter < iterationsCV; iter++) {
			evalDataWindowIter = rf->crossValEvalDataWindow(iter, dataset.size() - cantidadDatosRestringidos, partitionsCV);
			evalDataWindowIterVector.push_back(evalDataWindowIter);
		}

		bool fin = false;

		// Comienzo Inicializacion ----------------------------------------------------------------------------------
	
		// Inicializo vector de vectores que para cada valor de hiperparametros, marca las iteraciones completadas
		vector<vector<bool>> iteracionesTotales(cantValoresHiperparam, vector<bool>(iterationsCV, false));

		// Inicializo los esclavos disponibles como libres
		vector<bool> esclavos(numprocs-1, true);

		// Inicializo vector de resultados finales de cada iteracion
		vector<vector<int>> resultadosFinales(cantValoresHiperparam, vector<int>(iterationsCV, -1));

		// Fin Inicializacion ----------------------------------------------------------------------------------

		do {
			int esclavoLibre = fl->getIndexFreeSlave(esclavos);
			// Busco que iteracion queda por hacer => <valorHiperparam, numIteracion>
			pair<int, int> iterPendiente = fl->findIterationToMake(iteracionesTotales);

			// Si hay esclavo libre y tengo iteracion que mandar
			if (esclavoLibre != -1 && iterPendiente.first != -1 && iterPendiente.second != -1) {
				esclavos[esclavoLibre] = false;
				iteracionesTotales[iterPendiente.first][iterPendiente.second] = true;

				// Defino datos a enviar al esclavo
				int selected_slave = esclavoLibre + 1;
				int metaParameters[11];
				metaParameters[0] = evalDataWindowGlobal.first; // Indice comienzo de datos de evaluacion globales
				metaParameters[1] = evalDataWindowGlobal.second; // Indice fin de datos de evaluacion globales
				metaParameters[2] = evalDataWindowIterVector[iterPendiente.second].first; // Indice comienzo de datos de evaluacion de la iteracion
				metaParameters[3] = evalDataWindowIterVector[iterPendiente.second].second; // Indice fin de datos de evaluacion de la iteracion
				metaParameters[4] = indiceHiperParam == 0 ? valoresHiperparam[iterPendiente.first] : treesQty; // Hiperparam1: cantidad de arboles
				metaParameters[5] = indiceHiperParam == 1 ? valoresHiperparam[iterPendiente.first] : instancesQty; // Hiperparam2: cantidad de instancias
				metaParameters[6] = indiceHiperParam == 2 ? valoresHiperparam[iterPendiente.first] : attrQty; // Hiperparam3: cantidad de atributos
				metaParameters[7] = usoHilos; // Usar o no hilos en esclavo
				metaParameters[8] = 0; // Le dice al esclavo si debe terminar su ejecucion
				metaParameters[9] = iterPendiente.first; // Le dice al esclavo en que hiperparametro se esta trabajando
				metaParameters[10] = iterPendiente.second; // Le dice al esclavo en que iteracion se esta trabajando
				if (imprimirMaestro == 1)
					cout << "-> MASTER: Mando al slave " << selected_slave << ", el vector: ";

				// Envio datos al esclavo
				MPI_Send(&metaParameters[0], 11, MPI_INT, selected_slave, 0, MPI_COMM_WORLD);
			}

			// Termine mandar todas las iteraciones
			if (iterPendiente.first == -1 && iterPendiente.second == -1)
				fin = true;
			
			// Si No hay esclavos libres -> Espero recibir
			if (fl->getIndexFreeSlave(esclavos) == -1) { 
				int valoresRecibidos[3];
				MPI_Recv(&valoresRecibidos, 3, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
				if (imprimirMaestro == 1)
					fl->printMatrixInt(" --- Matriz de iteracion ---", iteracionesTotales);
				resultadosFinales[valoresRecibidos[1]][valoresRecibidos[2]] = valoresRecibidos[0];
				esclavos[status.MPI_SOURCE - 1] = true;
				if (imprimirMaestro == 1)
					cout << "-> MASTER: Recibo del slave: " << status.MPI_SOURCE << ", el valor: " << valoresRecibidos[0] << ", HP: " << valoresRecibidos[1] << ", Iter: "<< valoresRecibidos[2] << endl;
			}

			// No quedan mas iteraciones por mandar pero falta recibir calculos
			if (fin && fl->getIndexWorkingSlave(esclavos) != -1) {
				int valoresRecibidos[3];
				MPI_Recv(&valoresRecibidos, 3, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
				resultadosFinales[valoresRecibidos[1]][valoresRecibidos[2]] = valoresRecibidos[0];
				esclavos[status.MPI_SOURCE - 1] = true;
				if (imprimirMaestro == 1) 
					cout << "-> MASTER: Recibo del slave: " << status.MPI_SOURCE << ", el valor: " << valoresRecibidos[0] << ", HP: " << valoresRecibidos[1] << ", Iter: " << valoresRecibidos[2] << endl;
			}
		} // Sigo mientras no haya terminado de mandar las iteraciones o si termine de mandarlas pero quedan esclavos trabajando
		while (!fin || (fin && fl->getIndexWorkingSlave(esclavos) != -1));

		// Mando apagar todos los esclavos
		for (int i = 1; i < numprocs; i++) {
			int metaParameters[11] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
			MPI_Send(&metaParameters[0], 11, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
		cout << endl;
		cout << ">> TIEMPO TOTAL: " << duration.count() / 1000000 << " seg" << endl;
		string nombreResultados = fl->writeFile(indiceHiperParam, resultadosFinales, valoresHiperparam, iterationsCV, duration.count() / 1000000);
		cout << "Ver resultados en el archivo generado de nombre: " << nombreResultados << endl;
		cout << "-->End Master " << endl;
	}
	// ======================================================= MAESTRO =======================================================

	// ======================================================= ESCLAVO =======================================================
	if (myid != masterId) { // Esclavos reciben array, recalculan y lo envian al maestro denuevo

		int endWhile = 0;

		auto start2 = chrono::high_resolution_clock::now();

		do {
			int parameters[11];

			// Recibo datos de evaluacion y su largo
			MPI_Recv(&parameters[0], 11, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
			endWhile = parameters[8];
		
			if (endWhile != 1) {
				//Defino dataset de evaluacion Global
				dividedSlaveData = rf->divideDataForIteration(dataset, parameters[0], parameters[1], parameters[2], parameters[3]);
		
				globalEvalData = dividedSlaveData.at("globalEval");
				iterEvalData = dividedSlaveData.at("iterEval");
				iterTrainData = dividedSlaveData.at("iterTrain");

				vector<struct Node*> randomForestMergeado;
				vector<vector<struct Node*>> threadsResult;
				
				// Calculo random forest
				if (parameters[7] == 1) { // USO HILOS
					int numeroHilos = thread::hardware_concurrency() - 1;
					vector<thread*> hilos;
				
					for (unsigned int i = 0; i < numeroHilos; i++) {
						hilos.push_back(new thread(&RandomForest::randomForest, rf, ref(threadsResult), parameters[4] / numeroHilos, parameters[5], parameters[6], iterTrainData));
					}
					for (thread* hilo : hilos) {
						hilo->join();
					}
				}
				else { // NO USO HILOS
					rf->randomForest(threadsResult, parameters[4], parameters[5], parameters[6], iterTrainData);
				}

				// Uno los rf de cada hilo al rf total
				for (vector<struct Node*> segmentoRf : threadsResult) {
					for (struct Node* arbol : segmentoRf) {
						randomForestMergeado.push_back(arbol);
					}
				}

				// Calculo evaluateRandomForest
				float valorMetricas = rf->evaluateRandomForest(randomForestMergeado, iterEvalData);

				auto stop2 = chrono::high_resolution_clock::now();
				auto duration2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);

				if (imprimirEsclavo == 1) {
					cout << "* Esclavo: " << myid << " => Vector Recibido: "
						<< parameters[0] << " " << parameters[1] << " "
						<< parameters[2] << " " << parameters[3] << " "
						<< parameters[4] << " " << parameters[5] << " "
						<< parameters[6] << " " << parameters[7] << " " 
						<< parameters[8] << " " << parameters[9] << 
						" => HP: " << parameters[9] << ", Iter: " << parameters[10] <<
						" => Respondo con: " << parameters[3] / 2 << ". Tiempo: " << duration2.count() / 1000000 << "seg" << endl;
					cout << endl;
				}

				valorMetricas = valorMetricas * 100;
				int res[3];
				res[0] = (int) valorMetricas;
				res[1] = parameters[9];
				res[2] = parameters[10];

				// Envio evaluacion de iteracion al maestro
				MPI_Send(&res, 3, MPI_INT, masterId, tag, MPI_COMM_WORLD);
			}
		} while (endWhile != 1);
		cout << "-->End slave " << myid << endl;
	}
	// ======================================================= ESCLAVO =======================================================

	MPI_Finalize();
}