#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <string>
#include <thread>
#include <future>

using namespace std;

struct Node {
	string data;
	struct Node* left;
	struct Node* right;

	// val is the key or the value that
	// has to be added to the data part
	Node(string val)
	{
		data = val;

		// Left and right child for node
		// will be initialized to null
		left = NULL;
		right = NULL;
	}
};

class RandomForest
{
public:
	int rows; 		//cantidad de instancias del conjunto de entrenamiento
	int evalSize; 	//cantidad de instancias del conjunto de evaluacion
	int attr;		//cantidad de atributos de cada instancia

	int partitionSizeCV;  //cantidad de instancias de cada particion para validacion cruzada
	int partitionsCV;		  //cantidad de particiones en validacion cruzada
	int iterationsCV;       //cantidad de iteraciones en validacion cruzada

	RandomForest(int rows, int evalSize, int attr, int partitionSizeCV, int partitionsCV, int iterationsCV);

	// FUNCIONES DUMMY, borrar luego
	void calculate(int* vector, int vectorSize);
	void printVector(int* vector, int vectorSize);

	// Inicializa dataset de evaluacion
	vector<vector<char>> initEvalDataset(vector<vector<char>> dataset, int evalDataBegin, int evalDataEnd);

	// Implementacion de algoritmo Random Forest 
	void randomForest(vector<vector<struct Node*>> &threadsResult, int treeQuantity, int instancesQuantity, int attrQuantity, vector<vector<char> > trainDataRandomCV);

	// Clasificacion de conjunto de evaluacion con arboles generados por Random Forest 
	void evaluate();

	// Retorna un par con el inicio y el fin de las instancias de evaluacion de una iteracion
	pair<int, int> crossValEvalDataWindow(int iter, int datasetSize, int iterationsMax);

	// Dado el dataset de cross validation, retorna para una iteracion, que datos usar de evaluacion y entreno
	// Retorna dos vector<char> con las matrices, cada fila separada por un '/'
	map<string, vector<vector<char>>> divideDataForIteration(vector<vector<char>> dataset,
		unsigned int restringedDataBegin, unsigned int restringedDataEnd,
		unsigned int evalDataBegin, unsigned int evalDataEnd);
	float evaluateRandomForest(vector<struct Node*> trees, vector<vector<char> > evalData);

private:
	void printBT1(const string& prefix, struct Node* node, bool isLeft);
	void printBT2(struct Node* node);
	bool checkArray(int value, vector<int> myArray);
	pair <int, int> countClasses(vector<vector<char> > trainDataset);
	pair <int, int> entropyDataset(vector<vector<char> > trainDataset);
	float entropy(pair <int, int>  params, float total);
	pair < vector<vector<char> >, vector<vector<char> >> amountExamples(int attr_idex, int totalExamples, vector<vector<char> > trainDataset);
	float gain(int attr_idex, float entropyValue, vector<vector<char> > trainDataset);
	vector<vector<char> > filter_data_set(int attr, char cvalue, vector<vector<char> > trainDataset);
	int deleteElement(int arr[], int n, int x);
	struct Node* id3(vector<int> myAttrs, vector<vector<char> > trainId3);
	int countPossitive(vector<char> tags);
	int countEquals(vector<char> realTags, vector<char> predictTags);
	int countPossitiveEquals(vector<char> realTags, vector<char> predictTags);
	int countNegativeDiff(vector<char> realTags, vector<char> predictTags);
	float getAccuracy(vector<char> realTags, vector<char> predictTags);
	float getPrecision(vector<char> realTags, vector<char> predictTags);
	float getRecall(vector<char> realTags, vector<char> predictTags);
	float getF1(vector<char> realTags, vector<char> predictTags);
	char classify(struct Node* tree, vector<char> example);
	char evaluate(vector<struct Node*> trees, vector<char> example);
};

