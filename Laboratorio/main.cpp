// PARA EJECUTAR: mpiexec -np 2 MPI.exe
//	pair < vector<char>, vector<char>> res = make_pair(var1, var2);
//res.first
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

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

int hoja = 0;
int nodito = 0;

const int rows = 8090; 		//cantidad de instancias del conjunto de entrenamiento
const int evalSize = 899; 	//cantidad de instancias del conjunto de evaluacion
const int attr = 1025;		//cantidad de atributos de cada instancia

const int partitionSizeCV = 809;  //cantidad de instancias de cada particion para validacion cruzada
const int partitionsCV= 10;		  //cantidad de particiones en validacion cruzada
const int iterationsCV= 10;       //cantidad de iteraciones en validacion cruzada

vector<vector<char> > dataset;     //dataset qsar
vector<vector<char> > evalDataset; //dataset de evaluacion

vector<vector<vector<char> > > crossValidationDataset; //dataset para validacion cruzada

vector<vector<char> > evalDataCV;   //particion de evaluacion en validacion cruzada
vector<vector<char> > trainDataCV;  //conjunto de entrenamiento en validacion cruzada

//vector<vector<char> > trainCV;      //conjunto de entrenamiento utilizado en cada uno de los arboles de random forest


void printBT(const string& prefix, struct Node* node, bool isLeft)
{
    if( node != nullptr )
    {
        cout << prefix;

        cout << (isLeft ? "├──" : "└──" );

        // print the value of the node
        cout << node->data << endl;

        // enter the next tree level - left and right branch
        printBT( prefix + (isLeft ? "│   " : "    "), node->left, true);
        printBT( prefix + (isLeft ? "│   " : "    "), node->right, false);
    }
}

void printBT(struct Node* node)
{
    printBT("", node, false);    
}


// Lee el archivo y lo parsea para cargarlo a alguna matriz
void readFile(const char* fileName) {
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

	//cout << "Cantidad de instancias cargadas: " << dataset.size() << endl;
	//cout << "Cantidad de atributos por instancia: " << dataset[0].size() << endl;

}

bool checkArray(int value, vector<int> myArray) {
	std::vector<int>::iterator it;
	bool res = false;
	it = find(myArray.begin(), myArray.end(), value);
	if (it != myArray.end())
		res = true;
	return res;
}

// Inicializa dataset de evaluacion
void initEvalDataset() { 
    for (int i = 0; i < evalSize; i++) {
    	vector<char> row = dataset[i + rows];
		evalDataset.push_back(row);
    }

    //cout << "Cantidad de instancias conjunto de evaluacion: " << evalDataset.size() << endl;
}

pair <int, int> countClasses(vector<vector<char> > trainDataset) {
    int positives = 0;
    int negatives = 0;

	for (int i = 0; i < trainDataset.size(); i++) {
		vector<char> example = trainDataset[i];
		if (example[1024] == 'p') {
			positives += 1;
		}
		else if (example[1024] == 'n') {
  			negatives += 1;
		}
	}

	pair <int, int>  result = make_pair(positives, negatives);
    return result;
}

pair <int, int> entropyDataset(vector<vector<char> > trainDataset) {
    int positiveValues = 0;
    int negativeValues = 0;

	for (int i = 0; i < trainDataset.size(); i++) {
		vector<char> example = trainDataset[i];
        if (example[1024] == 'p'){
			positiveValues+=1;
		}
		else{
			negativeValues+=1;
		}
	}

	pair <int, int> result = make_pair(positiveValues, negativeValues);
	return result;
}

float entropy(pair <int, int>  params, float total) {
	float positiveResult = 0;
    float negativeResult = 0;
    float positiveCoefficient = 0;
    float negativeCoefficient = 0;
	float positiveValues = params.first;
	float negativeValues = params.second;

    if (total > 0) {
  		positiveCoefficient = positiveValues/total;
        negativeCoefficient = negativeValues/total;
	}
    if (positiveCoefficient != 0) {
        positiveResult = ((-1)*positiveCoefficient)*(log2(positiveCoefficient));
	}
    if (negativeCoefficient != 0) {
        negativeResult = ((-1)*negativeCoefficient)*(log2(negativeCoefficient));
	}
    return (positiveResult + negativeResult);    
}

pair < vector<vector<char> >, vector<vector<char> >> amountExamples(int attr_idex, int totalExamples, vector<vector<char> > trainDataset) {
	vector<vector<char> > examplesWithZero;
	vector<vector<char> > examplesWithOne;
	for (int j = 0; j < totalExamples; j++) {
		vector<char> row = trainDataset[j];

		if ((row[attr_idex]) == '0'){
			examplesWithZero.push_back(row);
		}
		else if ((row[attr_idex]) == '1'){
			examplesWithOne.push_back(row);
		}
	}
	pair < vector<vector<char> >, vector<vector<char> >> result = make_pair(examplesWithZero, examplesWithOne);
	return result;
}

float gain(int attr_idex, float entropyValue, vector<vector<char> > trainDataset) {
	float totalExamples = trainDataset.size();
    pair < vector<vector<char> >, vector<vector<char> >> examples = amountExamples(attr_idex, totalExamples, trainDataset);
    float examplesZero = examples.first.size();
	float examplesOne = examples.second.size();
	float S0 = (examplesZero/totalExamples);
    pair < int, int> resultZero = entropyDataset(examples.first); //falta pasar por param examples["examplesWithZero"]

    float entropyS0 = entropy(resultZero, examplesZero);

    float S1 = (examplesOne/totalExamples);
    pair < int, int> resultOne = entropyDataset(examples.second); //falta pasar por param examples["examplesWithOne"]
    float entropyS1 = entropy(resultOne, examplesOne);

    float gainValue = entropyValue - ((S0*entropyS0)+((S1)*entropyS1));
    return gainValue;
}


vector<vector<char> > filter_data_set(int attr, char cvalue, vector<vector<char> > trainDataset) {
    vector<vector<char> > resultDataset;
	for (int i = 0; i < trainDataset.size(); i++) {
		vector<char> row = trainDataset[i];
		if (row[i] == cvalue) {
			resultDataset.push_back(row);
		}
	}
    return resultDataset;
}


// This function removes an element x from arr[] and
// returns new size after removal (size is reduced only
// when x is present in arr[]
int deleteElement(int arr[], int n, int x)
{
	// Search x in array
	int i;
	for (i=0; i<n; i++)
		if (arr[i] == x)
			break;
	
	// If x found in array
	if (i < n)
	{
		// reduce size of array and move all
		// elements on space ahead
		n = n - 1;
		for (int j=i; j<n; j++)
			arr[j] = arr[j+1];
	}
	
	return n;
}

// Implementacion de algoritmo ID3 para construir arboles de decision
struct Node* id3(vector<int> myAttrs, vector<vector<char> > trainId3) {
	int attrsSize = myAttrs.size();
	int gainZero = 1; //definir gainZero como param para podar arbol
	pair <int, int> classes = countClasses(trainId3);
	struct Node* root;
    if (classes.first == trainId3.size()) {
		root = new Node("p");
	}
    else if (classes.second == trainId3.size()) {
		root = new Node("n");
	}
    else if (attrsSize == 0){
        if (classes.first >= classes.second){
			root = new Node("p");
		}
        else{
			root = new Node("n");
		}
	}
    else{
		pair <int, int> result = entropyDataset(trainId3);
		float entropyValue = entropy(result, trainId3.size());
        int bestAttr = myAttrs[0];
        float bestGain = gain(bestAttr, entropyValue, trainId3);
		for(int k = 0; k < attrsSize; ++k) {
			int attr = myAttrs[k];
		 	float attrGain = gain(attr, entropyValue, trainId3);
            if (attrGain > bestGain) {
				bestGain = attrGain;
				bestAttr = attr;
			}
		}
		//Corto recursión, atributos ya no me aportan info          
        if((gainZero == 1) && (bestGain == 0)) {
			string value;
            if (classes.first >= classes.second) {
                value = "p";
			}
            else {
 				value = "n";
			}
			root = new Node(value);
		}
        else {
			string bestAttrChar = to_string(bestAttr);
			root = new Node(bestAttrChar);
			vector<vector<char> > negativeDataset = filter_data_set(bestAttr, '0', trainId3);
            vector<vector<char> > positiveDataset = filter_data_set(bestAttr, '1', trainId3);
		
            if (negativeDataset.size() == 0) {
				root->left = new Node("p");
			}
            else{

				vector<int> copyAttrsLeft;
				copyAttrsLeft.clear();
				copyAttrsLeft = myAttrs;
				std::vector<int>::iterator it;
				int bestAttrIndex = 0;
				it = find(myAttrs.begin(), myAttrs.end(), bestAttr);
				if (it != myAttrs.end())
					bestAttrIndex = it - myAttrs.begin();
				copyAttrsLeft.erase(copyAttrsLeft.begin() + bestAttrIndex);
			
				root->left = id3(copyAttrsLeft, negativeDataset);
			}

           	if (positiveDataset.size() == 0) {
				root->right = new Node("n");
			}
            else{

				vector<int> copyAttrsRigth;
				copyAttrsRigth.clear();
				copyAttrsRigth = myAttrs;

				std::vector<int>::iterator it;
				int bestAttrIndex = 0;
				it = find(myAttrs.begin(), myAttrs.end(), bestAttr);
				if (it != myAttrs.end())
					bestAttrIndex = it - myAttrs.begin();
				copyAttrsRigth.erase(copyAttrsRigth.begin() + bestAttrIndex);
	
				root->right = id3(copyAttrsRigth, positiveDataset);


			}
		}           
	}
	//printf("arbol parcial:\n");
	//printBT(root);
	//cout << "arbol parcial id3: " << node.dump() << endl;
	return root;
}


// Implementacion de algoritmo Random Forest 
vector<struct Node*> randomForest(int treeQuantity, int instancesQuantity, int attrQuantity, vector<vector<char> > trainDataRandomCV) {
	vector<struct Node*> arboles;	
	arboles.clear();
	printf("random forest..\n");
	vector<vector<char> > trainCV = trainDataRandomCV;
	//Ejecutar algoritmo ID3 treeQuantity veces, generando treeQuantity arboles distintos
	for (int i = 0; i < treeQuantity; i++){
		
		//Generar el conjunto de entrenamiento trainCV tomando instancesQuantity elementos de trainDataCV con repeticion 
		//for (int j = 0; j < instancesQuantity; j++){
		//	int index = rand() % (partitionSizeCV*9); //tomo una instancia aleatoria entre los indices 0 y (partitionSizeCV*9).
		//	trainCV.push_back(trainDataRandomCV[index]);
		//}

		//Generar el conjunto de atributos a utilizar, tomando attrQuantity de los posibles (attrs-1)
		vector<int> selectedAttrs;
		for (int k = 0; k < attrQuantity; ++k)
		{
			int r;
			unsigned int n;
			do
			{
				r = rand() % (attr - 1);
				n = sizeof(selectedAttrs) / sizeof(int);

			} while (checkArray(r, selectedAttrs));
			selectedAttrs.push_back(r);
		}

		struct Node* resultID3 = id3(selectedAttrs, trainCV);
		//printf("resultado arbol id3:\n");
		//printBT(resultID3);
		arboles.push_back(resultID3);
		//trainCV.clear();
	}
	return arboles;
}

int countPossitive(vector<char> tags) {
	int count = 0;
	for(int i = 0; i < tags.size(); ++i){
		if (tags[i] == 'p') {
			count+=1;
		}
	}

	return count;
}

int countEquals(vector<char> realTags, vector<char> predictTags) {
	int count = 0;
	for(int i = 0; i < realTags.size(); ++i){
		if (realTags[i] == predictTags[i]) {
			count+=1;
		}
	}
	return count;
}

int countPossitiveEquals(vector<char> realTags, vector<char> predictTags) {
	int count = 0;
	for(int i = 0; i < realTags.size(); ++i){
		if (realTags[i] == predictTags[i] && predictTags[i] == 'p') {
			count+=1;
		}
	}

	return count;
}

int countNegativeDiff(vector<char> realTags, vector<char> predictTags) {
	int count = 0;
	for(int i = 0; i < realTags.size(); ++i){
		if (realTags[i] != predictTags[i] && predictTags[i] == 'n') {
			count+=1;
		}
	}

	return count;
}

float getAccuracy(vector<char> realTags, vector<char> predictTags){
	float equals = countEquals(realTags, predictTags);
	float base = realTags.size();
	float acc = equals / base;
	return acc;
}

float getPrecision(vector<char> realTags, vector<char> predictTags){
	float pre = 0;
	float equals = countPossitiveEquals(realTags, predictTags); // verdaderos positivos
	float base = countPossitive(predictTags);
	if (base != 0) {
		pre = equals / base;
	}
	return pre;
}

float getRecall(vector<char> realTags, vector<char> predictTags){
	float rec = 0;
	float equals = countPossitiveEquals(realTags, predictTags); // verdaderos positivos
	float diff = countNegativeDiff(realTags, predictTags);  // falsos negativos
	if (diff != 0) {
		rec = equals / (equals + diff);
	}
	return rec;
}

float getF1(vector<char> realTags, vector<char> predictTags){
	float f1 = 0;
	float pre = getPrecision(realTags, predictTags);
	float rec = getRecall(realTags, predictTags);
	if (pre != 0 || rec != 0) {
		f1 = (2 * pre * rec) / (pre + rec);
	}
	return f1;
}


char classify(struct Node* tree, vector<char> example) {
	string currentNode = tree->data;
	if (currentNode == "n") {
		return 'n';
	}
	if (currentNode == "p") {
		return 'p';
	}
	int indexAttr = stoi(currentNode);
	char attrValue = example[indexAttr];
	if (attrValue == '0') {
		return classify(tree->left, example);
	}
	else {
		return classify(tree->right, example);
	}
}

char evaluate(vector<struct Node*> trees, vector<char> example) {
	int positives = 0;
    int negatives = 0;
    char classification;
	for(int i = 0; i < trees.size(); ++i) {
		struct Node* tree = trees[i];
		classification = classify(tree, example);
	
		if (classification == 'n') {
			negatives+=1;
		}
		else {
			positives+=1;
		}
	} 
	if (negatives < positives) {
		return 'p';
	}
	else {
		return 'n';
	}
}



// Clasificacion de conjunto de evaluacion con arboles generados por Random Forest 
void evaluateRandomForest(vector<struct Node*> trees, vector<vector<char> > evalData) {
	vector<char> realTags;
	vector<char> predictTags;
	int positivos = 0;
	int positivosClassify = 0;
	
	for(int i = 0; i < evalData.size(); ++i){ 
		vector<char> example = evalData[i];
		char predict = evaluate(trees, example);
		realTags.push_back(example[1024]);

		if (predict == 'p') {
			positivosClassify+=1;
		}

		if (example[1024] == 'p') {
			positivos+=1;
		}
		predictTags.push_back(predict);
	}

	float accuracy = getAccuracy(realTags, predictTags);
	float precision = getPrecision(realTags, predictTags);
	float recall = getRecall(realTags, predictTags);
	float f1 = getF1(realTags, predictTags);

	cout << "% Accuracy: " << (accuracy * 100) << endl;
	cout << "% Precision: " << (precision * 100) << endl;
	cout << "% Recall: " << (recall * 100) << endl;
	cout << "% F1 Score: " << (f1 * 100) << endl;


}

// Implementacion de algoritmo de validacion cruzada 
void crossValidation() { 

    //init cross validation partitions
	for (int x = 0; x < partitionsCV; x++)
    {
    	int coef = partitionSizeCV*x; 
    	vector<vector<char> > crossValidationAux;
		for (int i = 0; i < partitionSizeCV; i++)
	    {
	    	vector<char> row = dataset[i + coef];
	    	crossValidationAux.push_back(row);
	    }
	    crossValidationDataset.push_back(crossValidationAux);
 	}
 	//cout << "Cantidad de matrices almacenadas en crossValidationDataset: " << crossValidationDataset.size() << endl;
    
 	//execute cross validation
 	for (int iter = 0; iter < iterationsCV; iter++)
    {
    	printf("---Iteracion %d validacion cruzada---\n", iter + 1);
    	for (int x = 0; x < partitionsCV; x++)
    	{

			//define eval dataset with 1 partition for each iteration
    		if (iter == x) {
    			evalDataCV = crossValidationDataset[x];
				//cout << "Cantidad de instancias dataset de evaluacion validacion cruzada: " << evalDataCV.size() << endl;
    		}
    		//define train dataset with (partitions - 1) partitions for each iteration
	    	else{

	    		for (int i = 0; i < partitionSizeCV; i++)
			    {
			    	trainDataCV.push_back(crossValidationDataset[x][i]);
			    }
	
	    	}

		}
		//cout << "Cantidad de instancias dataset de entrenamiento validacion cruzada: " << trainDataCV.size() << endl;

		//call random forest with trainDataCV 
		int treeQuantity = 2;  					   //este valor lo queremos "validar", puede ser 80, 90, 100, 110, etc.
		//int attrQuantity = (attr-1) - ((attr-1)/3);    //este valor lo queremos "validar", puede ser (attr-1), (attr-1)/2, (attr-1)/3, etc.
		int attrQuantity = 32;
		int instancesQuantity = ((partitionSizeCV*9)/3); //este valor lo queremos "validar", puede ser (partitionSizeCV), (partitionSizeCV)/2, etc.
		vector<struct Node*> resultTrees = randomForest(treeQuantity, instancesQuantity, attrQuantity, trainDataCV);
		//evaluate evalDataCV with random forest tree result
		evaluateRandomForest(resultTrees, evalDataCV);
		//clean vectors for next iteration
		resultTrees.clear();
		evalDataCV.clear();
		trainDataCV.clear();
	}


}

int main(int argc, char* argv[]) {

	//load dataset
    readFile("qsar.csv");

	auto rng = std::default_random_engine {};
	std::shuffle(std::begin(dataset), std::end(dataset), rng);
	//init evaluation dataset
	initEvalDataset();

	//cross validation "k = 10"
	crossValidation();

}
