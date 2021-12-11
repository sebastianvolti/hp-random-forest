#include "RandomForest.h"

RandomForest::RandomForest(int rows, int evalSize, int attr, int partitionSizeCV, int partitionsCV, int iterationsCV) {
	this->rows = rows;
	this->evalSize = evalSize;
	this->attr = attr;
	this->partitionSizeCV = partitionSizeCV;
	this->partitionsCV = partitionsCV;
	this->iterationsCV = iterationsCV;
}

void RandomForest::calculate(int* vector, int vectorSize) {
	for (int i = 0; i < vectorSize; i++) {
		vector[i] = 3 * vector[i];
	}
}

void RandomForest::printVector(int* vector, int vectorSize) {
	for (int i = 0; i < vectorSize; i++) printf("%4d", vector[i]); printf("\n");
}

vector<vector<char>> RandomForest::initEvalDataset(vector<vector<char>> dataset, int evalDataBegin, int evalDataEnd) {
	vector<vector<char>> evalDataset;
	for (int i = evalDataBegin; i < evalDataEnd; i++) {
		vector<char> row = dataset[i];
		evalDataset.push_back(row);
	}
	return evalDataset;
}

void RandomForest::evaluate() {
	printf("evaluacion post random forest\n");
}

pair<int, int> RandomForest::crossValEvalDataWindow(int iter, int datasetSize, int iterationsMax) {
	pair<int, int> dataBeginEnd;
	int totalIterInstances = datasetSize / iterationsMax;
	return make_pair(iter * totalIterInstances, (iter * totalIterInstances) + totalIterInstances);
}

map<string, vector<vector<char>>> RandomForest::divideDataForIteration(vector<vector<char>> dataset,
	unsigned int restringedDataBegin, unsigned int restringedDataEnd,
	unsigned int evalDataBegin, unsigned int evalDataEnd) {

	vector<vector<char>> globalEvalDataCV;
	vector<vector<char>> evalDataCV;   //particion de evaluacion en validacion cruzada
	vector<vector<char>> trainDataCV;  //conjunto de entrenamiento en calidacion cruzada

	for (unsigned int i = 0; i < dataset.size(); i++) {
		if (i >= evalDataBegin && i < evalDataEnd) {
			evalDataCV.push_back(dataset[i]);
		}
		else if (i >= restringedDataBegin && i < restringedDataEnd) {
			globalEvalDataCV.push_back(dataset[i]);
		}
		else {
			trainDataCV.push_back(dataset[i]);
		}
	}

	map<string, vector<vector<char>>> res;
	res.emplace("globalEval", globalEvalDataCV);
	res.emplace("iterEval", evalDataCV);
	res.emplace("iterTrain", trainDataCV);
	return res;
}

// Implementacion de algoritmo Random Forest 
void RandomForest::randomForest(vector<vector<struct Node*>>& threadsResult, int treeQuantity, int instancesQuantity, int attrQuantity, vector<vector<char> > trainDataRandomCV) {
	vector<vector<char> > trainCV = trainDataRandomCV;

	/*
	printf("tree qty %d\n", treeQuantity);
	printf("instances qty %d\n", instancesQuantity);
	printf("attr qty %d\n", attrQuantity);
	printf("train data qty %d\n", trainDataRandomCV.size());
	printf("partitions %d\n", partitionSizeCV);
	*/

	vector<struct Node*> res;

	//Ejecutar algoritmo ID3 treeQuantity veces, generando treeQuantity arboles distintos
	for (int i = 0; i < treeQuantity; i++) {
		//Generar el conjunto de entrenamiento trainCV tomando instancesQuantity elementos de trainDataCV con repeticion 
		//for (int j = 0; j < instancesQuantity; j++) {
		//	int index = rand() % (partitionSizeCV * 9); //tomo una instancia aleatoria entre los indices 0 y (partitionSizeCV*9).
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

		res.push_back(resultID3);
	}
	threadsResult.push_back(res);
}

// Clasificacion de conjunto de evaluacion con arboles generados por Random Forest 
float RandomForest::evaluateRandomForest(vector<struct Node*> trees, vector<vector<char> > evalData) {
	// printf("---Evaluacion post random forest\n");
	vector<char> realTags;
	vector<char> predictTags;

	for (int i = 0; i < evalData.size(); ++i) {
		vector<char> example = evalData[i];
		char predict = evaluate(trees, example);
		realTags.push_back(example[1024]);
		predictTags.push_back(predict);
	}

	float accuracy = getAccuracy(realTags, predictTags);
	float precision = getPrecision(realTags, predictTags);
	float recall = getRecall(realTags, predictTags);
	float f1 = getF1(realTags, predictTags);

	//cout << "% Accuracy: " << (accuracy * 100) << endl;
	//cout << "% Precision: " << (precision * 100) << endl;
	//cout << "% Recall: " << (recall * 100) << endl;
	//cout << "    % F1 Score: " << (f1 * 100) << endl;
	return f1;
}

// ------------------------------------------------------------ FUNCIONES AUXILIARES ------------------------------------------------------------

int RandomForest::countPossitive(vector<char> tags) {
	int count = 0;
	for (int i = 0; i < tags.size(); ++i) {
		if (tags[i] == 'p') {
			count += 1;
		}
	}
	return count;
}

int RandomForest::countEquals(vector<char> realTags, vector<char> predictTags) {
	int count = 0;
	for (int i = 0; i < realTags.size(); ++i) {
		if (realTags[i] == predictTags[i]) {
			count += 1;
		}
	}
	return count;
}

int RandomForest::countPossitiveEquals(vector<char> realTags, vector<char> predictTags) {
	int count = 0;
	for (int i = 0; i < realTags.size(); ++i) {
		if (predictTags[i] == 'p' && realTags[i] == predictTags[i]) {
			count += 1;
		}
	}
	return count;
}

int RandomForest::countNegativeDiff(vector<char> realTags, vector<char> predictTags) {
	int count = 0;
	for (int i = 0; i < realTags.size(); ++i) {
		if (predictTags[i] == 'n' && realTags[i] != predictTags[i]) {
			count += 1;
		}
	}
	return count;
}

float RandomForest::getAccuracy(vector<char> realTags, vector<char> predictTags) {
	float equals = countEquals(realTags, predictTags);
	float base = realTags.size();
	float acc = equals / base;
	return acc;
}

float RandomForest::getPrecision(vector<char> realTags, vector<char> predictTags) {
	float pre = 0;
	float equals = countPossitiveEquals(realTags, predictTags); // verdaderos positivos
	float base = countPossitive(predictTags);
	if (base != 0) {
		pre = equals / base;
	}
	return pre;
}

float RandomForest::getRecall(vector<char> realTags, vector<char> predictTags) {
	float rec = 0;
	float equals = countPossitiveEquals(realTags, predictTags); // verdaderos positivos
	float diff = countNegativeDiff(realTags, predictTags);  // falsos negativos
	if (diff != 0) {
		rec = equals / (equals + diff);
	}
	return rec;
}

float RandomForest::getF1(vector<char> realTags, vector<char> predictTags) {
	float f1 = 0;
	float pre = getPrecision(realTags, predictTags);
	float rec = getRecall(realTags, predictTags);
	if (pre != 0 || rec != 0) {
		f1 = (2 * pre * rec) / (pre + rec);
	}
	return f1;
}


char RandomForest::classify(struct Node* tree, vector<char> example) {
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

char RandomForest::evaluate(vector<struct Node*> trees, vector<char> example) {
	int positives = 0;
	int negatives = 0;
	char classification;
	for (int i = 0; i < trees.size(); ++i) {
		struct Node* tree = trees[i];
		classification = classify(tree, example);
		if (classification == 'n') {
			negatives += 1;
		}
		else {
			positives += 1;
		}
	}
	if (negatives < positives) {
		return 'p';
	}
	else {
		return 'n';
	}
}

void RandomForest::printBT1(const string& prefix, struct Node* node, bool isLeft)
{
	if (node != nullptr)
	{
		cout << prefix;

		cout << (isLeft ? "???" : "???");

		// print the value of the node
		cout << node->data << endl;

		// enter the next tree level - left and right branch
		printBT1(prefix + (isLeft ? "?   " : "    "), node->left, true);
		printBT1(prefix + (isLeft ? "?   " : "    "), node->right, false);
	}
}

void RandomForest::printBT2(struct Node* node)
{
	printf("Print..\n");
	printBT1("", node, false);
}

bool RandomForest::checkArray(int value, vector<int> myArray) {
	std::vector<int>::iterator it;
	bool res = false;
	it = find(myArray.begin(), myArray.end(), value);
	if (it != myArray.end())
		res = true;
	return res;
}

pair <int, int> RandomForest::countClasses(vector<vector<char> > trainDataset) {
	int positives = 0;
	int negatives = 0;

	for (unsigned int i = 0; i < trainDataset.size(); i++) {
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

pair <int, int> RandomForest::entropyDataset(vector<vector<char> > trainDataset) {
	int positiveValues = 0;
	int negativeValues = 0;

	for (unsigned int i = 0; i < trainDataset.size(); i++) {
		vector<char> example = trainDataset[i];
		if (example[1024] == 'p') {
			positiveValues += 1;
		}
		else {
			negativeValues += 1;
		}
	}

	pair <int, int> result = make_pair(positiveValues, negativeValues);
	return result;
}

float RandomForest::entropy(pair <int, int>  params, float total) {
	float positiveResult = 0;
	float negativeResult = 0;
	float positiveCoefficient = 0;
	float negativeCoefficient = 0;
	float positiveValues = params.first;
	float negativeValues = params.second;

	if (total > 0) {
		positiveCoefficient = positiveValues / total;
		negativeCoefficient = negativeValues / total;
	}
	if (positiveCoefficient != 0) {
		positiveResult = ((-1) * positiveCoefficient) * (log2(positiveCoefficient));
	}
	if (negativeCoefficient != 0) {
		negativeResult = ((-1) * negativeCoefficient) * (log2(negativeCoefficient));
	}
	return (positiveResult + negativeResult);
}

pair < vector<vector<char> >, vector<vector<char> >> RandomForest::amountExamples(int attr_idex, int totalExamples, vector<vector<char> > trainDataset) {
	vector<vector<char> > examplesWithZero;
	vector<vector<char> > examplesWithOne;
	for (int j = 0; j < totalExamples; j++) {
		vector<char> row = trainDataset[j];

		if ((row[attr_idex]) == '0') {
			examplesWithZero.push_back(row);
		}
		else if ((row[attr_idex]) == '1') {
			examplesWithOne.push_back(row);
		}
	}
	pair < vector<vector<char> >, vector<vector<char> >> result = make_pair(examplesWithZero, examplesWithOne);
	return result;
}

float RandomForest::gain(int attr_idex, float entropyValue, vector<vector<char> > trainDataset) {
	float totalExamples = trainDataset.size();
	pair < vector<vector<char> >, vector<vector<char> >> examples = amountExamples(attr_idex, totalExamples, trainDataset);
	float examplesZero = examples.first.size();
	float examplesOne = examples.second.size();
	float S0 = (examplesZero / totalExamples);
	pair < int, int> resultZero = entropyDataset(examples.first); //falta pasar por param examples["examplesWithZero"]

	float entropyS0 = entropy(resultZero, examplesZero);

	float S1 = (examplesOne / totalExamples);
	pair < int, int> resultOne = entropyDataset(examples.second); //falta pasar por param examples["examplesWithOne"]
	float entropyS1 = entropy(resultOne, examplesOne);

	float gainValue = entropyValue - ((S0 * entropyS0) + ((S1)*entropyS1));
	return gainValue;
}

vector<vector<char> > RandomForest::filter_data_set(int attr, char cvalue, vector<vector<char> > trainDataset) {
	vector<vector<char> > resultDataset;
	for (unsigned int i = 0; i < trainDataset.size(); i++) {
		vector<char> row = trainDataset[i];
		if (row[attr] == cvalue) {
			resultDataset.push_back(row);
		}
	}
	return resultDataset;
}

int RandomForest::deleteElement(int arr[], int n, int x)
{
	// Search x in array
	int i;
	for (i = 0; i < n; i++)
		if (arr[i] == x)
			break;

	// If x found in array
	if (i < n)
	{
		// reduce size of array and move all
		// elements on space ahead
		n = n - 1;
		for (int j = i; j < n; j++)
			arr[j] = arr[j + 1];
	}

	return n;
}

// Implementacion de algoritmo ID3 para construir arboles de decision
struct Node* RandomForest::id3(vector<int> myAttrs, vector<vector<char> > trainId3) {
	
	int attrsSize = myAttrs.size();
	int gainZero = 1; //definir gainZero como param para podar arbol
	pair <int, int> classes = countClasses(trainId3);
	struct Node* root;
	if (classes.first == trainId3.size()) {
		root = new Node("p");
		//cout << "    primer if fin" << endl;
	}
	else if (classes.second == trainId3.size()) {
		//cout << "    segundo if comienzo" << endl;
		root = new Node("n");
		//cout << "    segundo if fin" << endl;
	}
	else if (attrsSize == 0) {
		//cout << "    tercer if comienzo" << endl;
		if (classes.first >= classes.second) {
			root = new Node("p");
		}
		else {
			root = new Node("n");
		}
		//cout << "    tercer if fin" << endl;
	}
	else {
		//cout << "    Ultimo else comienzo" << endl;
		pair <int, int> result = entropyDataset(trainId3);
		float entropyValue = entropy(result, trainId3.size());
		int bestAttr = myAttrs[0];
		float bestGain = gain(bestAttr, entropyValue, trainId3);
		
		for (int k = 0; k < attrsSize; ++k) {
			int attr = myAttrs[k];
			float attrGain = gain(attr, entropyValue, trainId3);
			if (attrGain > bestGain) {
				bestGain = attrGain;
				bestAttr = attr;
			}
		}

		//Corto recursi�n, atributos ya no me aportan info          
		if ((gainZero == 1) && (bestGain == 0)) {
			//cout << "      Ultimo else corto recursion" << endl;
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
			//cout << "      Ultimo else corto recursion else" << endl;
			string bestAttrChar = to_string(bestAttr);
			root = new Node(bestAttrChar);
			vector<vector<char> > negativeDataset = filter_data_set(bestAttr, '0', trainId3);
			vector<vector<char> > positiveDataset = filter_data_set(bestAttr, '1', trainId3);

			// << "      Ultimo else corto recursion else primer parte" << endl;
			if (negativeDataset.size() == 0) {
				root->left = new Node("p");
			}
			else {
				vector<int> copyAttrs;
				//copy(myAttrs, myAttrs + attrsSize, copyAttrs);
				copyAttrs = myAttrs;

				std::vector<int>::iterator it;
				int bestAttrIndex = 0;
				it = find(myAttrs.begin(), myAttrs.end(), bestAttr);
				if (it != myAttrs.end())
					bestAttrIndex = it - myAttrs.begin();
				copyAttrs.erase(copyAttrs.begin() + bestAttrIndex);
				//int newSize = deleteElement(copyAttrs, attrsSize, bestAttr);
				root->left = id3(copyAttrs, negativeDataset);
			}

			//cout << "      Ultimo else corto recursion else primer parte fin" << endl;
			//cout << "      Ultimo else corto recursion else segunda parte" << endl;
			if (positiveDataset.size() == 0) {
				root->right = new Node("n");
			}
			else {
				vector<int> copyAttrs;
				//copy(myAttrs, myAttrs + attrsSize, copyAttrs);
				copyAttrs = myAttrs;

				std::vector<int>::iterator it;
				int bestAttrIndex = 0;
				it = find(myAttrs.begin(), myAttrs.end(), bestAttr);
				if (it != myAttrs.end())
					bestAttrIndex = it - myAttrs.begin();
				copyAttrs.erase(copyAttrs.begin() + bestAttrIndex);
				//int newSize = deleteElement(copyAttrs, attrsSize, bestAttr);
				root->right = id3(copyAttrs, positiveDataset);
			}
			//cout << "      Ultimo else corto recursion else segunda parte fin" << endl;
		}
		//cout << "    Ultimo else fin" << endl;
	}
	//printf("arbol parcial:\n");
	//printBT(root);
	//cout << "arbol parcial id3: " << node.dump() << endl;
	return root;
}