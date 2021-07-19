#include <stdio.h>
#include <stdlib.h>
#include "DataProvider.h"
#include "ProgressMonitor.h"
#include "GeneExecutor.h"
#include "ParallelGeneNetworkExecutor.h"
#include "GraphGeneNetworkExecutor.h"
#include "MatrixFunctionConverterExecutor.h"
#include "FileMatrixDataProvider.h"
#include "HostUtils.h"
#include "KernelUtils.h"

using namespace std;

const string FILE_PARAM = "-f";
const string GRAPH_PARAM = "-g";
const string PARALLEL_PARAM = "-p";
const string GENERATE_MATRIX = "-genmat";
const string CONVERT_MATRIX_CNET = "-cnet";

const int EXEC_TYPE_GRAPH = 1;
const int EXEC_TYPE_PARALLEL = 2;
const int EXEC_GENERATE_RANDOM_MATRIX = 3;
const int EXEC_CONVERT_MATRIX_CNET = 4;
const int EXEC_TYPE_INVALID = -1;

void showHelp() {
	cout << "Invalid arguments\n.Execution: \n";
	cout << "  [Executable] " << FILE_PARAM
			<< " [matrix file name] <pathway_option>\n";

	cout << ">> The <pathway_option> can be: \n";
	cout << "\t[" << GRAPH_PARAM
			<< "] [N_BLOCK_DIV] [N_THREAD_DIV] (execute graph based algorithm in a single GPU) \n";

	cout << "\t[" << PARALLEL_PARAM
			<< "] [N_BLOCK_DIV] [N_THREAD_DIV] [N_GPUS] (execute parallel search based algorithm with multiple GPUs)\n";

	cout << "\t[" << GENERATE_MATRIX
			<< "] [QTD] [N] (execute random generation of one or more regulation matrix)\n";

	cout << "\t[" << CONVERT_MATRIX_CNET
			<< "] Print .cnet file format to <matrix file name> file\n";

	cout
			<< "\n\nExample: [Executable] -f matrixfile.txt -p 128 32 2 \n"
				"This will execute parallel strategy with blockDim/threadDim divisor factor 128 and 32 with 2 GPUs.";
}

int main( int argc, char **argv ) {

	int execType = EXEC_TYPE_INVALID;

	string filename = "";
	int quantityMatrix = -1;
	int matrixSizeXY = -1;
	int numberOfInputs = 2;

	for (int i = 1; i < argc; i++) {
		string param( argv[i] );

		if( FILE_PARAM.compare( param ) == 0 ) {
			if( (i + 1) < argc ) {
				filename = argv[++i];
			} else {
				execType = EXEC_TYPE_INVALID;
				break;
			}
		} else {

			if( GRAPH_PARAM.compare( param ) == 0 ) {
				execType = EXEC_TYPE_GRAPH;

				if( (i + 2) < argc ) {
					// Numero de blocos e threads por linha de comando
					int bloDiv = atoi( argv[++i] );
					int thDiv = atoi( argv[++i] );
					setBlockAndThreadConfig( bloDiv, thDiv );
				}

				if( (i + 1) < argc ) {
					// Numero de GPU threads
					int numGPUThreads = atoi( argv[++i] );
					setNumberOfGPUThreads( numGPUThreads );
				}

			} else if( PARALLEL_PARAM.compare( param ) == 0 ) {
				execType = EXEC_TYPE_PARALLEL;

				if( (i + 2) < argc ) {
					// Numero de blocos e threads por linha de comando
					int bloDiv = atoi( argv[++i] );
					int thDiv = atoi( argv[++i] );
					setBlockAndThreadConfig( bloDiv, thDiv );
				}

				if( (i + 1) < argc ) {
					// Numero de GPU threads
					int numGPUThreads = atoi( argv[++i] );
					setNumberOfGPUThreads( numGPUThreads );
				}

			} else if( GENERATE_MATRIX.compare( param ) == 0 ) {
				execType = EXEC_GENERATE_RANDOM_MATRIX;
				if( (i + 2) < argc ) {
					quantityMatrix = atoi( argv[++i] );
					matrixSizeXY = atoi( argv[++i] );
					// Optional
					if( (i + 1) < argc ) {
						numberOfInputs = atoi( argv[++i] );
					}
				} else {
					execType = EXEC_TYPE_INVALID;
					break;
				}

			} else if( CONVERT_MATRIX_CNET.compare( param ) == 0 ) {
				execType = EXEC_CONVERT_MATRIX_CNET;
			} else {
				execType = EXEC_TYPE_INVALID;
				break;
			}
		}
	}

	// Final validations
	if( execType == EXEC_TYPE_INVALID || (filename == "" && execType
			== EXEC_TYPE_GRAPH) || (filename == "" && execType
			== EXEC_TYPE_PARALLEL) ) {
		showHelp();
		return 1;
	}
	// Execution

	if( execType == EXEC_GENERATE_RANDOM_MATRIX ) {
		FileMatrixDataProvider matrixGenerator;
		matrixGenerator.generateRandomMatrix( quantityMatrix, matrixSizeXY,
				numberOfInputs );
	} else {
		GeneExecutor *geneExecutor;
		DataProvider *dataProvider;
		ProgressMonitor progressMonitor;
		FileMatrixDataProvider fileMatrixDataProvider( filename );
		dataProvider = &fileMatrixDataProvider;

		if( execType == EXEC_TYPE_GRAPH ) {
			// 2. Execution: Analisys after complete graph states building
			//
			GraphGeneNetworkExecutor *graphNetworkExecutor =
			new GraphGeneNetworkExecutor( dataProvider,
					&progressMonitor );
			geneExecutor = graphNetworkExecutor;

		} else if( execType == EXEC_TYPE_PARALLEL ) {
			// 3. Execution: Analisys by parallel attractor search
			//
			ParallelGeneNetworkExecutor *geneNetworkExecutor =
			new ParallelGeneNetworkExecutor( dataProvider,
					&progressMonitor );
			geneExecutor = geneNetworkExecutor;

		} else if( execType == EXEC_CONVERT_MATRIX_CNET ) {
			MatrixFunctionConverterExecutor *matrixConverterExecutor =
			new MatrixFunctionConverterExecutor( dataProvider );
			geneExecutor = matrixConverterExecutor;
		}

		geneExecutor->execute();

		delete geneExecutor;
	}
	return 0;
}
