// JNI Libraries
#include <jni.h>
#include "br_usp_ime_bioinfogen_JBioinfoNativeTask.h"

// Project Libraries
#include <stdio.h>
#include <stdlib.h>
#include "DataProvider.h"
#include "ProgressMonitor.h"
#include "JavaProgressMonitor.h"
#include "GeneExecutor.h"
#include "ParallelGeneNetworkExecutor.h"
#include "FileMatrixDataProvider.h"
#include "KernelUtils.h"

using namespace std;

JNIEXPORT jint JNICALL Java_br_usp_ime_bioinfogen_JBioinfoNativeTask_execute( JNIEnv *env,
		jobject jsource,
		jstring jtaskId,
		jstring jtaskFileName,
		jint jgridDim,
		jint jblockDim,
		jobject jprogressStatus ) {
	jint FINALIZED = 2;
	jint ERROR = -1;
	try {

		cout << "Java_br_ime_usp_bioinfogen_JBioinfoNative_execute [Started]"
				<< endl;

		const char *charTaskId = env->GetStringUTFChars( jtaskId, 0 );
		string taskId( charTaskId );

		const char *charTaskFileName =
				env->GetStringUTFChars( jtaskFileName, 0 );
		string inputFileName( charTaskFileName );

		cout << "Task: " << taskId << " Input:" << inputFileName << endl;

		jint blockDiv = jgridDim;
		jint threadDiv = jblockDim;
		int numGPUThreads = 2;
		setBlockAndThreadConfig( blockDiv, threadDiv );
		setNumberOfGPUThreads( numGPUThreads );

		DataProvider *dataProvider;
		FileMatrixDataProvider fileMatrixDataProvider( inputFileName );
		dataProvider = &fileMatrixDataProvider;

		JavaProgressMonitor javaProgressProgressMonitor( env, &jprogressStatus );
		ProgressMonitor *progressMonitor;
		progressMonitor = &javaProgressProgressMonitor;

		ParallelGeneNetworkExecutor geneNetworkExecutor( dataProvider,
				progressMonitor );
		GeneExecutor *geneExecutor;
		geneExecutor = &geneNetworkExecutor;
		geneExecutor->execute();

		cout
				<< "Java_br_ime_usp_bioinfogen_JBioinfoNative_execute [Releasing objects]"
				<< endl;

		env->ReleaseStringUTFChars( jtaskId, charTaskId );
		env->ReleaseStringUTFChars( jtaskFileName, charTaskFileName );

		cout << "Java_br_ime_usp_bioinfogen_JBioinfoNative_execute [Finalized]"
				<< endl;

		return FINALIZED;

	} catch (int e) {
		cout << "Fatal error on execution: " << e;
	}
	return ERROR;
}

