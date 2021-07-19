/*
 * JavaProgressMonitor.cpp
 *
 *  Created on: 17/01/2013
 *      Author: william
 */
#include <jni.h>
#include <map>

#include "ProgressMonitor.h"
#include "JavaProgressMonitor.h"

#define TASK_STATE_RUNNING 1
#define TASK_STATE_FINALIZED 2
#define TASK_STATE_CANCELLED 3

JavaProgressMonitor::JavaProgressMonitor( JNIEnv *env,
		jobject *_jprogressStatus ) :
	ProgressMonitor() {

	if( env->GetJavaVM( &jvm ) < 0 ) {
		printf( "[WARN] JVM not setted properly! Setted NULL.\n" );
		jvm = NULL;
	} else {
		printf( "[INFO] JVM setted successfully!\n" );
	}
	jprogressStatus = _jprogressStatus;
}

JavaProgressMonitor::~JavaProgressMonitor() {

}

void JavaProgressMonitor::notifyThreadFinish( int deviceId ) {
	if( jvm == NULL ) {
		printf(
				"[WARN] OMPThread[%d] - JavaProgressMonitor::notifyThreadFinish > "
					"Cannot Detach current thread. JVM is null, returning...\n",
				deviceId );
	} else {
		jint errorCode = jvm->DetachCurrentThread();
		if( errorCode < 0 ) {
			printf(
					"[WARN] OMPThread[%d] - JavaProgressMonitor::notifyThreadFinish > "
						"Cannot Detach current thread. Error on detaching code [%d], returning...\n",
					deviceId, errorCode );
		}
	}
}

void JavaProgressMonitor::progressStarted( int deviceCount,
		int deviceId,
		int blockDim,
		int threadDim ) {

	printf( "OMPThread[%d] - JavaProgressMonitor::progressStarted > "
		"deviceCount[%d],  blockDim[%d], threadDim[%d]\n", deviceId,
			deviceCount, blockDim, threadDim );

	// http://192.9.162.55/docs/books/jni/html/other.html#26206
	// Need attach each thread do JNIEnv.
	JNIEnv *env;
	if( jvm == NULL || jvm->AttachCurrentThreadAsDaemon( (void **) &env, NULL )
			< 0 ) {
		printf(
				"[WARN] OMPThread[%d] - JavaProgressMonitor::progressStarted > "
					"Cannot attach current thread. JNIEnv not setted, returning...\n",
				deviceId );
		return;
	}

	if( blockDim == 0 && threadDim == 0 ) {
		return;
	}

	jclass cls;
	jfieldID fid;

	cls = env->GetObjectClass( *jprogressStatus );
	if( cls == 0 ) {
		printf( "GetObjectClass returned 0\n" );
		return;
	}

	fid = env->GetFieldID( cls, "nGpus", "I" );
	if( fid == 0 ) {
		printf( "GetFieldID nGpus returned 0\n" );
		return;
	}
	env->SetIntField( *jprogressStatus, fid, deviceCount );

	fid = env->GetFieldID( cls, "blockDim", "I" );
	if( fid == 0 ) {
		printf( "GetFieldID blockDim returned 0\n" );
		return;
	}
	env->SetIntField( *jprogressStatus, fid, blockDim );

	fid = env->GetFieldID( cls, "threadDim", "I" );
	if( fid == 0 ) {
		printf( "GetFieldID threadDim returned 0\n" );
		return;
	}
	env->SetIntField( *jprogressStatus, fid, threadDim );

	fid = env->GetFieldID( cls, "state", "I" );
	if( fid == 0 ) {
		printf( "GetFieldID state returned 0\n" );
		return;
	}
	env->SetIntField( *jprogressStatus, fid, TASK_STATE_RUNNING );
}

void JavaProgressMonitor::progressStatus( int deviceCount,
		int deviceId,
		long long int currentIteration,
		long long int endIteration,
		float processingStatus,
		float kernelTimeIteration,
		long long int numberOfAttractorFounded,
		map<long long int, long long int> *result ) {

	// http://192.9.162.55/docs/books/jni/html/other.html#26206
	// Need attach each thread do JNIEnv.
	JNIEnv *env;
	if( jvm == NULL || jvm->AttachCurrentThreadAsDaemon( (void **) &env, NULL )
			< 0 ) {
		printf(
				"[WARN] OMPThread[%d] - JavaProgressMonitor::progressStarted >  "
					"Cannot attach current thread. JNIEnv not setted, returning...\n",
				deviceId );
		return;
	}

	jclass cls;
	jfieldID fid;

	cls = env->GetObjectClass( *jprogressStatus );
	if( cls == 0 ) {
		printf( "GetObjectClass returned 0\n" );
		return;
	}

	fid = env->GetFieldID( cls, "state", "I" );
	if( fid == 0 ) {
		printf( "GetFieldID returned 0\n" );
		return;
	}
	env->SetIntField( *jprogressStatus, fid, TASK_STATE_RUNNING );

	jint jdeviceCount = deviceCount;
	jint jdeviceId = deviceId;
	jdouble jprocessIndicator = (double) processingStatus;
	jmethodID mid = env->GetMethodID( cls, "setProgressIndicator", "(IID)V" );
	if( mid == 0 ) {
		printf( "GetMethodID returned 0\n" );
		return;
	}
	env->CallVoidMethod( *jprogressStatus, mid, jdeviceCount, jdeviceId,
			jprocessIndicator );
}

void JavaProgressMonitor::progressFinished( int deviceCount,
		long long int numberOfStates,
		double duration,
		map<long long int, long long int> *globalCountMap ) {

	// http://192.9.162.55/docs/books/jni/html/other.html#26206
	// Need attach each thread do JNIEnv.
	JNIEnv *env;
	if( jvm == NULL || jvm->AttachCurrentThreadAsDaemon( (void **) &env, NULL )
			< 0 ) {
		printf( "[WARN] JavaProgressMonitor::progressFinished >  "
			"Cannot attach current thread. JNIEnv not setted, returning...\n" );
		return;
	}

	jclass cls;
	jfieldID fid;
	jlongArray jattractorStates;
	jlongArray jattractorBasinSize;

	cls = env->GetObjectClass( *jprogressStatus );
	if( cls == 0 ) {
		printf( "GetObjectClass returned 0\n" );
		return;
	}

	//	(I)V : (I) int argument and void return
	jint jnumberOfAttractors = globalCountMap->size();
	jmethodID mid = env->GetMethodID( cls, "buildAttractorStateArrays", "(I)V" );
	if( mid == 0 ) {
		printf( "GetMethodID buildAttractorStateArrays returned 0\n" );
		return;
	}
	env->CallVoidMethod( *jprogressStatus, mid, jnumberOfAttractors );

	// [J : double array field
	fid = env->GetFieldID( cls, "attractorStates", "[J" );
	if( fid == 0 ) {
		printf( "GetFieldID attractorStates returned 0\n" );
		return;
	}
	jattractorStates = (jlongArray) env->GetObjectField( *jprogressStatus, fid );

	// [J : double array field
	fid = env->GetFieldID( cls, "attractorBasinSize", "[J" );
	jattractorBasinSize = (jlongArray) env->GetObjectField( *jprogressStatus,
			fid );

	jlong *attractorStates = env->GetLongArrayElements( jattractorStates, 0 );
	jlong *attractorBasinSize = env->GetLongArrayElements( jattractorBasinSize,
			0 );

	int i = 0;
	for (map<long long int, long long int>::iterator itCountMap =
			globalCountMap->begin(); itCountMap != globalCountMap->end(); ++itCountMap, i++) {
		attractorStates[i] = itCountMap->first;
		attractorBasinSize[i] = itCountMap->second;
	}

	// Return array elements to java.
	env->ReleaseLongArrayElements( jattractorStates, attractorStates, 0 );
	env->ReleaseLongArrayElements( jattractorBasinSize, attractorBasinSize, 0 );

	// I : int field
	fid = env->GetFieldID( cls, "state", "I" );
	env->SetIntField( *jprogressStatus, fid, TASK_STATE_FINALIZED );

	// D : double field
	fid = env->GetFieldID( cls, "duration", "D" );
	env->SetDoubleField( *jprogressStatus, fid, duration );
}

