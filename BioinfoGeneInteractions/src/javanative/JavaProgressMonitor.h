/*
 * JavaProgressMonitor.h
 *
 *  Created on: 17/01/2013
 *      Author: william
 */

class JavaProgressMonitor: public ProgressMonitor {
	public:
		JavaProgressMonitor( JNIEnv *env, jobject *jprogressStatus );

		virtual ~JavaProgressMonitor();

		virtual void progressStarted( int deviceCount,
				int deviceId,
				int blockDim,
				int threadDim );

		virtual void progressStatus( int deviceCount,
				int deviceId,
				long long int currentIteration,
				long long int endIteration,
				float processingStatus,
				float kernelTimeIteration,
				long long int numberOfAttractorFounded,
				map<long long int, long long int> *result );

		virtual void progressFinished( int deviceCount,
				long long int numberOfStates,
				double executionTime,
				map<long long int, long long int> *globalCountMap );

		virtual void notifyThreadFinish( int deviceId );

	protected:
		jobject *jprogressStatus;
		JavaVM *jvm;
};
