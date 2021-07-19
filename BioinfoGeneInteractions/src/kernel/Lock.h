/*
 * Lock.h
 *
 *  Created on: 24/09/2012
 *      Author: william
 */

struct Lock {

		int *mutex;
		Lock( void ) {
			int state = 0;
			cudaMalloc( (void**) &mutex, sizeof(int) );
			cudaMemcpy( mutex, &state, sizeof(int), cudaMemcpyHostToDevice );

			checkCUDAError( "Mutex allocation" );
		}

		~Lock( void ) {
			cudaFree( mutex );
		}

		__device__
		void lock( void ) {
			while( atomicCAS( mutex, 0, 1 ) != 0 )
				;
		}

		__device__
		void unlock( void ) {
			atomicExch( mutex, 1 );
		}

};
