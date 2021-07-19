/*
 * HostUtils.c
 *
 *  Created on: 02/05/2011
 *      Author: william
 */
#include <stdio.h>
#include "HostUtils.h"
#include "GraphKernel.h"

int * getPointerToMatrix( size_t size ) {
	int *matrix = (int*) malloc( size * sizeof(int) );
	return matrix;
}

Edge * getPointerToEdges( size_t size ) {
	Edge *egs = NULL;
	long memory = size * sizeof(Edge);
	if( memory > 0 ) {
		egs = (Edge*) malloc( memory );
	}
	return egs;
}

int * buildWindow( int wm, int wn ) {
	// Exemple for a window
	// 0 3 6
	// 1 4 7
	// 2 5 8
	//
	// window: 1 2 3 4 5 6 7 8 9

	int sizeWindow = (wm * wn);
	int *window = getPointerToMatrix( sizeWindow );

	for (int i = 0; i < sizeWindow; i++) {
		window[i] = 0;
		if( i == 1 || i == 3 || i == 4 || i == 5 || i == 7 ) {
			window[i] = 1;
		}
	}
	return window;
}

void printMatrix( const char *prefixLine,
		const char *title,
		int *mat,
		int lengthBreak,
		int size ) {
	printf( "\n%s ::%s:: \n", prefixLine, title );
	printf( "%s", prefixLine );
	for (int i = 0; i < size; i++) {
		if( i % lengthBreak == 0 ) {
			printf( "\n" );
			printf( "%s", prefixLine );
		}
		printf( "%2d", mat[i] );
	}
	printf( "\n" );
}

void printMatrix( const char *title, int *mat, int lengthBreak, int size ) {
	printMatrix( "", title, mat, lengthBreak, size );
}
