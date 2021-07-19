/*
 * HostUtils.h
 *
 *  Created on: 02/05/2011
 *      Author: william
 */

extern int * getPointerToMatrix( size_t size );

extern struct EDGE * getPointerToEdges( size_t size );

extern int * buildWindow( int wm, int wn );

extern
void printMatrix( const char *prefixLine,
		const char *title,
		int *mat,
		int lengthBreak,
		int size );
extern void
printMatrix( const char *title, int *mat, int lengthBreak, int size );

