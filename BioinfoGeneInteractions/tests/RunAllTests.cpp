/*
 * Main.c
 *
 *  Created on: 09/02/2011
 *      Author: william
 */

#include <iostream>
#include <assert.h>

using namespace std;

void testHashFunction();
void testIntegerToBits();
void testGrayCode();

int main( int argc, char **argv ) {

	cout << "RunAllTests Execution \n";
	//	testHashFunction();
	//	testIntegerToBits();
	testGrayCode();
	assert(0);

}
