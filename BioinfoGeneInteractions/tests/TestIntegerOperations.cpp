#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

/* itoa:  convert n to characters in s */
void itoa( long long int n, char s[] ) {
	int i;
	i = 0;
	do {
		// Generate digits in reverse order
		s[i++] = n % 2 + '0'; /* get next digit */
		//		cout << "n = " << n << "\n";
	} while( (n /= 2) > 0 ); /* delete it */
	s[i] = '\0';
}

void testIntegerToBits() {
	// numero maximo representado 2^1000 -1
	char *str = (char*) malloc( 1000 * sizeof(char) );

	long long int N = (long long int) pow( 2, 999 );
	//	itoa( N, str );
	printf( "[%s] \n", str );
}
