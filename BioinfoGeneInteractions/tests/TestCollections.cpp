#include <iostream>
#include <math.h>
#include <list>
#include <map>

using namespace std;

void testCollections() {

	list<int> mylist;
	mylist.push_back( 1 );
	mylist.push_back( 2 );
	mylist.push_back( 3 );
	mylist.push_back( 4 );
	mylist.push_back( 5 );

	cout << "List values \n";

	list<int>::iterator it;
	int i = 0;
	for (it = mylist.begin(); it != mylist.end(); ++it) {
		cout << "L[" << i << "] = " << *it << " \n ";
		i++;
	}

	cout << "size = " << mylist.size() << "\n";

	// Map manipulation. O(logn) operations
	map<int, int> mapa;
	pair<map<int, int>::iterator, bool> ret;

	for (i = 0; i < 10; i++) {

		cout << "find " << i << " = " << (mapa.find( i ) != mapa.end()) << "\n";

		mapa.insert( pair<int, int> ( i, i ) );

	}

	map<int, int>::iterator itMap;
	for (itMap = mapa.begin(); itMap != mapa.end(); ++itMap) {
		cout << "find key " << itMap->first << " = " << (mapa.find(
				itMap->first ) != mapa.end()) << "\n";

		cout << "Key: " << itMap->first << " Value: " << itMap->second << "\n";
	}

}
