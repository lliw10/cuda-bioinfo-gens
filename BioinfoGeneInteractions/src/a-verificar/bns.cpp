#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>
#include "Solver.h"

#define INDEX_OF_FIRST_VARIABLE 1
#define MAXIMUM_NUMBER_OF_RELEVANT_NODES_COUNTED 100

#define PRINT_STATE

#define PRINTF(X,Y)
//#define PRINTF(X,Y) printf(X,Y)

#define PRINT(X) printf(#X":%d\n",X)
//#define PRINT(X)

#define ASSERT_ON

#ifdef ASSERT_ON
#define ASSERT(x)           assert(x)
#else
#define ASSERT(x)
#endif

/* Reading from file informaition*/
int number_of_var, redunt_number_of_var, number_print_var;
int *var2red_var;
int *redunt_var2var;

#define MAX_LENGTH 10000
#define MAX_NUMBER_OF_INPUTS 20
/*---------------------------------*/

struct GLOBAL_VAR {
		vec<vec<Lit> > *orig_clauses;
		Solver *S;
		unsigned number_of_var;
		unsigned depth;
} global_var;

static unsigned size_nonred_array;

void ReadNET( char *fileName, Solver& S );

void read_minterm_file_to_solver( char * FileName,
		Solver &S,
		int *input_array,
		int input_array_size,
		Lit ouput_lit );
/* --------------------------------------------------------------------------- */

/*Compare stats a and b in vector of stats stored in stats_sequance.*/
/*The first state in sequence assumed to have index 0, each state consists of sequence of values of */
bool compare_states( unsigned a,
		unsigned b,
		unsigned number_var,
		vec<lbool>& stats_sequance ) {
	lbool *a_pr, *b_pr;
	unsigned size = stats_sequance.size();
	ASSERT(!( size < (a+1)*number_var || size < (b+1)*number_var ));

	a_pr = &stats_sequance[a * number_var];
	b_pr = &stats_sequance[b * number_var];

	while( number_var-- ) {
		if( *a_pr != *b_pr )
			return false;
		a_pr++;
		b_pr++;
	}
	return true;

}

/*Make state with index a from the stats_sequance not allowed in future solutions of S in position with index b */
void constrain_state( unsigned a,
		vec<lbool>& stats_sequance,
		unsigned b,
		Solver &S,
		unsigned number_var ) {

	vec < Lit > lits;
	int a_tmp = a * number_var;
	int b_tmp = b * number_var;

	ASSERT( a_tmp+number_var <= stats_sequance.size() );
	ASSERT( b_tmp+number_var <= S.nVars() );

	while( number_var-- ) {
		if( stats_sequance[a_tmp] != l_Undef ) {

#ifdef PRINT_STATE
			printf( "%s", (stats_sequance[a_tmp] == l_True) ? "1" : "0" );
#endif

			lits.push( (stats_sequance[a_tmp] == l_True) ? ~Lit( b_tmp ) : Lit(
					b_tmp ) );
		} else {
			printf( "-" );
		}
		a_tmp++;
		b_tmp++;
	}
	S.addClause( lits );
#ifdef PRINT_STATE
	puts( " " );
#endif
}

void add_clauses_with_variable_shift( vec<vec<Lit> > &clauses2add,
		Solver &S,
		int shift ) {

	unsigned i;
	vec < Lit > lits;
	lits.clear();

	i = clauses2add.size();
	while( i-- ) {//PRINT(i);
		int j = clauses2add[i].size();//PRINT(j);
		while( j-- ) {//PRINT(toInt(clauses2add[i][j]) + (shift+shift));
			lits.push( toLit( toInt( clauses2add[i][j] ) + (shift + shift) ) );
		}

		/*
		 j=lits.size();
		 while(j--){
		 PRINT(toInt(lits[j]));

		 }
		 */

		S.addClause( lits );
		lits.clear();
	}

}

void construct_depth( unsigned k ) {

	int i;
	int depth_left_to_build = k - global_var.depth;
	Solver *S = global_var.S;

	if( depth_left_to_build < 0 )
		return;

	i = global_var.number_of_var * depth_left_to_build;
	while( i-- ) {
		(*S).newVar();
	}

	for (i = global_var.depth - 1; i < (k - 1); i++) {
		add_clauses_with_variable_shift( *global_var.orig_clauses, *S, i
				* global_var.number_of_var );
	}

	global_var.depth = k;
}

int main( int argc, char *argv[] ) {

	int i, j;
	Solver S;
	unsigned atractor_count = 0;
	unsigned atractor_stats_count = 0;
	vec < Lit > lits;

	ReadNET( argv[1], S );

	vec<Clause*>* orig_clauses_pr = S.Clauses();
	vec < vec<Lit> > orig_clauses;

	lits.clear();

	i = (*orig_clauses_pr).size();
	while( i-- ) {
		int j = (*((*orig_clauses_pr)[i])).size();
		//PRINT(j);

		while( j-- ) {
			//PRINT(toInt((*((*orig_clauses_pr)[i]))[j]));
			lits.push( (*((*orig_clauses_pr)[i]))[j] );
		}

		orig_clauses.push( lits );
		lits.clear();
	}

	/*Handel assignments of variables made in solver*/
	i = number_of_var;
	while( i-- ) {
		if( S.value( i ) != l_Undef ) {
			lits.push( (S.value( i ) == l_True) ? Lit( i ) : ~Lit( i ) );
			orig_clauses.push( lits );
			lits.clear();
		}
	}

	//PRINT(orig_clauses.size());

	global_var.orig_clauses = &orig_clauses;
	global_var.S = &S;
	global_var.number_of_var = number_of_var;
	global_var.depth = 2;

	if( number_of_var < 100 ) {
		construct_depth( number_of_var );
	} else {
		construct_depth( 100 );
	}

	//puts("Start searching...");
	while( 1 ) {

		if( !S.solve() )
			break;
		/*
		 for(i=0;i<3;i++){
		 for (int y = i*number_of_var; y < i*number_of_var+number_of_var; y++){
		 if (S.model[y] != l_Undef){
		 printf( "%s", (S.model[y]==l_True)?"1":"0");
		 }else{
		 printf("-");
		 }
		 }puts(" ");
		 }
		 */

		for (i = 1; i < global_var.depth; i++) {

			if( compare_states( 0, i, number_of_var, S.model ) ) {
				atractor_count++;
				atractor_stats_count += i;
				//PRINT(atractor_stats_count);
				for (j = 0; j < i; j++) {
					constrain_state( j, S.model, 0, S, number_of_var );
				}
				printf( "Attractor %d is of length %d\n\n", atractor_count, i );
				break;
			}
		}

		if( global_var.depth == i ) {
			construct_depth( global_var.depth << 1 );
			//printf("Depth of unfolding transition relation is increased to %d\n",global_var.depth);
		}
	}
	printf( "The number of attractors is %d\n", atractor_count );

	//PRINT(atractor_count);
	//PRINT(atractor_stats_count);
	//PRINT(global_var.depth);
	//printf(ret ? "SATISFIABLE\n" : "UNSATISFIABLE\n");
	return 0;

}

/* Read in NET file and returns 2^n transition function*/
void ReadNET( char *fileName, Solver& S ) {

	char temp[MAX_LENGTH];
	char slask[MAX_LENGTH];
	int i, k, tmp_1;

	int input_array[MAX_NUMBER_OF_INPUTS];
	int number_of_inputs, current_number_of_inputs = 0;
	int current_var;
	FILE *fp, *fp_temp;
	vec < Lit > lits;

	unsigned line_count = 0;

	//the following allows counting of line
#define fgets(X,Y,Z);  fgets((X),(Y),(Z));line_count++;

	/* Open NET file for reading */
	if( (fp = fopen( fileName, "r" )) == NULL ) {
		fprintf( stderr, "Couldn't open NET file: %s\n", fileName );
		exit( 1 );
	}

	/* Reading number of non-redundent var*/
	while( !feof( fp ) ) {
		fgets(temp,MAX_LENGTH, fp);
		if( strncmp( temp, ".v", 2 ) == 0 ) {
			/* Find size of the cubes */
			sscanf( temp, "%s %i", slask, &number_of_var );
			break;
		}
	}
	var2red_var = (int*) calloc( number_of_var + 1, sizeof(int) );

	size_nonred_array = number_of_var;

	redunt_number_of_var = number_of_var;
	redunt_var2var = (int*) calloc( redunt_number_of_var + 1, sizeof(int) );

	number_print_var = number_of_var;

	int count_nodes;
	for (count_nodes = 1; count_nodes <= number_of_var; count_nodes++) {
		redunt_var2var[count_nodes] = count_nodes;
		var2red_var[count_nodes] = count_nodes;
	}

	if( feof( fp ) ) {
		fprintf(
				stderr,
				"Wrong format of input file. End of file is reached but no node function is red" );
		exit( 1 );
	}

	k = number_of_var;
	while( k-- ) {
		S.newVar();
		S.newVar();
	}

	//RBN_init_global_var(bdd_manager);

	/*-------------------- Filling information about nodes  -------------------*/

	while( !feof( fp ) ) {
		fgets(temp,MAX_LENGTH, fp);
		next_vertex: if( strncmp( temp, ".n", 2 ) == 0 ) {
			/* Find size of the cubes */

			i = 3;
			sscanf( &temp[i], "%d", &current_var );

			if( current_var < 0 || current_var > redunt_number_of_var ) {
				fprintf(
						stderr,
						"Wrong format of input file in line %d.The varible %d in string:\n %s exceeds number of declared variables.\n",
						line_count, current_var, temp );
				exit( 1 );
			}

			i++;
			/* go to first space */
			while( temp[i] != ' ' ) {
				if( temp[i] == '\n' ) {
					fprintf(
							stderr,
							"Wrong format of input file in line %d. Wrong format of the string: %s\n",
							line_count, temp );
					exit( 1 );
				}
				i++;
			}
			i++;
			/* go to the end of sequense of spaces */
			while( temp[i] == ' ' ) {
				i++;
			}
			if( temp[i] == '\n' ) {
				fprintf(
						stderr,
						"Wrong format of input file in line %d. Wrong format of the string: %s\n",
						line_count, temp );
				exit( 1 );
			}
			sscanf( &temp[i], "%d", &number_of_inputs );

			if( number_of_inputs > MAX_NUMBER_OF_INPUTS ) {
				fprintf(
						stderr,
						"Wrong format of input file in line %d . Number of inputs exceeds allowed number of inputs %s\n",
						line_count, temp );
				exit( 1 );
			}

			while( 1 ) {
				i++;
				while( temp[i] != ' ' ) {
					if( temp[i] == '\n' ) {
						goto end_loops;
					}
					i++;
				}
				i++;
				/* go to the end of sequense of spaces */
				while( temp[i] == ' ' ) {
					i++;
				}
				if( temp[i] == '\n' ) {
					goto end_loops;
				}
				sscanf( &temp[i], "%d", &tmp_1 );

				if( tmp_1 < 0 || tmp_1 > redunt_number_of_var ) {
					fprintf(
							stderr,
							"Wrong format of input file in line %d. The varible %d in string:\n %s exceeds number of declared variables.\n",
							line_count, tmp_1, temp );
					exit( 1 );
				}

				if( redunt_var2var[tmp_1] == 0 ) {
					fprintf(
							stderr,
							"Wrong format of input file in line %d. One of the input was not declared in list of inputs %s\n",
							line_count, temp );
					exit( 1 );
				}

				input_array[current_number_of_inputs++] = redunt_var2var[tmp_1];

			}
			end_loops:

			if( current_number_of_inputs != number_of_inputs ) {
				fprintf(
						stderr,
						"Wrong format of input file in line %d. Declared number of inputs is not equal to actual number of input %s\n",
						line_count, temp );
				exit( 1 );
			}

			if( number_of_inputs == 0 ) {
				fgets(temp,MAX_LENGTH, fp);

				if( temp[0] == '1' ) {
					lits.push( Lit( current_var - INDEX_OF_FIRST_VARIABLE ) );
					S.addClause( lits );
					lits.clear();
				} else {
					if( temp[0] != '0' ) {
						printf( "Node %d assumed to be constant 0\n",
								current_var );
					}
					lits.push( ~Lit( current_var - INDEX_OF_FIRST_VARIABLE ) );
					S.addClause( lits );
					lits.clear();
				}
				goto next_vertex;
			}

			/*---------------  Reading in the function of the node ---------------*/

			current_number_of_inputs = 0;

			while( !feof( fp ) ) {
				fgets(temp,MAX_LENGTH, fp);
				if( temp[0] == '0' || temp[0] == '1' || temp[0] == '-' ) {

					for (i = 0; i < number_of_inputs; i++) {
						if( temp[i] == '-' )
							continue;

						lits.push( (temp[i] == '1') ? ~Lit( number_of_var
								+ input_array[i] - INDEX_OF_FIRST_VARIABLE )
								: Lit( number_of_var + input_array[i]
										- INDEX_OF_FIRST_VARIABLE ) );

						if( temp[i] != '1' && temp[i] != '0' ) {
							fprintf(
									stderr,
									"Wrong format of input file in line %d. Wrong line:%s\n",
									line_count, temp );
							exit( 1 );
						}
					}

					ASSERT((number_of_var + current_var - INDEX_OF_FIRST_VARIABLE) < S.nVars());

					i++;
					if( temp[i] == '1' ) {
						lits.push( Lit( current_var - INDEX_OF_FIRST_VARIABLE ) );
					} else {
						if( temp[i] == '0' ) {
							lits.push( ~Lit( current_var
									- INDEX_OF_FIRST_VARIABLE ) );
						} else {
							fprintf(
									stderr,
									"Wrong format of input file in line %d. Unexpected charecter in output of minterm in line:%s\n",
									line_count, temp );
							exit( 1 );
						}
					}
					S.addClause( lits );
					lits.clear();

				} else {

					if( redunt_var2var[current_var] == 0 ) {
						fprintf(
								stderr,
								"Wrong format of input file.The node:%d was not declared \n",
								current_var );
						exit( 1 );
					}

					break;

				}
			}

			if( !feof( fp ) ) {
				goto next_vertex;
			}

			/*We are here after we reach the end of the file*/
			if( redunt_var2var[current_var] == 0 ) {
				fprintf(
						stderr,
						"Wrong format of input file.The node:%d was not declared \n",
						current_var );
				exit( 1 );
			}

		}/* -------- end processing input informaition of a vertex ------------*/
	}
	/*end processing the file*/

	fclose( fp );

#undef fgets(X,Y,Z);

}

void read_minterm_file_to_solver( char * fileName,
		Solver &S,
		int *input_array,
		int input_array_size,
		Lit output_lit ) {

	char temp[MAX_LENGTH];
	FILE *fp;
	vec < Lit > lits;
	int i;

	//printf("%d ",toInt(output_lit));
	//getchar();
	lits.clear();

	if( (fp = fopen( fileName, "r" )) == NULL ) {
		fprintf( stderr, "Couldn't open file: %s\n", fileName );
		exit( 1 );
	}

	while( !feof( fp ) ) {
		fgets( temp, MAX_LENGTH, fp );
		if( temp[0] == '0' || temp[0] == '1' || temp[0] == '-' ) {
			break;
		}
	}

	for (i = 0; i < input_array_size; i++) {
		//PRINT(input_array[i]);
	}
	while( temp[0] != '.' ) {
		//puts(temp);
		for (i = 0; i < input_array_size; i++) {
			if( temp[i] == '-' )
				continue;
			lits.push( (temp[i] == '1') ? ~Lit( input_array[i]
					- INDEX_OF_FIRST_VARIABLE ) : Lit( input_array[i]
					- INDEX_OF_FIRST_VARIABLE ) );
			//printf("%d:%d ",i,toInt(lits[lits.size()-1]));
		}
		//puts(" ");
		lits.push( output_lit );
		S.addClause( lits );
		lits.clear();

		fgets( temp, MAX_LENGTH, fp );
	}

	fclose( fp );
}
