#include <string>

using std::string;

class GeneExecutor {

	public:
		GeneExecutor( string _name );

		virtual ~GeneExecutor();

		/**
		 * Execute the base of an operation. First, read resources and build
		 * context for an task execution. After execute a specific task calling
		 * doExecute. Next, finalize cleaning up resources and
		 * memory allocation.
		 */
		void execute();

		string getName();

	protected:

		string name;

		/**
		 * Execution of gene specific algorithm.
		 */
		virtual void doExecute();

};
