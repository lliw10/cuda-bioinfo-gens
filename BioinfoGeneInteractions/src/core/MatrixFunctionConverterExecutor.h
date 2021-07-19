#include <string>

using std::string;

class MatrixFunctionConverterExecutor: public GeneExecutor {

	public:
		MatrixFunctionConverterExecutor( DataProvider *dataProvider );

		virtual ~MatrixFunctionConverterExecutor();

	protected:
		DataProvider *dataProvider;

		/**
		 * Implementation of GeneExecution::doExecute to make an
		 * execution of gene operation.
		 */
		void virtual doExecute();

};

