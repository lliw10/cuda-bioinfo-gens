#include <string>

using std::string;

class GeneNetworkAnalisys: public GeneExecutor {

	public:
		GeneNetworkAnalisys( DataProvider *_dataProvider );

		virtual ~GeneNetworkAnalisys();

	protected:

		DataProvider *dataProvider;

		/**
		 * Implementation of GeneExecution::doExecute to make an
		 * execution of gene operation.
		 */
		void virtual doExecute();

};

