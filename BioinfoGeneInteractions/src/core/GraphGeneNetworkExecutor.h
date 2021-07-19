#include <string>

using std::string;

class GraphGeneNetworkExecutor: public GeneExecutor {

	public:
		GraphGeneNetworkExecutor( DataProvider *dataProvider,
				ProgressMonitor *progressMonitor );

		virtual ~GraphGeneNetworkExecutor();

	protected:
		DataProvider *dataProvider;
		ProgressMonitor *progressMonitor;

		/**
		 * Implementation of GraphGeneNetworkExecutor::doExecute to make an
		 * execution of gene operation based on graph algorithm.
		 */
		virtual void doExecute();

};

