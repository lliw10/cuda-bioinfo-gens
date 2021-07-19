#include <string>

using std::string;

class ParallelGeneNetworkExecutor: public GeneExecutor {

	public:
		ParallelGeneNetworkExecutor( DataProvider *dataProvider,
				ProgressMonitor *progressMonitor );

		virtual ~ParallelGeneNetworkExecutor();

	protected:

		DataProvider *dataProvider;
		ProgressMonitor *progressMonitor;

		/**
		 * Implementation of GeneExecution::doExecute to make an
		 * execution of gene operation.
		 */
		virtual void doExecute();
};

