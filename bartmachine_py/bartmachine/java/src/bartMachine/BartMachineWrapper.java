package bartMachine;

import java.io.Serializable;

/**
 * This class extends bartMachineRegressionMultThread and exposes protected methods as public methods.
 * It is used by the Python implementation to access protected methods that are not accessible through Py4J.
 */
@SuppressWarnings("serial")
public class BartMachineWrapper extends bartMachineRegressionMultThread implements Serializable {
    
    /**
     * Exposes the protected getGibbsSamplesForPrediction method as a public method.
     * 
     * @param records The records to predict on.
     * @param num_cores_evaluate The number of cores to use for evaluation.
     * @return The Gibbs samples for prediction.
     */
    public double[][] getGibbsSamplesForPredictionPublic(double[][] records, int num_cores_evaluate) {
        return getGibbsSamplesForPrediction(records, num_cores_evaluate);
    }
}
