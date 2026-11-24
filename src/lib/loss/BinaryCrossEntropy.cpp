#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}
double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
	const uint64_t n = predictedOutput.size();
	double costvalue = 0.0;
	for (uint64_t i = 0; i < n; ++i) {

		const double clap = clampProbability(predictedOutput[i]);

		const double a = std::log(clap);

		const double c = std::log(1.0 - clap);

		const double b = (1 - actualOutput[i]);

		costvalue = costvalue - ((actualOutput[i] * a + b * c) / n);
	}
	return costvalue;
}
double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
	const double n = predictedOutput.size();
	double correct = 0.0;

		for (uint64_t i = 0;i < n;++i) {

			double zwi = predictedOutput[i] > 0.5 ? 1.0 : 0.0;

			if (zwi == actualOutput[i]) {

				correct = correct + 1;
			}
		}
	return correct / n;
}
std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
	const double n = predictedOutput.size();

	std::vector<double> gradient(n, 0.0);

	for (uint64_t i = 0;i < n;++i) {

		gradient[i] = gradient[i] - ((actualOutput[i] / clampProbability(predictedOutput[i])) - (1.0 - actualOutput[i]) / (1.0 - clampProbability(predictedOutput[i]))) / n;
	} 
	return gradient;
}



}
