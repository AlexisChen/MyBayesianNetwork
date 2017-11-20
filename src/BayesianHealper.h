#ifndef _BAYESIAN_HEALPER_H_
#define _BAYESIAN_HEALPER_H_

#include <vector>
#include "bitmap.hpp"

using INTVEC = std::vector<std::vector<int>>;
using DOUBLEVEC = std::vector<std::vector<double>>;
using CAHRVEC = std::vector<std::vector<unsigned char>>;
class BayesianHealper
{
public:
  BayesianHealper();
  ~BayesianHealper();
  void RunTrainingSet(CAHRVEC trainingImages, std::vector<unsigned char> trainingLabels);
  void WriteEvaluationBitmap();
  double GetProbabilityOfDigitGivenTestImg(int c, std::vector<unsigned char> testImage);
  void RunTestSet(CAHRVEC testImages, std::vector<unsigned char> testLabels);
  DOUBLEVEC GetNetworkMatrix(); //return 10 by 10 matrix of matching probability
  INTVEC GetClassificationMatrix(){return mNetworkMatrix;}
  std::vector<double> GetClassPriorProbability();

  void WriteNetworkMatrix();
  void WriteClassificationMatrix();

private:
  int mTotalNumberOfSamples = 60000;

  std::vector<int> mFrequencyOfEachClass;
  std::vector<double> mClassPriorProbability;
  //10 by 768 int matrix to store the count of count of feature given class

  INTVEC mNetworkMatrix;
  DOUBLEVEC mNetworkProbabilityMatrix;

  //10 by 10 int matrix to store the the count of matches
  //populated at the end of training.
  INTVEC mClassificationMatrix;

};
#endif
