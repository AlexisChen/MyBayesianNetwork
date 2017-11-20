
#include "BayesianHealper.h"
#include <math.h>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>


BayesianHealper::BayesianHealper()
:mFrequencyOfEachClass(10, 0)
,mClassPriorProbability(10,0)
,mNetworkMatrix(10, std::vector<int>(784, 0))
,mNetworkProbabilityMatrix(10, std::vector<double>(784, 0))
,mClassificationMatrix(10, std::vector<int> (10,0))
{}

BayesianHealper::~BayesianHealper(){}


void BayesianHealper::RunTrainingSet(CAHRVEC trainingImages, std::vector<unsigned char> trainingLabels)
{
  int trainSetSize = 60000;
  int trainImageSize = 784;
  for(int i = 0 ; i < trainSetSize; ++i)
  {
    for (int j = 0; j < trainImageSize ; ++j) {
      int pixelValue = static_cast<int>(trainingImages[i][j]);//0 for black 1 for white
      int label = static_cast<int>(trainingLabels[i]);//correctlabel
      mNetworkMatrix[label][j] += pixelValue;
      mFrequencyOfEachClass[label] ++;
    }
  }
  mNetworkProbabilityMatrix = GetNetworkMatrix();
  mClassPriorProbability = GetClassPriorProbability();
}

double BayesianHealper::GetProbabilityOfDigitGivenTestImg(int c, std::vector<unsigned char> testImage)
{
  int imageSize = 784;
  double probabilitySum = 0;
  for(int i = 0; i < imageSize; ++i)
  {
    int pixelValue = static_cast<int>(testImage[i]);
    double prob = pixelValue == 1? mNetworkProbabilityMatrix[c][i]: (1- mNetworkProbabilityMatrix[c][i] );
    probabilitySum += log(prob);
  }
  probabilitySum += log(mClassPriorProbability[c]);
  return probabilitySum;
}


//populate the classification matrix
void BayesianHealper::RunTestSet(CAHRVEC testImages, std::vector<unsigned char> testLabels)
{

  int testSetSize = 10000;
  for (size_t i = 0; i < testSetSize; ++i) {
    int row = static_cast<int>(testLabels[i]);
    //predict for each testimage
    int maxProbability = -1;
    int mostLikelyDigit = -1;//column
    //for each of the digit calculate probability choose the max.
    for(int j = 0; j < 10; ++j)
    {
      double currProbability = GetProbabilityOfDigitGivenTestImg(j, testImages[i]);
      if(currProbability>maxProbability)
      {
        maxProbability = currProbability;
        mostLikelyDigit = j;
      }
    }
    mClassificationMatrix[row][mostLikelyDigit] += 1;
  }

}

void BayesianHealper::WriteEvaluationBitmap()
{
  int numLabels = 10;
  int numFeatures = 784;
  for (int c=0; c<numLabels; c++) {
    std::vector<unsigned char> classFs(numFeatures);
    for (int f=0; f<numFeatures; f++) {
        //TODO: get probability of pixel f being white given class c
        double p = mNetworkProbabilityMatrix[c][f];
        uint8_t v = 255*p;
        classFs[f] = (unsigned char)v;
    }
    std::stringstream ss;
    ss << "../output/digit" <<c<<".bmp";
    Bitmap::writeBitmap(classFs, 28, 28, ss.str(), false);
  }
}



DOUBLEVEC BayesianHealper::GetNetworkMatrix()
{
  DOUBLEVEC ret;
  int i = 0;
  std::transform( mNetworkMatrix.begin(), mNetworkMatrix.end(), std::back_inserter(ret), [this, &i](const std::vector<int> row){
    ++i;
    std::vector<double> temp;
    int denominator = mFrequencyOfEachClass[i];
    std::transform(row.begin(), row.end(),std::back_inserter(temp),[&denominator](const int& a) {
      return a+1 / denominator+2;
    });
    return temp;
  });
  return ret;
}


std::vector<double> BayesianHealper::GetClassPriorProbability()
{
  std::vector<double> ret;
  std::transform(mFrequencyOfEachClass.begin(), mFrequencyOfEachClass.end(), std::back_inserter(ret), [this](const int& a) {
    return a / mTotalNumberOfSamples;
  });
  return ret;
}

void BayesianHealper::WriteNetworkMatrix()
{
  std::string fileName = "network.txt";
  std::ofstream ofile(fileName);
  if(ofile.is_open())
  {
    int imageSize = 784;
    for(int c = 0; c < 10; ++c)
    {
      for(int p = 0; p < imageSize; ++p)
      {
        ofile << mNetworkProbabilityMatrix[c][p]<<std::endl;
      }
    }
    for (size_t c = 0; c < 10; c++) {
      ofile << mClassPriorProbability[c]<<std::endl;
    }
  }
  ofile.close();
}

void BayesianHealper::WriteClassificationMatrix()
{
  std::string fileName = "classification-summary.txt";
  std::ofstream ofile(fileName);
  if(ofile.is_open()){
    int correctCount = 0;
    for (size_t i = 0; i < 10; i++) {
      for (size_t j = 0; j < 10; j++) {
        ofile<< mClassificationMatrix[i][j]<<" ";
        if(i == j)
        {
          correctCount += mClassificationMatrix[i][j];
        }
      }
      ofile<<std::endl;
    }
    ofile<<"accuracy: "<<correctCount*1.0/100<<"%%"<<std::endl;
  }
  ofile.close();

}
