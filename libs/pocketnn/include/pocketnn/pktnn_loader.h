#ifndef PKTNN_LOADER_H
#define PKTNN_LOADER_H

#include <iostream>
#include <fstream>
#include <sstream>

#include "pktnn_mat.h"
#include "pktnn_mat3d.h"

namespace pktnn
{
    class pktloader
    {
    public:
        enum class Dataset
        {
            diabetes,
            mnist
        };
        static void csvLoader(pktmat &saveToMat, std::string fileName);
        // static bool file_exists(const std::string& name);
        // static void downloadDataset(Dataset dataset);
        static void parseDatasetDiabetes(pktmat &saveToMat, std::string fileName);
        static int reverseInt(int i);
        static pktmat3d **loadMnistImages(int numImagesToLoad);
        // load the MNIST dataset
        static void loadMnistImages(pktmat &images, int numImagesToLoad, bool isTrain, bool debugging = false);
        static void loadMnistLabels(pktmat &labels, int numLabelsToLoad, bool isTrain, bool debugging = false);
        // load the FashionMNIST dataset
        static void loadFashionMnistImages(pktmat &images, int numImagesToLoad, bool isTrain);
        static void loadFashionMnistLabels(pktmat &labels, int numLabelsToLoad, bool isTrain);
        // load the timeseries datasets (MIT-BIH and hypnogram)
        static void loadTimeSeriesData(pktmat &dataMat, std::string filename, int numExamplesToLoad, bool debugging = false);
        static void loadTimeSeriesLabels(pktmat &labels, std::string filename, int numLabelsToLoad, bool debugging = false);
    };

    // extern "C" {
    //     void _downloadDataset(pktloader::Dataset dataset);
    // }
}

#endif
