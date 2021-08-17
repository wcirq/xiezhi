#include "trt_test.h"
#include <vector>
#include <assert.h>


PluginFactory::PluginFactory() : m_ReorgLayer{ nullptr }, m_RegionLayer{ nullptr }
{
	for (int i = 0; i < m_MaxLeakyLayers; ++i) m_LeakyReLULayers[i] = nullptr;
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData,
	size_t serialLength)
{
	printf("@@@@@%s", layerName);
	assert(isPlugin(layerName));
	if (std::string(layerName).find("leaky") != std::string::npos)
	{
		assert(m_LeakyReLUCount >= 0 && m_LeakyReLUCount <= m_MaxLeakyLayers);
		assert(m_LeakyReLULayers[m_LeakyReLUCount] == nullptr);
		/*m_LeakyReLULayers[m_LeakyReLUCount]
			= unique_ptr_INvPlugin(nvinfer1::plugin::createPReLUPlugin(serialData, serialLength));*/
		++m_LeakyReLUCount;
		return m_LeakyReLULayers[m_LeakyReLUCount - 1].get();
	}
	else if (std::string(layerName).find("reorg") != std::string::npos)
	{
		assert(m_ReorgLayer == nullptr);
		/*m_ReorgLayer = unique_ptr_INvPlugin(
			nvinfer1::plugin::createYOLOReorgPlugin(serialData, serialLength));*/
		return m_ReorgLayer.get();
	}
	else if (std::string(layerName).find("region") != std::string::npos)
	{
		assert(m_RegionLayer == nullptr);
		/*m_RegionLayer = unique_ptr_INvPlugin(
			 nvinfer1::plugin::createYOLORegionPlugin(serialData, serialLength));*/
		return m_RegionLayer.get();
	}
	else if (std::string(layerName).find("yolo") != std::string::npos)
	{
		/*assert(m_YoloLayerCount >= 0 && m_YoloLayerCount < m_MaxYoloLayers);
		assert(m_YoloLayers[m_YoloLayerCount] == nullptr);
		m_YoloLayers[m_YoloLayerCount]
			= unique_ptr_IPlugin(new YoloLayerV3(serialData, serialLength));
		++m_YoloLayerCount;
		return m_YoloLayers[m_YoloLayerCount - 1].get();*/
	}
	else
	{
		std::cerr << "ERROR: Unrecognised layer : " << layerName << std::endl;
		assert(0);
		return nullptr;
	}
}

bool PluginFactory::isPlugin(const char* name)
{
	return ((std::string(name).find("leaky") != std::string::npos)
		|| (std::string(name).find("reorg") != std::string::npos)
		|| (std::string(name).find("region") != std::string::npos)
		|| (std::string(name).find("yolo") != std::string::npos));
}

void PluginFactory::destroy()
{
	m_ReorgLayer.reset();
	m_RegionLayer.reset();

	for (int i = 0; i < m_MaxLeakyLayers; ++i)
	{
		m_LeakyReLULayers[i].reset();
	}

	for (int i = 0; i < m_MaxYoloLayers; ++i)
	{
		m_YoloLayers[i].reset();
	}

	m_LeakyReLUCount = 0;
	m_YoloLayerCount = 0;
}

nvinfer1::ICudaEngine* loadTRTEngine(const std::string planFilePath, PluginFactory* pluginFactory,
	Logger& logger)
{
	// reading the model in memory
	std::cout << "Loading TRT Engine..." << std::endl;
	assert(fileExists(planFilePath));
	std::stringstream trtModelStream;
	trtModelStream.seekg(0, trtModelStream.beg);
	std::ifstream cache(planFilePath, std::ios::binary | std::ios::in);
	assert(cache.good());
	trtModelStream << cache.rdbuf();
	cache.close();

	// calculating model size
	trtModelStream.seekg(0, std::ios::end);
	const auto modelSize = trtModelStream.tellg();
	trtModelStream.seekg(0, std::ios::beg);
	void* modelMem = malloc(modelSize);
	trtModelStream.read((char*)modelMem, modelSize);

	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine
		= runtime->deserializeCudaEngine(modelMem, modelSize, pluginFactory);
	free(modelMem);
	runtime->destroy();
	std::cout << "Loading Complete!" << std::endl;

	return engine;
}


void writePlanFileToDisk(nvinfer1::ICudaEngine* m_Engine, std::string m_EnginePath)
{
	std::cout << "Serializing the TensorRT Engine..." << std::endl;
	assert(m_Engine && "Invalid TensorRT Engine");
	nvinfer1::IHostMemory* m_ModelStream = m_Engine->serialize();
	assert(m_ModelStream && "Unable to serialize engine");
	assert(!m_EnginePath.empty() && "Enginepath is empty");

	// write data to output file
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);
	gieModelStream.write(static_cast<const char*>(m_ModelStream->data()), m_ModelStream->size());
	std::ofstream outFile;
	outFile.open(m_EnginePath, std::ios::binary | std::ios::out);
	outFile << gieModelStream.rdbuf();
	outFile.close();

	std::cout << "Serialized plan file cached at location : " << m_EnginePath << std::endl;
}


void conv2d() {
	std::string m_EnginePath = "./model.engine";
	Logger m_Logger;
	std::vector<nvinfer1::Weights> trtWeights;

	nvinfer1::IBuilder* m_Builder = nvinfer1::createInferBuilder(m_Logger);;
	nvinfer1::IBuilderConfig* config = m_Builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* m_Network = m_Builder->createNetworkV2(0U);

	// 定义输入层
	std::string inputName = "inputs";
	std::string outputName = "outputs";
	int m_BatchSize = 1;
	int m_InputC = 1;
	int m_InputH = 4;
	int m_InputW = 4;
	int m_InputSize = m_InputC * m_InputH * m_InputW;

	nvinfer1::ITensor* data = m_Network->addInput(
		inputName.c_str(),
		nvinfer1::DataType::kFLOAT,
		nvinfer1::DimsCHW{ static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW) });
	assert(data != nullptr);

	// 进行归一化操作，下面定shape与输入一致的，value全为255的矩阵
	nvinfer1::Dims divDims{3, 
		{ static_cast<int>(m_InputC), static_cast<int>(m_InputH), static_cast<int>(m_InputW) },
		{ nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kSPATIAL } };
	nvinfer1::Weights divWeights{ nvinfer1::DataType::kFLOAT, nullptr, static_cast<int64_t>(m_InputSize) };
	float* divWt = new float[m_InputSize];
	for (uint32_t w = 0; w < m_InputSize; ++w) divWt[w] = 1.0;
	divWeights.values = divWt;
	trtWeights.push_back(divWeights);
	nvinfer1::IConstantLayer* constDivide = m_Network->addConstant(divDims, divWeights);
	assert(constDivide != nullptr);

	// 进行归一化操作，即除以255， addElementWise方法第三个参数指定了运算方式为 “除法”
	nvinfer1::IElementWiseLayer* elementDivide = m_Network->addElementWise(
		*data,
		*constDivide->getOutput(0), 
		nvinfer1::ElementWiseOperation::kDIV);
	assert(elementDivide != nullptr);

	nvinfer1::ITensor* inputs = elementDivide->getOutput(0);

	int nbOutputMaps = 3; // 输出特征矩阵数
	int kernelSizeHW = 2; // 卷积核尺寸
	int pad = 0; // 
	int stride = 1; // 
	int group = 1; // 
	int kernelSize = kernelSizeHW * kernelSizeHW * nbOutputMaps * m_InputC;
	int biasSize = nbOutputMaps;
	int outputW = (m_InputW - kernelSizeHW + 2 * pad) / stride + 1;
	int outputH = (m_InputH - kernelSizeHW + 2 * pad) / stride + 1;
	int outputSize = outputW * outputH * m_InputC;

	nvinfer1::Weights kernelWeights{ nvinfer1::DataType::kFLOAT, nullptr, kernelSize };
	nvinfer1::Weights biasWeights{ nvinfer1::DataType::kFLOAT, nullptr, 0 };

	float *convKernel = new float[kernelSize];
	for (size_t i = 0; i < kernelSize; i++) convKernel[i] = i%3;
	kernelWeights.values = convKernel;

	/*
	float *convBias = new float[biasSize];
	for (size_t i = 0; i < biasSize; i++) convBias[i] = 0;
	biasWeights.values = convBias;
	biasWeights.count = biasSize;
	*/

	nvinfer1::IConvolutionLayer* conv = m_Network->addConvolutionNd(
		*inputs,
		nbOutputMaps,
		nvinfer1::DimsHW{ kernelSizeHW, kernelSizeHW },
		kernelWeights,
		biasWeights);
	assert(conv != nullptr);
	conv->setPaddingNd(nvinfer1::DimsHW{ pad,pad });
	conv->setStrideNd(nvinfer1::DimsHW{ stride ,stride });
	conv->setNbGroups(group);
	conv->setName("conv2d");
	nvinfer1::ITensor* outputs = conv->getOutput(0);

	outputs->setName(outputName.c_str());
	m_Network->markOutput(*outputs);

	m_Builder->setMaxBatchSize(m_BatchSize);
	config->setMaxWorkspaceSize(1 << 20);

	m_Builder->allowGPUFallback(true);

	nvinfer1::ICudaEngine* m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *config);
	assert(m_Engine != nullptr);
	// Serialize the engine
	writePlanFileToDisk(m_Engine, m_EnginePath);

	// Load the engine
	PluginFactory *m_PluginFactory = new PluginFactory();
	nvinfer1::ICudaEngine* m_ReEngine = loadTRTEngine(m_EnginePath, m_PluginFactory, m_Logger);

	nvinfer1::IExecutionContext* m_Context = m_ReEngine->createExecutionContext();
	assert(m_Context != nullptr);
	int m_InputBindingIndex = m_ReEngine->getBindingIndex(inputName.c_str());
	assert(m_InputBindingIndex != -1);
	assert(m_BatchSize <= static_cast<uint32_t>(m_ReEngine->getMaxBatchSize()));

	std::vector<void*> m_DeviceBuffers;
	m_DeviceBuffers.resize(m_ReEngine->getNbBindings(), nullptr);
	assert(m_InputBindingIndex != -1);

	int state;
	state = cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndex), m_BatchSize * m_InputSize * sizeof(float));

	int tensorVolume = outputW * outputH * nbOutputMaps;

	int tensorBindingIndex = m_ReEngine->getBindingIndex(outputName.c_str());
	float* tensorHostBuffer{ nullptr };
	cudaStream_t m_CudaStream = nullptr;

	assert(tensorBindingIndex != -1);
	state = cudaMalloc(&m_DeviceBuffers.at(tensorBindingIndex), m_BatchSize * tensorVolume * sizeof(float));
	state = cudaMallocHost(&tensorHostBuffer, tensorVolume * m_BatchSize * sizeof(float));
	state = cudaStreamCreate(&m_CudaStream);

	float* images = new float[m_InputSize];;
	for (size_t i = 0; i < m_InputSize; i++) images[i] = i%5;

	// 推理
	state = cudaMemcpyAsync(m_DeviceBuffers.at(m_InputBindingIndex), images,
		m_BatchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice,
		m_CudaStream);

	m_Context->enqueue(m_BatchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
	cudaMemcpyAsync(tensorHostBuffer, m_DeviceBuffers.at(tensorBindingIndex),
		m_BatchSize * tensorVolume * sizeof(float),
		cudaMemcpyDeviceToHost, m_CudaStream); 
	cudaStreamSynchronize(m_CudaStream);

	// 打印输入输出层
	printf("\nimages: \n");
	for (int i = 0; i < m_InputSize; i++)
	{
		printf("%2.0f", images[i]);
		if ((i + 1) % m_InputW == 0) printf("\n");
		if ((i + 1) % (m_InputW*m_InputH) == 0) printf("\n");
	}

	printf("\nconvKernel: \n");
	for (int i = 0; i < kernelSize; i++)
	{
		printf("%2.0f", convKernel[i]);
		if ((i + 1) % kernelSizeHW == 0) printf("\n");
		if ((i + 1) % (kernelSizeHW*kernelSizeHW) == 0) printf("\n");
	}

	printf("\noutputs: \n");
	for (int i = 0; i < tensorVolume * m_BatchSize; i++)
	{
		std::cout << tensorHostBuffer[i] << " ";
		if ((i + 1) % outputW == 0) printf("\n");
		if ((i + 1) % (outputW*outputH) == 0) printf("\n");
	}

	std::cout << "" << std::endl;
}