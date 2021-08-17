#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "cuda_runtime.h"

class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
	{

	}

	~Logger()
	{

	}
	nvinfer1::ILogger& getTRTLogger()
	{
		return *this;
	}

	void log(nvinfer1::ILogger::Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity == Severity::kINFO) return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: " << msg << std::endl; break;
		case Severity::kERROR: std::cerr << "ERROR: " << msg << std::endl; break;
		case Severity::kWARNING: std::cerr << "WARNING: " << msg << std::endl; break;
		case Severity::kINFO: std::cerr << "INFO: " << msg << std::endl; break;
		case Severity::kVERBOSE: break;
			//  default: std::cerr <<"UNKNOW:"<< msg << std::endl;break;
		}
	}
};

class PluginFactory : public nvinfer1::IPluginFactory
{

public:
	PluginFactory();
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
	bool isPlugin(const char* name);
	void destroy();

private:
	static const int m_MaxLeakyLayers = 72;
	static const int m_ReorgStride = 2;
	static constexpr float m_LeakyNegSlope = 0.1f;
	static const int m_NumBoxes = 5;
	static const int m_NumCoords = 4;
	static const int m_NumClasses = 80;
	static const int m_MaxYoloLayers = 3;
	int m_LeakyReLUCount = 0;
	int m_YoloLayerCount = 0;
	nvinfer1::plugin::RegionParameters m_RegionParameters{ m_NumBoxes, m_NumCoords, m_NumClasses,
														  nullptr };

	struct INvPluginDeleter
	{
		void operator()(nvinfer1::plugin::INvPlugin* ptr)
		{
			if (ptr)
			{
				ptr->destroy();
			}
		}
	};
	struct IPluginDeleter
	{
		void operator()(nvinfer1::IPlugin* ptr)
		{
			if (ptr)
			{
				ptr->terminate();
			}
		}
	};
	typedef std::unique_ptr<nvinfer1::plugin::INvPlugin, INvPluginDeleter> unique_ptr_INvPlugin;
	typedef std::unique_ptr<nvinfer1::IPlugin, IPluginDeleter> unique_ptr_IPlugin;

	unique_ptr_INvPlugin m_ReorgLayer;
	unique_ptr_INvPlugin m_RegionLayer;
	unique_ptr_INvPlugin m_LeakyReLULayers[m_MaxLeakyLayers];
	unique_ptr_IPlugin m_YoloLayers[m_MaxYoloLayers];
};

void conv2d();