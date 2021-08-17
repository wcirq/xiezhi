#include <tchar.h>
#include <stdio.h>
#include <direct.h>
#include <iostream>
#include <Python.h>

int great_function_from_python(int a) {
	int res;
	PyObject *pModule, *pFunc;
	PyObject *pArgs, *pValue;

	/* import */
	pModule = PyImport_Import(PyBytes_FromString("great_module"));

	/* great_module.great_function */
	pFunc = PyObject_GetAttrString(pModule, "great_function");

	/* build args */
	pArgs = PyTuple_New(1);
	// PyObject *args1 = PyUnicode_FromString("../air.jpg");
	PyObject *args1 = PyLong_FromLong(a);

	PyTuple_SetItem(pArgs, 0, args1);

	/* call */
	pValue = PyObject_CallObject(pFunc, pArgs);

	res = PyArg_Parse(pValue, "i", &res);
	return res;
}

int main(int argc, char *argv[])
{
	great_function_from_python(1);
	std::cout << "argv[0]" << argv[0] << std::endl;
	const wchar_t* pName = (wchar_t*)argv[0];
	Py_SetProgramName(pName);
	Py_Initialize();
	PyRun_SimpleString("import numpy as np\n");
	Py_Finalize();
	return 0;
}
