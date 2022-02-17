# executorch
Simple and portable executor of PyTorch programs

Design goals:
* Minimal binary size (< 50KB not including kernels)
* Minimal framework tax: loading program, initializing executor, kernel and backend-delegate dispatch, runtime * memory utilization
* Portable (cross-compile across many toolchains)
* Executes ATen kernels (or ATen custom kernels)
* Executes custom op kernels
* Supports inter op asynchronous execution
* Supports static memory allocation (heapless)
* Supports custom allocation across memory hierarchies
* Supports control flow needed by models
* Allows selective build of kernels
* Allows backend delegation with lightweight interface
