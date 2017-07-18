#ifndef EXECUTIONCONFIG_H_
#define EXECUTIONCONFIG_H_

// GENERATION will only generate the set of kernels in the
// kernels folder. CALCULATION will use the generated kernels
// and run them one by one.
// Use should run the code in GENERATION mode first, and then
// change the mode into CALCULATION whether on FPGA or GPU.
class ExecutionMode {
public:
  enum executionMode {GENERATION, CALCULATION, ALL};
};

// Defines whether we are going to run our code on FPGA or GPU
class TargetDevice {
public:
  enum targetDevice {FPGA, GPU};
};



#endif // EXECUTIONCONFIG_H_
