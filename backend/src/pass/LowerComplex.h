#ifndef PASS_LOWERCOMPLEX_H
#define PASS_LOWERCOMPLEX_H

#include "pass/pass.h"
#include "bir/Instruction.h"
#include "bir/AffineExpr.h"
#include "Arch.h"

float ceil_div(int x, int y);
float floor_div(int x, int y);

class LowerComplex : public Pass {
 public:
  LowerComplex() {}
  virtual ~LowerComplex() {}
  virtual int run(bir::Module &module) override;

  void LowerConv2d(bir::Conv2dInst *I, Arch *arch);
  void FindTilingFactorConv2D(bir::Conv2dInst *I, Arch *A, std::unordered_map<std::string, std::vector<int>> &tiling);
  void LowerConv2dToInstruction(bir::Conv2dInst *I, std::vector<bir::LoopAxis*>& loopnest, bir::loopmap& values, int level);

  void LowerConv2ddw(bir::Conv2ddwInst *I, Arch *arch);
  void FindTilingFactorConv2DDW(bir::Conv2ddwInst *I, Arch *A, std::unordered_map<std::string, std::vector<int>> &tiling);
  void LowerConv2ddwToInstruction(bir::Conv2ddwInst *I, std::vector<bir::LoopAxis*>& loopnest, bir::loopmap& values, int level);
  int cnt = 0;
};

#endif
