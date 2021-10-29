#ifndef BIR_LOOP_H
#define BIR_LOOP_H

#include <vector>
#include <string>

#include "bir/LoopAxis.h"
#include "bir/Function.h"
#include "bir/BasicBlock.h"
#include "bir/Instruction.h"

namespace bir {

class LoopInst :public Function, public Instruction {
 public:
  LoopInst(const std::string &Name, BasicBlock *Parent)
    : Instruction(Name, Parent, InstructionType::Loop), isParallel(false) {};

  LoopAxis axis;
  bool isParallel;

  void setIsParallel(bool isParallel) {
    this->isParallel = isParallel;
  }
  bool getParallel() {
    return isParallel;  
  }
  void setAxis(LoopAxis &axis) {
    this->axis = axis;
    this->axis.SetLoopInst(this);
  }
  LoopAxis *getAxis() {return &axis;}
}; 

}

#endif
