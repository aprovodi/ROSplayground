#ifndef DEPTH_IMG_H_
#define DEPTH_IMG_H_

#include <iostream>
#include <ros/time.h>

#include <kfusionCPU_benchmark/fluent_interface.h>

namespace dvo_benchmark
{

class DepthImg
{
public:
  DepthImg() {};
  virtual ~DepthImg() {};

  FI_ATTRIBUTE(DepthImg, ros::Time, DepthTimestamp)
  FI_ATTRIBUTE(DepthImg, std::string, DepthFile)

  friend std::ostream& operator <<(std::ostream& out, const DepthImg& img);
  friend std::istream& operator >>(std::istream& in, DepthImg& img);
};

std::ostream& operator <<(std::ostream& out, const DepthImg& img)
{
  out
    << img.DepthTimestamp_ << " "
    << img.DepthFile_ << std::endl;

  return out;
}

std::istream& operator >>(std::istream& in, DepthImg& img)
{
  double timestamp;
  in >> timestamp;
  img.DepthTimestamp_.fromSec(timestamp);
  in >> img.DepthFile_;


  return in;
}

} /* namespace benchmark */
#endif /* DEPTH_IMG_H_ */
