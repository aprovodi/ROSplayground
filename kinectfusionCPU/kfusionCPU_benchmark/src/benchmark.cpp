#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>  // for sensor_msgs::Image
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <kfusionCPU_benchmark/file_reader.h>
#include <kfusionCPU_benchmark/depth_img.h>
#include <kfusionCPU_benchmark/groundtruth.h>
#include <kfusionCPU_benchmark/tools.h>

#include "kfusionCPU/kfusionCPU.h"
#include "kfusionCPU/tsdfVolume.h"

#include "kfusionCPU_benchmark/SceneCloudView.h"


static const char WINDOW[] = "Image window";

/**
 * Converts the given raw depth image (type CV_16UC1) to a CV_32FC1 image rescaling every pixel with the given scale
 * and replacing 0 with NaNs.
 */
void convertRawDepthImage(const cv::Mat& input, cv::Mat& output, float scale)
{
  output.create(input.rows, input.cols, CV_32FC1);

  const unsigned short* input_ptr = input.ptr<unsigned short>();
  float* output_ptr = output.ptr<float>();

  for(int idx = 0; idx < input.size().area(); idx++, input_ptr++, output_ptr++)
  {
    if(*input_ptr == 0)
    {
      *output_ptr = std::numeric_limits<float>::quiet_NaN();
    }
    else
    {
      *output_ptr = ((float) *input_ptr) * scale;
    }
  }
}
/*
void convertRawDepthImageSse(const cv::Mat& input, cv::Mat& output, float scale)
{
  output.create(input.rows, input.cols, CV_32FC1);

  const unsigned short* input_ptr = input.ptr<unsigned short>();
  float* output_ptr = output.ptr<float>();

  __m128 _scale = _mm_set1_ps(scale);
  __m128 _zero  = _mm_setzero_ps();
  __m128 _nan   = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());

  for(int idx = 0; idx < input.size().area(); idx += 8, input_ptr += 8, output_ptr += 8)
  {
    __m128 _input, mask;
    __m128i _inputi = _mm_load_si128((__m128i*) input_ptr);

    // load low shorts and convert to float
    _input = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_inputi, _mm_setzero_si128()));

    mask = _mm_cmpeq_ps(_input, _zero);

    // zero to nan
    _input = _mm_or_ps(_input, _mm_and_ps(mask, _nan));
    // scale
    _input = _mm_mul_ps(_input, _scale);
    // save
    _mm_store_ps(output_ptr + 0, _input);

    // load high shorts and convert to float
    _input = _mm_cvtepi32_ps(_mm_unpackhi_epi16(_inputi, _mm_setzero_si128()));

    mask = _mm_cmpeq_ps(_input, _zero);

    // zero to nan
    _input = _mm_or_ps(_input, _mm_and_ps(mask, _nan));
    // scale
    _input = _mm_mul_ps(_input, _scale);
    // save
    _mm_store_ps(output_ptr + 4, _input);
  }
}
*/

cv::Mat load(std::string rgb_file, std::string depth_file)
{
  cv::Mat rgb, grey, grey_s16, depth, depth_inpainted, depth_mask, depth_mono, depth_float;

  bool rgb_available = false;
  //rgb = cv::imread(rgb_file, 1);
  depth = cv::imread(depth_file, -1);

  if(rgb.type() != CV_32FC1)
  {
    if(rgb.type() == CV_8UC3)
    {
      cv::cvtColor(rgb, grey, CV_BGR2GRAY);
      rgb_available = true;
    }
    else
    {
      grey = rgb;
    }

    grey.convertTo(grey_s16, CV_32F);
  }
  else
  {
    grey_s16 = rgb;
  }

  if(depth.type() != CV_32FC1)
  {
    convertRawDepthImage(depth, depth_float, 1.0f / 5000.0f);
  }
  else
  {
    depth_float = depth;
    depth_float *= 0.005f;
  }

  return depth_float;
}

sensor_msgs::ImagePtr map_to_image(const cv::Mat& map, std::string& encoding) {
    cv_bridge::CvImagePtr cv_image(new cv_bridge::CvImage);
    cv_image->image = map;
    cv_image->encoding = encoding;
    return cv_image->toImageMsg();
}

class BenchmarkNode
{
public:
  struct Config
  {
    bool EstimateTrajectory;
    std::string TrajectoryFile;

    std::string RgbdPairFile;
    std::string GroundtruthFile;

    bool ShowGroundtruth;
    bool ShowEstimate;

    bool EstimateRequired();
    bool VisualizationRequired();
  };

  BenchmarkNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
  ~BenchmarkNode()  {      cv::destroyWindow(WINDOW);  }

  bool configure();

  void run();

private:
  ros::NodeHandle& nh_, nh_private_;
  ros::Publisher img_pub_;
  Config cfg_;
  cvpr_tum::kfusionCPU kfusionCPU_;
  SceneCloudView scene_cloud_view_;

  std::ostream *trajectory_out_;
  dvo_benchmark::FileReader<dvo_benchmark::Groundtruth> *groundtruth_reader_;
  dvo_benchmark::FileReader<dvo_benchmark::DepthImg> *rgbdpair_reader_;

};

BenchmarkNode::BenchmarkNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private) :
    nh_(nh),
    nh_private_(nh_private),
    trajectory_out_(0),
    groundtruth_reader_(0),
    rgbdpair_reader_(0),
    scene_cloud_view_()
{
    // init Kfusion
    Eigen::Vector3f volume_size = Eigen::Vector3f::Constant(4.0/*meters*/);
    kfusionCPU_.volume().setSize(volume_size);

    kfusionCPU_.volume().setPositiveTsdfTruncDist(0.020f/*meters*/);
    kfusionCPU_.volume().setNegativeTsdfTruncDist(-0.020f/*meters*/);

    uint32_t queue_size = 1;
    std::string topic_dimage = nh_.resolveName("/kfusionCPU/dimage");
    img_pub_ = nh_.advertise<sensor_msgs::Image> (topic_dimage, 1);

    cv::namedWindow(WINDOW);
}

bool BenchmarkNode::configure()
{
  // dataset files related stuff
  if(nh_private_.getParam("rgbdpair_file", cfg_.RgbdPairFile))
  {
    rgbdpair_reader_ = new dvo_benchmark::FileReader<dvo_benchmark::DepthImg>(cfg_.RgbdPairFile);
    rgbdpair_reader_->skipComments();

    if(!rgbdpair_reader_->next())
    {
      std::cerr << "Failed to open '" << cfg_.RgbdPairFile << "'!" << std::endl;
      return false;
    }
  }
  else
  {
    std::cerr << "Missing 'rgbdpair_file' parameter!" << std::endl;
    return false;
  }

  // ground truth related stuff
  nh_private_.param("show_groundtruth", cfg_.ShowGroundtruth, false);
  if(cfg_.ShowGroundtruth)
  {
    if(nh_private_.getParam("groundtruth_file", cfg_.GroundtruthFile))
    {
      groundtruth_reader_ = new dvo_benchmark::FileReader<dvo_benchmark::Groundtruth>(cfg_.GroundtruthFile);
      groundtruth_reader_->skipComments();

      if(!groundtruth_reader_->next())
      {
        std::cerr << "Failed to open '" << cfg_.GroundtruthFile << "'!" << std::endl;
        return false;
      }
    }
    else
    {
      std::cerr << "Missing 'groundtruth_file' parameter!" << std::endl;
      return false;
    }
  }

  return true;
}

void BenchmarkNode::run()
{
  // setup camera parameters
//  dvo::core::IntrinsicMatrix intrinsics = dvo::core::IntrinsicMatrix::create(525.0f, 525.0f, 320.0f, 240.0f);

  // initialize first pose
  Eigen::Affine3f trajectory;

  if(groundtruth_reader_ != 0)
  {
    dvo_benchmark::findClosestEntry(*groundtruth_reader_, rgbdpair_reader_->entry().DepthTimestamp());
    dvo_benchmark::toPoseEigen(groundtruth_reader_->entry(), trajectory);
  }
  else
  {
    trajectory.setIdentity();
  }

  Eigen::Vector3f tcam_shift(kfusionCPU_.volume().getSize()(2) / 2.f, kfusionCPU_.volume().getSize()(2) / 2.f, 0.f);
  trajectory.translation() += tcam_shift;

  kfusionCPU_.setInitalCameraPose(trajectory);

  std::string folder = cfg_.RgbdPairFile.substr(0, cfg_.RgbdPairFile.find_last_of("/") + 1);

  std::vector<dvo_benchmark::DepthImg> pairs;
  rgbdpair_reader_->readAllEntries(pairs);

  //std::cout << "number of images read: " << pairs.size() << std::endl;

  cv::Mat current;

  for(std::vector<dvo_benchmark::DepthImg>::iterator it = pairs.begin(); ros::ok() && it != pairs.end(); ++it)
  {
    current = load("", folder + it->DepthFile());

    //img_pub_.publish(map_to_image(current, "32FC1"));
    cv::imshow(WINDOW, current);
    cv::waitKey(3);

    kfusionCPU_(current);

    Eigen::Affine3d transform = kfusionCPU_.getCameraPose().cast<double>();
    //transform.translation() += world_size_ / 2;

    scene_cloud_view_.addCoordinateSystem(transform, false);
    scene_cloud_view_.updateCamPCloud(transform.translation());

    scene_cloud_view_.renderScene(kfusionCPU_.volume(), true);

/*

    if(cfg_.ShowGroundtruth)
    {
      Eigen::Affine3d groundtruth_pose;

//      dvo_benchmark::findClosestEntry(*groundtruth_reader_, it->RgbTimestamp());
//      dvo_benchmark::toPoseEigen(groundtruth_reader_->entry(), groundtruth_pose);

      tf::Transform tmp_tf;

      tf::poseMsgToTF(pose->pose.pose, tmp_tf);
      tf::TransformTFToEigen(tmp_tf, cam_pose_);
      //cam_pose_.translation() += world_size_ / 2;

      scene_cloud_view_.addCoordinateSystem(cam_pose_, false);
      scene_cloud_view_.updateCamPCloud(cam_pose_.translation());

      scene_cloud_view_.renderScene(tsdf_volume_, true);
*/
/*
      visualizer->trajectory("groundtruth")->
          color(dvo::visualization::Color::green()).
          add(groundtruth_pose);

      visualizer->camera("groundtruth")->
          color(dvo::visualization::Color::green()).
          update(current->level(0), intrinsics, groundtruth_pose).
          show(cfg_.ShowEstimate ? dvo::visualization::CameraVisualizer::ShowCamera : dvo::visualization::CameraVisualizer::ShowCameraAndCloud);

    }
    */

  }

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "benchmark", ros::init_options::AnonymousName);

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  BenchmarkNode benchmark(nh, nh_private);

  if(benchmark.configure())
  {
    benchmark.run();
  }

  return 0;
}
