/* includes //{ */

#include <ros/ros.h>
#include <ros/package.h>
#include <nodelet/nodelet.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Header.h>
#include <vision_msgs/Detection2DArray.h>

#include <hailort_yolo_common/detection_inference.hpp>
#include <hailort_yolo_common/utils.hpp>
#include <hailort_yolo_common/coco_names.hpp>
#include <hailo/hailort.hpp>

//}

namespace mrs_yolo_detector
{

/* class MrsYoloDetector //{ */

class MrsYoloDetector : public nodelet::Nodelet {

public:
  virtual void onInit();

private:
  void                                 imageCallback(const sensor_msgs::ImageConstPtr &image_msg);
  static vision_msgs::Detection2DArray objects_to_detection2d(const std::vector<yolo_cpp::Object> &objects, const std_msgs::Header &header);

  std::atomic<bool> is_initialized_ = false;

  cv_bridge::CvImageConstPtr last_cam_image_ptr_;

  std::unique_ptr<yolo_cpp::YoloHailoRT> yolo_;
  std::vector<std::string>               class_names_;
  std::string                            model_;
  double                                 confidence_;
  double                                 nms_;
  int                                    fps_;

  image_transport::Subscriber sub_image_;
  image_transport::Publisher  pub_image_;

  ros::Publisher pub_detection2d_;
};

//}

/* onInit() //{ */

void MrsYoloDetector::onInit() {

  ROS_INFO("[MrsYoloDetector] Initializing.");

  ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

  ros::Time::waitForValid();

  ROS_INFO("[MrsYoloDetector] Subscribing camera.");

  image_transport::ImageTransport it(nh);
  sub_image_ = it.subscribe("camera_input", 1, &MrsYoloDetector::imageCallback, this);
  // sub_image_ = nh.subscribe("camera_input", 1, &MrsYoloDetector::imageCallback, this, ros::TransportHints().tcpNoDelay());

  pub_image_ = it.advertise("img_output", 1);
  // pub_image_ = nh.advertise<sensor_msgs::Image>("img_output", 1);
  //
  pub_detection2d_ = nh.advertise<vision_msgs::Detection2DArray>("detections", 1);

  ROS_INFO("[MrsYoloDetector] Loading parameters.");

  nh.param<std::string>("yolo_parameters/model/default_value", model_, "yolov10s.hef");
  nh.param<double>("yolo_parameters/confidence/default_value", confidence_, 0.3);
  nh.param<double>("yolo_parameters/nms/default_value", nms_, 0.45);
  nh.param<int>("control/fps", fps_, 1);

  std::string model_path = ros::package::getPath("mrs_yolo_detector") + "/hef/" + model_;

  ROS_INFO("[MrsYoloDetector] Loading model path: %s.", model_path.c_str());

  class_names_ = yolo_cpp::COCO_CLASSES;

  yolo_ = std::make_unique<yolo_cpp::YoloHailoRT>(model_path, confidence_, nms_);

  ROS_INFO("[MrsYoloDetector] Model loaded.");

  is_initialized_ = true;
}

//}

// | ---------------------- msg callbacks --------------------- |


/* imageCallback() //{ */

void MrsYoloDetector::imageCallback(const sensor_msgs::ImageConstPtr &image_msg) {

  if (!is_initialized_) {
    return;
  }

  auto img = cv_bridge::toCvCopy(image_msg, image_msg->encoding);

  bool converted_to_rgb8 = false;

  cv::Mat frame;

  if (img->image.channels() == 4) {

    cv::cvtColor(img->image, frame, cv::COLOR_BGRA2RGB);

    converted_to_rgb8 = true;

  } else {

    frame = img->image;
  }

  ROS_INFO("[MrsYoloDetector]: inference started");

  ros::Time now = ros::Time::now();

  auto objects = this->yolo_->inference(frame);

  auto end = std::chrono::system_clock::now();

  ROS_INFO("[MrsYoloDetector]: inference finished, took %.3f s", (ros::Time::now() - now).toSec());

  yolo_cpp::utils::draw_objects(frame, objects, this->class_names_);

  vision_msgs::Detection2DArray detections = objects_to_detection2d(objects, img->header);
  pub_detection2d_.publish(detections);

  auto pub_img = cv_bridge::CvImage(img->header, converted_to_rgb8 ? "rgb8" : image_msg->encoding, frame).toImageMsg();

  this->pub_image_.publish(pub_img);
}

//}

/* objects_to_detection2da() //{ */

vision_msgs::Detection2DArray MrsYoloDetector::objects_to_detection2d(const std::vector<yolo_cpp::Object> &objects, const std_msgs::Header &header) {

  vision_msgs::Detection2DArray detection2d;

  detection2d.header = header;

  for (const auto &obj : objects) {

    vision_msgs::Detection2D det;

    det.bbox.center.x = obj.rect.x + obj.rect.width / 2;
    det.bbox.center.y = obj.rect.y + obj.rect.height / 2;
    det.bbox.size_x   = obj.rect.width;
    det.bbox.size_y   = obj.rect.height;

    det.results.resize(1);
    det.results[0].id    = obj.label;
    det.results[0].score = obj.prob;
    detection2d.detections.emplace_back(det);
  }

  return detection2d;
}

//}

}  // namespace mrs_yolo_detector

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mrs_yolo_detector::MrsYoloDetector, nodelet::Nodelet);
