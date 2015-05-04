#include <ros/package.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <tr1/memory>
#include "cv.h"
#include <sstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/norms.h>
#include <pcl/features/normal_3d.h> 
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/common/common.h>

#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <sys/types.h>
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>
#include <string.h>
#include "std_msgs/String.h"

// ROS
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>

#include <object_detection/DetectObject.h>
#include <pose_estimation/PoseEstimation.h>

typedef pose_estimation::PoseEstimation PoseEstimation;
typedef object_detection::DetectObject MsgT;

using namespace std;
using namespace tr1;
using namespace message_filters;
using namespace sensor_msgs;
using namespace tf;
using namespace pcl_ros;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointXYZRGB PointA;
typedef pcl::PointCloud<PointT> CloudT;

MsgT msgglobal;
MsgT keep_latest_best_pose;

PoseEstimation poseEstimationMsgT;

ros::Publisher publish_for_vizualizer;

vector<image_transport::SubscriberFilter *> subscribterVector;

vector<ros::Subscriber*> point_cloud_vector;
vector<ros::Subscriber*> stereo_point_cloud_vector;

image_transport::SubscriberFilter* sub_temp_1;
image_transport::SubscriberFilter* sub_temp_2;

vector<message_filters::TimeSynchronizer<Image, Image> *> timeSync;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicyT;
vector<message_filters::Synchronizer<SyncPolicyT> *> timeSyncApprox;

pcl::PointCloud<PointT> stereo_pointCloud;
pcl::PointCloud<PointT> carmine_pointCloud;

//ros::ServiceClient pose_estimation_service_client;
ros::ServiceClient getPose;
string object_path;

void stereoPointCloudSaver(ros::NodeHandle nh, string name);

void stereoPointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& PointCloudROS){
	pcl::fromROSMsg<PointT>(*PointCloudROS, stereo_pointCloud);
	//std::cout << "stereo_pointCloud.size(): " << stereo_pointCloud.size() << std::endl;
}

void kinectPointCloudSaver(ros::NodeHandle, string name);

void kinectPointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& PointCloudROS){
	pcl::fromROSMsg<PointT>(*PointCloudROS, carmine_pointCloud);
}

void stereoPointCloudSaver(ros::NodeHandle nh, string name){

	std::stringstream PointCloudPath;
	PointCloudPath << name << "/points";

	stereo_point_cloud_vector.push_back(new ros::Subscriber());
	*stereo_point_cloud_vector.back() = nh.subscribe (PointCloudPath.str(), 1, stereoPointCloudCallback);
}

void kinectPointCloudSaver(ros::NodeHandle nh, string name){
	std::stringstream PointCloudPath;
	PointCloudPath << name << "/depth_registered/points";


	point_cloud_vector.push_back(new ros::Subscriber());
	*point_cloud_vector.back() = nh.subscribe (PointCloudPath.str(), 1, kinectPointCloudCallback);
}

pcl::PointCloud<PointT> cutConveyourBelt(pcl::PointCloud<PointA>::Ptr object, pcl::PointCloud<PointT> scene, MsgT msgglobal1){
    //pcl::io::savePCDFile("conveyour01.pcd", *object);
	tf::Transform transform;
	tf::transformMsgToTF(msgglobal1.response.poses[0], transform);

	Eigen::Matrix4f m_init, m;
	transformAsMatrix(transform, m_init);
	m_init(12) = m_init(12)/1000;
	m_init(13) = m_init(13)/1000;
	m_init(14) = m_init(14)/1000;
	//ROS_INFO_STREAM("m_init:" << m_init);
	m = m_init;

	//ROS_INFO_STREAM("m" << m);
	pcl::PointCloud<PointA>::Ptr object_transformed (new pcl::PointCloud<PointA>);
	pcl::PointCloud<PointA>::Ptr conveyour (new pcl::PointCloud<PointA>);
	conveyour->height = 1;
	conveyour->width = 8;
	conveyour->points.resize(conveyour->height * conveyour->width);

	(*conveyour)[0] = (*object)[5892]; (*conveyour)[1] = (*object)[1848];
	(*conveyour)[2] = (*object)[35255]; (*conveyour)[3] = (*object)[41087];
	(*conveyour)[4] = (*object)[45012]; (*conveyour)[5] = (*object)[45013];
	(*conveyour)[6] = (*object)[45014]; (*conveyour)[7] = (*object)[45015];

	pcl::transformPointCloud(*conveyour, *object_transformed, m);
	//----------------
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*object_transformed, centroid);
	Eigen::Matrix3f covariance;
	computeCovarianceMatrixNormalized(*object_transformed, centroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
	eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));

	// move the points to the that reference frame
	Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
	p2w.block<3,3>(0,0) = eigDx.transpose();
	p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
	pcl::PointCloud<PointA> cPoints;
	pcl::transformPointCloud(*object_transformed, cPoints, p2w);
	PointA min_pt, max_pt;
	pcl::getMinMax3D(cPoints, min_pt, max_pt);
	const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());
	// final transform

	const Eigen::Quaternionf qfinal(eigDx);
	const Eigen::Vector3f tfinal = eigDx*mean_diag + centroid.head<3>();
	pcl::PointXYZRGB minp, maxp;
	Eigen::Matrix4f _tr = Eigen::Matrix4f::Identity();
	_tr.topLeftCorner<3,3>() = qfinal.toRotationMatrix();
	_tr.block<3,1>(0,3) = tfinal;

	float _x = (max_pt.x-min_pt.x)* 0.8;// * 0.4;// * 3;
	float _y = (max_pt.y-min_pt.y) * 0.8;// * 0.6;// * 3;//* 0.3;// * 0.3;
	float _z = (max_pt.z-min_pt.z) * 0.5;// * 0.5;// * 0.6;

	//****
	_tr = _tr.inverse().eval();

	pcl::PointCloud<PointT>::Ptr cloud_bm (new pcl::PointCloud<PointT>);
	*cloud_bm = scene;

	pcl::PointIndices::Ptr object_indices (new pcl::PointIndices);
	for (size_t i = 0; i < cloud_bm->size(); i++){
		PointT p = (*cloud_bm)[i];

		p.getVector4fMap() = _tr * p.getVector4fMap();
		if(fabsf(p.x) <= _x && fabsf(p.y) <= _y && fabsf(p.z) <= _z ) {
			object_indices->indices.push_back(i);
		}
	}
	pcl::PointCloud<PointT> small_cube;

	small_cube.height = 1;
	small_cube.width = object_indices->indices.size();
	small_cube.points.resize(small_cube.height * small_cube.width);

	pcl::copyPointCloud (*cloud_bm , object_indices->indices, small_cube );
        pcl::io::savePCDFile("small_cube01.pcd", small_cube);
	return small_cube;
}
//--------------------------------------------------------------------------------------------
void storeResults(PoseEstimation::Response &resp, MsgT data, sensor_msgs::PointCloud2 scenei, pcl::PointCloud<PointT> object){
	// Store results
    resp.labels_int.clear();
    resp.poses.clear();
    resp.pose_value.clear();

    resp.labels_int.resize(data.response.poses.size());
	resp.poses.reserve(data.response.poses.size());
	resp.pose_value.resize(data.response.poses.size());

	sensor_msgs::PointCloud2 objecti;
	pcl::toROSMsg(object, objecti);

	resp.scene = scenei;
	resp.object = objecti;

	resp.poses = data.response.poses;
	resp.pose_value = data.response.pose_value;
	resp.labels_int = data.response.labels_int;

	//publish here
	publish_for_vizualizer.publish(resp);
}
//-------------------------------------------------------------------------------------------------
void detectRotorcaps(pcl::PointCloud<PointT> cutScene, PoseEstimation::Response &resp){

    ros::NodeHandle nh("~");
    std::string rotorcapPCD;
    nh.getParam("rotorcapPCD", rotorcapPCD);

    // get properties on gripper name and on grasp database directory

	pcl::console::print_value(" Starting rotorcap detection! \n");

//	std::string rotorcap = "/home/acat2/LH4_workspace/MobileManipulator/Testers/sdu_sim/pose_estimation/data/rotorcap_middle.pcd";

	sensor_msgs::PointCloud2 scenei;
	pcl::toROSMsg(cutScene, scenei);

	MsgT msgrotorcaps;
	msgrotorcaps.request.visualize = false;
	msgrotorcaps.request.rotorcap = true;
	msgrotorcaps.request.table = false;
        msgrotorcaps.request.threshold = 5;
	msgrotorcaps.request.cothres = 1;
        msgglobal.request.objects.clear();
	msgrotorcaps.request.objects.push_back(rotorcapPCD);

	pcl::PointCloud<PointT> object;
	pcl::io::loadPCDFile(rotorcapPCD, object);

	msgrotorcaps.request.cloud = scenei;
	if(getPose.call(msgrotorcaps)) {
		storeResults(resp, msgrotorcaps, scenei, object);
	}

}
//--------------------------------------------------------------------------------------------
void detectConveyourBeltAndRotorcaps(PoseEstimation::Response &resp){
	//---POSE ESTIMATION BLOCK
	ROS_INFO("Subscribing to /service/getPose ...");
	ros::service::waitForService("/service/getPose");

	sensor_msgs::PointCloud2 scenei;
	pcl::toROSMsg(carmine_pointCloud, scenei);

	msgglobal.request.visualize = false;
	msgglobal.request.table = false;
    	msgglobal.request.threshold = 10;
	msgglobal.request.cothres = 1;

	msgglobal.request.objects.clear();
	msgglobal.request.objects.push_back(object_path);

	msgglobal.request.cloud = scenei;
	if(getPose.call(msgglobal)) {

		pcl::PointCloud<PointA>::Ptr object(new pcl::PointCloud<PointA>());
		pcl::io::loadPCDFile(object_path, *object);

		ROS_INFO_STREAM("number of ints:" << msgglobal.response.labels_int.size() );
		ROS_INFO_STREAM( "number of pose_value:" << msgglobal.response.pose_value.size() );
		if (stereo_pointCloud.size() > 1){
			pcl::console::print_warn("Using carmine and stereo!\n");
			pcl::PointCloud<pcl::PointXYZRGBA> outSmall = cutConveyourBelt(object, stereo_pointCloud, msgglobal);
			detectRotorcaps(outSmall, resp);

		} else {
			pcl::console::print_error("Stereo is empty, using only carmine!\n");
			pcl::PointCloud<pcl::PointXYZRGBA> outSmall = cutConveyourBelt(object, carmine_pointCloud, msgglobal);
			detectRotorcaps(outSmall, resp);
		}
		keep_latest_best_pose = msgglobal;
	} else {
		ROS_ERROR("Something went wrong when calling /object_detection/global");
	}
}

//------------------------------------------------------------------------------------------------------------------------
bool pose_estimation_service(PoseEstimation::Request &req, PoseEstimation::Response &resp){
	ROS_INFO("Starting pose estimation service\n!");
	std::string scenario = req.scenario;

	if(scenario == "save_point_clouds"){
		ros::NodeHandle nh("~");
		std::string pcddir;
    		nh.getParam("pcddir", pcddir);
		
		if ((stereo_pointCloud.size() < 0 && carmine_pointCloud.size() < 0)){
			pcl::console::print_warn("Waiting for point clouds!\n");
		}
		else {
			if (carmine_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"carmine_PC.pcd", carmine_pointCloud);
			if (stereo_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"stereo_PC.pcd", stereo_pointCloud);
			pcl::console::print_value("Saving stereo and carmine point clouds\n");
		}
	}
	else if (scenario == "detect_rotorcaps_on_coveyour_belt"){
		ros::NodeHandle nh("~");
        std::string pcddir;
    		nh.getParam("conveyor_belt_2PCD", object_path);
            nh.getParam("pcddir", pcddir);

		pcl::console::print_value("detecting conveyor, then rotorcaps\n");
        if (carmine_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"carmine_PC.pcd", carmine_pointCloud);
        if (stereo_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"stereo_PC.pcd", stereo_pointCloud);

        if (carmine_pointCloud.size()> 0) detectConveyourBeltAndRotorcaps(resp);
                else pcl::console::print_error("Cannot grasp frame from carmine! Are you sure it is running?!");
          
                
	}
	else if (scenario == "detect_only_rotorcaps"){
		if (keep_latest_best_pose.response.poses.size() > 0) {
		ros::NodeHandle nh("~");
        std::string pcddir;
  		nh.getParam("conveyor_belt_2PCD", object_path);
        nh.getParam("pcddir", pcddir);
        if (carmine_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"carmine_PC.pcd", carmine_pointCloud);
        if (stereo_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"stereo_PC.pcd", stereo_pointCloud);

			pcl::PointCloud<PointA>::Ptr object(new pcl::PointCloud<PointA>());
			pcl::io::loadPCDFile(object_path, *object);

			if (stereo_pointCloud.size() > 1){
				pcl::console::print_warn("Using carmine and stereo!\n");
				pcl::PointCloud<pcl::PointXYZRGBA> outSmall = cutConveyourBelt(object, stereo_pointCloud, keep_latest_best_pose);
				detectRotorcaps(outSmall, resp);

			} else {
				pcl::console::print_error("Stereo is empty, using only carmine!\n");
				pcl::PointCloud<pcl::PointXYZRGBA> outSmall = cutConveyourBelt(object, carmine_pointCloud, keep_latest_best_pose);
				detectRotorcaps(outSmall, resp);
			}
		} else {
			pcl::console::print_error("There is no conveyour pose! Get conveyour pose first!\n");
		}

	}


	return true;
}
//------------------------------------------------------------------------------------------------------------------------
/*
 * Main entry point
 */
int main(int argc, char **argv) {

	// setup node
	// Initialize node
	const std::string name = "inSceneDetector";
	ros::init(argc, argv, name);
	ros::NodeHandle nh("~");
	ROS_INFO("waiting for service/getPose");
	getPose = nh.serviceClient<MsgT>("/service/getPose");
	stereoPointCloudSaver(nh, "/pikeBack");
	kinectPointCloudSaver(nh, "/carmine1");
	ROS_INFO("Starting services!");
	// Start
	ros::ServiceServer servglobal = nh.advertiseService<PoseEstimation::Request, PoseEstimation::Response>("detect", pose_estimation_service);

	publish_for_vizualizer = nh.advertise<PoseEstimation::Response>("vizualize", 1000);
	ros::spin();

	return 0;
}
