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
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
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
Eigen::Matrix4f keep_latest_best_pose;
//-------------------------------
Eigen::Matrix4f local_screen_shot_pose;
bool local_screen_shot_pose_detected;
pcl::PointCloud<PointA> screen_shot_object; 
//----------------------------
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

//for constrains
std::vector<double> constr_conveyor, constr_table;


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


////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<PointT> getCutRegion(pcl::PointCloud<PointA>::Ptr object_transformed, float _cut_x, float _cut_y, float _cut_z, pcl::PointCloud<PointT> scene){
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

    float _x = (max_pt.x-min_pt.x)* _cut_x;
    float _y = (max_pt.y-min_pt.y) * _cut_y;
    float _z = (max_pt.z-min_pt.z) * _cut_z;

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
    if (small_cube.size() > 0) pcl::io::savePCDFile("small_cube01.pcd", small_cube);


    return small_cube;
}
//----------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<PointT> getCutRegionForTable(pcl::PointCloud<PointA>::Ptr object_transformed, float _cut_x, float _cut_y, float _cut_z, float _cut_x1, float _cut_y1, float _cut_z1, pcl::PointCloud<PointT> scene){
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

    float _x = (max_pt.x-min_pt.x)* _cut_x;
    float _y = (max_pt.y-min_pt.y) * _cut_y;
    float _z = (max_pt.z-min_pt.z) * _cut_z;

    //****
    _tr = _tr.inverse().eval();

    pcl::PointCloud<PointT>::Ptr cloud_bm (new pcl::PointCloud<PointT>);
    *cloud_bm = scene;

    pcl::PointIndices::Ptr object_indices (new pcl::PointIndices);
    for (size_t i = 0; i < cloud_bm->size(); i++){
        PointT p = (*cloud_bm)[i];

        p.getVector4fMap() = _tr * p.getVector4fMap();
        if (p.x < (min_pt.x * _cut_x) || p.y < (min_pt.y * _cut_y) || p.z < (min_pt.z * _cut_z)
                || p.x > max_pt.x * _cut_x1|| p.y > max_pt.y* _cut_y1 || p.z > max_pt.z * _cut_z1 )
        {

        }else object_indices->indices.push_back(i);

    }
    pcl::PointCloud<PointT> small_cube;

    small_cube.height = 1;
    small_cube.width = object_indices->indices.size();
    small_cube.points.resize(small_cube.height * small_cube.width);

    pcl::copyPointCloud (*cloud_bm , object_indices->indices, small_cube );

    if (small_cube.size() > 0) pcl::io::savePCDFile("small_cube01.pcd", small_cube);
else std::cout << "small cube has 0 points\n";

    return small_cube;
}
//------------------------------------------------------------------------------
pcl::PointCloud<PointT> cutTable(pcl::PointCloud<PointA>::Ptr object, pcl::PointCloud<PointT> scene, MsgT msgglobal1){

    tf::Transform transform;
    tf::transformMsgToTF(msgglobal1.response.poses[0], transform);

    Eigen::Matrix4f m_init, m;
    transformAsMatrix(transform, m_init);
    m_init(12) = m_init(12)/1000;
    m_init(13) = m_init(13)/1000;
    m_init(14) = m_init(14)/1000;
    m = m_init;

    /* m(0) = 0.999995;    m(1) = -0.00224549;    m(2) = 0.00208427;
    m(4) = 0.00224671;    m(5) = 0.999997;    m(6) = -0.000595095;
    m(8) = -0.00208288;    m(9) = 0.000599744;    m(10) = 0.999998;
    m(12) = 7.07721/1000;    m(13) = -0.0677484/1000;    m(14) = -0.741719/1000;
    m(3) = 0;    m(7) = 0;    m(11) = 0;  m(15) = 1; */

    pcl::PointCloud<PointA>::Ptr object_transformed (new pcl::PointCloud<PointA>);
    pcl::PointCloud<PointA>::Ptr table (new pcl::PointCloud<PointA>);
    table->height = 1;
    table->width = 8;
    table->points.resize(table->height * table->width);
    
    (*table)[0] = (*object)[108000]; (*table)[1] = (*object)[108001];
    (*table)[2] = (*object)[108002]; (*table)[3] = (*object)[108003];
    (*table)[4] = (*object)[108004]; (*table)[5] = (*object)[108005];
    (*table)[6] = (*object)[108006]; (*table)[7] = (*object)[108007];
    

    pcl::transformPointCloud(*table, *object_transformed, m);
    pcl::PointCloud<PointT> small_cube = getCutRegionForTable(object_transformed, 1.8, 1.8, 1.8, 3, 1, 1, scene);
    return small_cube;
}
//------------------------------------------------------------------

pcl::PointCloud<PointT> cutConveyourBelt(pcl::PointCloud<PointA>::Ptr object, pcl::PointCloud<PointT> scene, Eigen::Matrix4f m){//MsgT msgglobal1){

    /*tf::Transform transform;
    tf::transformMsgToTF(msgglobal1.response.poses[0], transform);

    Eigen::Matrix4f m_init, m;
    transformAsMatrix(transform, m_init);
    m_init(12) = m_init(12)/1000;
    m_init(13) = m_init(13)/1000;
    m_init(14) = m_init(14)/1000;
    m = m_init; */


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
    pcl::PointCloud<PointT> small_cube = getCutRegion(object_transformed, 0.8, 0.8, 0.5, scene);
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
//??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

void convert(Eigen::Matrix4f &source, geometry_msgs::Transform &dest)
{
    tf::Transform tf_dest;
    Eigen::Matrix3f rot;
    rot << source(0,0), source(0,1), source(0,2),
            source(1,0), source(1,1), source(1,2),
            source(2,0), source(2,1), source(2,2);
    Eigen::Quaternionf quad(rot);
    quad.normalize();
    tf::Quaternion q(quad.x(), quad.y(), quad.z(), quad.w());
    tf_dest.setOrigin(tf::Vector3(source(0,3), source(1,3), source(2,3)));
    tf_dest.setRotation(q);
    dest.rotation.x = quad.x();
    dest.rotation.y = quad.y();
    dest.rotation.z = quad.z();
    dest.rotation.w = quad.w();

    dest.translation.x = source(0,3)*1000;
    dest.translation.y = source(1,3)*1000;
    dest.translation.z = source(2,3)*1000;
}


void icpForRotorcaps(PoseEstimation::Response &resp, MsgT &data, pcl::PointCloud<PointT> scene, pcl::PointCloud<PointT> object){

    for (int i = 0; i < data.response.poses.size(); i++) {

        tf::Transform transform;
        tf::transformMsgToTF(data.response.poses[i], transform);

        Eigen::Matrix4f rotorcap_pose;
        transformAsMatrix(transform, rotorcap_pose);
        rotorcap_pose(12) = rotorcap_pose(12)/1000;
        rotorcap_pose(13) = rotorcap_pose(13)/1000;
        rotorcap_pose(14) = rotorcap_pose(14)/1000;
       
        pcl::PointCloud<PointT>::Ptr scenePtr(new pcl::PointCloud<PointT>);
        scenePtr = scene.makeShared();

        pcl::PointCloud<PointT>::Ptr transformedObjectPtr(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(object, *transformedObjectPtr, rotorcap_pose);


        pcl::VoxelGrid<PointT> sor2;
        sor2.setInputCloud (transformedObjectPtr);
        sor2.setLeafSize (0.001f, 0.001f, 0.001f);
        sor2.filter (*transformedObjectPtr);


        //downsample everything
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(transformedObjectPtr);
        icp.setInputTarget(scenePtr);
        icp.setMaximumIterations(200);

        pcl::PointCloud<PointT> Final;
        icp.align(Final);


        if (icp.hasConverged ())
        {
            Eigen::Matrix4f m = icp.getFinalTransformation() * rotorcap_pose;
            geometry_msgs::Transform dest;
            convert(m, dest);

            //store results
           // data.response.poses[i] = dest;
           data.response.poses.push_back(dest);
        }
    }
}


//-------------------------------------------------------------------------------------------------
bool detectRotorcaps(pcl::PointCloud<PointT> cutScene, PoseEstimation::Response &resp, std::vector<double> constr, bool viz){

    ros::NodeHandle nh("~");
    std::string rotorcapPCD;
    nh.getParam("rotorcapPCD", rotorcapPCD);

    // get properties on gripper name and on grasp database directory

    pcl::console::print_value(" Starting rotorcap detection! \n");

    sensor_msgs::PointCloud2 scenei;
    pcl::toROSMsg(cutScene, scenei);

    MsgT msgrotorcaps;
    msgrotorcaps.request.visualize = viz;
    msgrotorcaps.request.rotorcap = true;
    msgrotorcaps.request.table = false;
    msgrotorcaps.request.threshold = 5;
    msgrotorcaps.request.cothres = 1;
    msgrotorcaps.request.objects.clear();
    msgrotorcaps.request.objects.push_back(rotorcapPCD);

    msgrotorcaps.request.constrains = constr;

    pcl::PointCloud<PointT> object;
    pcl::io::loadPCDFile(rotorcapPCD, object);

    msgrotorcaps.request.cloud = scenei;
    if(getPose.call(msgrotorcaps)) {
       // icpForRotorcaps(resp, msgrotorcaps, cutScene, object);
        storeResults(resp, msgrotorcaps, scenei, object);
        return true;
    } else return false;

}
//--------------------------------------------------------------------------------------------
bool detectConveyourBeltAndRotorcaps(PoseEstimation::Response &resp, bool viz, Eigen::Matrix4f& m){
    //---POSE ESTIMATION BLOCK
    ROS_INFO("Subscribing to /service/getPose ...");
    ros::service::waitForService("/service/getPose");

    sensor_msgs::PointCloud2 scenei;
    pcl::toROSMsg(carmine_pointCloud, scenei);
    MsgT msgconveyor;
    msgconveyor.request.visualize = viz;
    msgconveyor.request.table = false;
    msgconveyor.request.threshold = 10;
    msgconveyor.request.cothres = 1;

    msgconveyor.request.objects.clear();
    msgconveyor.request.objects.push_back(object_path);

    msgconveyor.request.cloud = scenei;
    pcl::PointCloud<pcl::PointXYZRGBA> outSmall;
    
    //pcl::console::print_warn("Trying to detect conveyor!\n");
    if (getPose.call(msgconveyor)){

        tf::Transform transform;
        tf::transformMsgToTF(msgconveyor.response.poses[0], transform);
        Eigen::Matrix4f m_init;
        transformAsMatrix(transform, m_init);
        m_init(12) = m_init(12)/1000;
        m_init(13) = m_init(13)/1000;
        m_init(14) = m_init(14)/1000;
        m = m_init;

        keep_latest_best_pose = m;
        return true;
    } else {
        return false;
        //ROS_ERROR("Something went wrong when calling /object_detection/global");
    }
}
//----------------------------
pcl::PointCloud<pcl::PointXYZRGBA> detectTableAndRotorcaps(PoseEstimation::Response &resp, bool viz){
    //---POSE ESTIMATION BLOCK
    ROS_INFO("Subscribing to /service/getPose ...");
    ros::service::waitForService("/service/getPose");

    sensor_msgs::PointCloud2 scenei;
    pcl::toROSMsg(carmine_pointCloud, scenei);

    msgglobal.request.visualize = viz;
    msgglobal.request.table = false;
    msgglobal.request.threshold = 10; // was 20
    msgglobal.request.cothres = 1;

    msgglobal.request.objects.clear();
    msgglobal.request.objects.push_back(object_path);

    msgglobal.request.cloud = scenei;

    pcl::PointCloud<pcl::PointXYZRGBA> outSmall;

    if(getPose.call(msgglobal)) {
        pcl::PointCloud<PointA>::Ptr object(new pcl::PointCloud<PointA>());
        pcl::io::loadPCDFile(object_path, *object);

        if (stereo_pointCloud.size() > 1){
            outSmall = cutTable(object, stereo_pointCloud, msgglobal);
        }
    } else {
        ROS_ERROR("Something went wrong when calling /object_detection/global");
    }
    return outSmall;
}
//-----------------------------------------------------------------------------
void detectScreenShot(PoseEstimation::Response &resp, bool viz){
    //---POSE ESTIMATION BLOCK
    ROS_INFO("Subscribing to /service/getPose ...");
    ros::service::waitForService("/service/getPose");

    sensor_msgs::PointCloud2 scenei;
    pcl::toROSMsg(carmine_pointCloud, scenei);

    MsgT msgScreenShot;
    msgScreenShot.request.visualize = viz;
    msgScreenShot.request.table = false;
    msgScreenShot.request.threshold = 20;
    msgScreenShot.request.cothres = 1;

    msgScreenShot.request.objects.clear();
    msgScreenShot.request.objects.push_back(object_path);

    msgScreenShot.request.cloud = scenei;
    if(getPose.call(msgScreenShot)) {
        tf::Transform transform;
        tf::transformMsgToTF(msgScreenShot.response.poses[0], transform);

        Eigen::Matrix4f m_init;
        transformAsMatrix(transform, m_init);
        m_init(12) = m_init(12)/1000;
        m_init(13) = m_init(13)/1000;
        m_init(14) = m_init(14)/1000;
        local_screen_shot_pose = m_init;
        local_screen_shot_pose_detected = true;
        pcl::console::print_value("Found screen shot pose!\n");
    } else {
        ROS_ERROR("Something went wrong when calling /object_detection/global");
    }
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//----------------------------------------------------------------------------
//_x=1.8 --_y=1.8 --_z=1.8 --_x2=3 --_y2=1 --_z2=1 conveyor
pcl::PointCloud<PointT> cutSceneScreenShot(string object_name, pcl::PointCloud<PointT> scene) {
    
    pcl::PointCloud<PointA>::Ptr object_transformed (new pcl::PointCloud<PointA>);
    pcl::PointCloud<PointA>::Ptr obj (new pcl::PointCloud<PointA>);
    obj->height = 1;
    obj->width = 8;
    obj->points.resize(obj->height * obj->width);

    pcl::PointCloud<PointT> small_cube;

    if (object_name == "table") {
        (*obj)[0] = (screen_shot_object)[224000]; (*obj)[1] = (screen_shot_object)[224001];
        (*obj)[2] = (screen_shot_object)[224002]; (*obj)[3] = (screen_shot_object)[224003];
        (*obj)[4] = (screen_shot_object)[224004]; (*obj)[5] = (screen_shot_object)[224005];
        (*obj)[6] = (screen_shot_object)[224006]; (*obj)[7] = (screen_shot_object)[224007];
        pcl::transformPointCloud(*obj, *object_transformed, local_screen_shot_pose);
        small_cube = getCutRegionForTable(object_transformed, 1, 1, 1, 3, 1, 1, scene);
    } else if (object_name == "conveyor") {
       /* (*obj)[0] = (screen_shot_object)[224008]; (*obj)[1] = (screen_shot_object)[224009];
        (*obj)[2] = (screen_shot_object)[224010]; (*obj)[3] = (screen_shot_object)[224011];
        (*obj)[4] = (screen_shot_object)[224012]; (*obj)[5] = (screen_shot_object)[224013];
        (*obj)[6] = (screen_shot_object)[224014]; (*obj)[7] = (screen_shot_object)[224015]; */
        /*(*obj)[0] = (screen_shot_object)[307208]; (*obj)[1] = (screen_shot_object)[307209];
        (*obj)[2] = (screen_shot_object)[307210]; (*obj)[3] = (screen_shot_object)[307211];
        (*obj)[4] = (screen_shot_object)[307212]; (*obj)[5] = (screen_shot_object)[307213];
        (*obj)[6] = (screen_shot_object)[307214]; (*obj)[7] = (screen_shot_object)[307215]; */

	(*obj)[0] = (screen_shot_object)[72008]; (*obj)[1] = (screen_shot_object)[72009];
        (*obj)[2] = (screen_shot_object)[72010]; (*obj)[3] = (screen_shot_object)[72011];
        (*obj)[4] = (screen_shot_object)[72012]; (*obj)[5] = (screen_shot_object)[72013];
        (*obj)[6] = (screen_shot_object)[72014]; (*obj)[7] = (screen_shot_object)[72015];

        pcl::transformPointCloud(*obj, *object_transformed, local_screen_shot_pose);
        //small_cube = getCutRegionForTable(object_transformed, 1.8, 1.8, 1.8, 3, 2, 1, scene);
        small_cube = getCutRegionForTable(object_transformed, 1, 1, 1, 1, 1, 0.5, scene);
    }
    return small_cube;
}

//----------------------------------------------------------------------------
void saveLocallyPointClouds(std::string pcddir){
    pcl::io::savePCDFile(pcddir+"carmine_PC.pcd", carmine_pointCloud);
    pcl::io::savePCDFile(pcddir+"carmine_PC.pcd", carmine_pointCloud);
    pcl::io::savePCDFileBinary(pcddir+"stereo_PC_binary.pcd", stereo_pointCloud);
    pcl::io::savePCDFile(pcddir+"stereo_PC.pcd", stereo_pointCloud);
}

//----------------------------------------------------------------------------------------------------------------------
pcl::PointCloud<PointT> processConveyorBelt(PoseEstimation::Response &resp){
    Eigen::Matrix4f m, m_backUp;
    m_backUp(0) =  0.999826;    m_backUp(1) = -0.0161273;    m_backUp(2) = -0.0106435;
    m_backUp(4) =  0.0159015;    m_backUp(5) = 0.999664;    m_backUp(6) = -0.0209254;
    m_backUp(8) = 0.0109774;    m_backUp(9) = 0.0207531;    m_backUp(10) = 0.999735;
    m_backUp(12) = 0.0392795;    m_backUp(13) = -0.0450697;    m_backUp(14) = 0.0324294;
    m_backUp(3) = 0;    m_backUp(7) = 0;    m_backUp(11) = 0;  m_backUp(15) = 1;

    pcl::PointCloud<PointA>::Ptr object(new pcl::PointCloud<PointA>());
    pcl::io::loadPCDFile(object_path, *object);

    bool detected = detectConveyourBeltAndRotorcaps(resp, false, m);

    //check if it is not a wrong conveyor belt
    if (abs(m(12) - m_backUp(12)) > 0.1 || abs(m(13) - m_backUp(13)) > 0.1 && abs(m(14) - m_backUp(14)) > 0.1){ m = m_backUp;
        pcl::console::print_error("USING!");}

    pcl::PointCloud<PointT> outSmall;
    if (detected) outSmall = cutConveyourBelt(object, stereo_pointCloud, m);
    else {
        pcl::console::print_error("Using predefined m, stereo point cloud %d and carmine %d\n", stereo_pointCloud.size(), carmine_pointCloud.size());
        outSmall = cutConveyourBelt(object, stereo_pointCloud, m_backUp);
    }
    return outSmall;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pcl::PointCloud<PointT> extractPlane(pcl::PointCloud<PointT> cur, float thr){
    pcl::PointCloud<PointT> out;
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<PointT> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (thr);


    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud<PointT, PointT>(cur, *cloud);
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    
    // Create the filtering object
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    extract.setNegative (true);
    extract.filter (out);
    return out;
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
            pcl::console::print_value("Waiting for point clouds!\n");
        }
        else {
            if (carmine_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"carmine_PC.pcd", carmine_pointCloud);
            if (stereo_pointCloud.size() > 0) pcl::io::savePCDFile(pcddir+"stereo_PC.pcd", stereo_pointCloud);
            pcl::console::print_value("Saving stereo and carmine point clouds\n");
        }
    }
    //..................................................................
    else if(scenario == "detect_rotorcaps_on_table"){
        ros::NodeHandle nh("~");
        std::string pcddir;
        nh.getParam("table_2PCD", object_path);
        nh.getParam("pcddir", pcddir);

        if (carmine_pointCloud.size()> 0 && stereo_pointCloud.size() > 0) {
            saveLocallyPointClouds(pcddir);
            pcl::console::print_value("Detecting rotorcaps\n");

            pcl::PointCloud<pcl::PointXYZRGBA> small_table;
            //we have the big pose, just cut conveyor and estimate rotorcaps
            if (local_screen_shot_pose_detected) {
                small_table = cutSceneScreenShot("table", stereo_pointCloud);
                small_table = extractPlane(small_table, 0.0055);
            } else {
                pcl::console::print_value("Detecting table, then rotorcaps\n");
                small_table = detectTableAndRotorcaps(resp, false);
            }
            if (small_table.size() > 0) detectRotorcaps(small_table, resp, constr_table, false);

        }
        else pcl::console::print_error("Cannot grasp frame from carmine & stereo! Are you sure they are running?!");
    }
    ///------------------------------------------------------------------------------
    else if (scenario == "detect_rotorcaps_on_coveyour_belt"){
        ros::NodeHandle nh("~");
        std::string pcddir;
        nh.getParam("conveyor_belt_2PCD", object_path);
        nh.getParam("pcddir", pcddir);

        pcl::PointCloud<PointT> outSmall;
        if (stereo_pointCloud.size() > 0 && carmine_pointCloud.size() > 0) {
            saveLocallyPointClouds(pcddir);
            pcl::console::print_value("Detecting rotorcaps\n");
            
            //we have the big pose, just cut conveyor and estimate rotorcaps
            if (local_screen_shot_pose_detected) {
                outSmall = cutSceneScreenShot("conveyor", stereo_pointCloud);
                outSmall = extractPlane(outSmall, 0.007);
            }
            else {
                //detect conveyor
                pcl::console::print_value("Detecting conveyor, then rotorcaps\n");
                processConveyorBelt(resp);
            }
            bool rotorcaps_detected = detectRotorcaps(outSmall, resp, constr_conveyor, false);

        }
        else pcl::console::print_error("Cannot grasp frame from carmine & stereo! Are you sure they are running?! stereo size: %d carmine size: %d\n", stereo_pointCloud.size(), carmine_pointCloud.size());
    }
    //-------------------------------------------------------
    else if (scenario == "detect_only_rotorcaps"){
        ros::NodeHandle nh("~");
        std::string pcddir;
        nh.getParam("conveyor_belt_2PCD", object_path);
        nh.getParam("pcddir", pcddir);

        pcl::PointCloud<PointA>::Ptr object(new pcl::PointCloud<PointA>());
        pcl::io::loadPCDFile(object_path, *object);

        if (stereo_pointCloud.size() > 1 && carmine_pointCloud.size()> 0){
            saveLocallyPointClouds(pcddir);

            pcl::PointCloud<pcl::PointXYZRGBA> outSmall = cutConveyourBelt(object, stereo_pointCloud, keep_latest_best_pose);
            detectRotorcaps(outSmall, resp, constr_conveyor, false);
        }
        else pcl::console::print_error("There is no conveyour pose! Get conveyour pose first!\n");
        
    }
    //------------------------------------------------------------------------------------------------------------
    else if (scenario == "detect_screen_shot"){
        ros::NodeHandle nh("~");
        std::string pcddir;
        nh.getParam("scene_screenshot_PCD", object_path);
 	
        nh.getParam("pcddir", pcddir);

        if (carmine_pointCloud.size() > 0) {
            pcl::console::print_value("Detecting scene screenshot!\n");
            saveLocallyPointClouds(pcddir);
            detectScreenShot(resp, false);
        }
    }
    return true;
}
//------------------------------------------------------------------------------------------------------------------------
void fillConstrains(){
    constr_conveyor.push_back(-0.01395);
    constr_conveyor.push_back(-0.76624);
    constr_conveyor.push_back(-0.64241);

    constr_conveyor.push_back(-0.043472);
    constr_conveyor.push_back(-0.018472);
    constr_conveyor.push_back(0.856371);
    constr_conveyor.push_back(0.05);
    //-----------------------------------
    constr_table.push_back(-0.02571);
    constr_table.push_back(-0.7486);
    constr_table.push_back(-0.6624);

    constr_table.push_back(-0.301625);
    constr_table.push_back(-0.207056);
    constr_table.push_back(1.191860);
    constr_table.push_back(0.15);
    local_screen_shot_pose_detected = false;

}

bool once = false;
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

    pcl::console::setVerbosityLevel(pcl::console::L_WARN);
    
    if (!once) {
        // Start
        fillConstrains();

        //load screen shot
        string screensShotObjectPath;
        nh.getParam("scene_screenshot_PCD",  screensShotObjectPath);
        pcl::io::loadPCDFile(screensShotObjectPath, screen_shot_object);

        //--------------------
        once = true;
    }

    ros::ServiceServer servglobal = nh.advertiseService<PoseEstimation::Request, PoseEstimation::Response>("detect", pose_estimation_service);

    publish_for_vizualizer = nh.advertise<PoseEstimation::Response>("vizualize", 1000);
    ros::spin();

    return 0;
}
