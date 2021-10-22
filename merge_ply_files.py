try:
    import os
    import open3d as o3d
    import numpy as np
    import copy
except ImportError as e:
    print(e)

# TODO: Insert the path to where your point clouds are located here (This syntax works for Mac but may be different for Windows)
path_to_point_clouds = "/Users/pete/Documents/Programming/Python/IIT Digitsl Twin/Point_Clouds/"

def load_point_clouds(voxel_size=0.0):
    pcds = []

    filenames = next(os.walk(path_to_point_clouds), (None, None, []))[2]  # [] if no file
    for file in filenames:
        if file.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(path_to_point_clouds + file)
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
            pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * 2, max_nn=30))
            pcds.append(pcd_down)

    # TODO: Translate or rotate pcd files here if points are registering incorrectly
    pcds[0].translate((0, 10, 0))

    R = pcds[1].get_rotation_matrix_from_xyz((0, 0, np.pi/2))
    pcds[1] = pcds[1].rotate(R, center=(0,0,0))

    R = pcds[2].get_rotation_matrix_from_xyz((0, 0, 3* np.pi/2))
    pcds[2] = pcds[2].rotate(R, center=(0,0,0))

    return pcds

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


# Voxel size can be modified here
voxel_size = 0.02


pcds_down = load_point_clouds(voxel_size)

print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 150
max_correspondence_distance_fine = voxel_size * 15
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

# Loads point clouds given a voxel size
pcds = load_point_clouds(voxel_size)
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds[point_id]
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)

# Writes the merged point clouds
o3d.io.write_point_cloud("multiway_registration.ply", pcd_combined_down)
o3d.visualization.draw_geometries([pcd_combined_down])