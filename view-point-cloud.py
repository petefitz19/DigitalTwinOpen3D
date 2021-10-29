import open3d as o3d
import glob  # Perform file glob matching
import argparse  # For handling CLI args
import multiprocessing as mp

arg_parser = argparse.ArgumentParser(description='View point clouds.')
arg_parser.add_argument('pcd_directory', type=str,
                        default='./.',
                        help='''Directory to read point clouds from.
Defaults to the directory from which this python script is called from''')
arg_parser.add_argument('pcds_to_view', metavar='pcds_to_view', type=int,
                        default=-1, nargs='*',
                        help='''Which point cloud to view.
Defaults to -1 to view all point clouds in directory specified.
Provide point cloud number from 1->N''')

arg_parser.add_argument('voxel_downsample_factor', type=float,
                        default=0.3, nargs='?',
                        help="Size of voxel cube to merge all points into")


def read_all_point_cloud_files(dir_name: str) -> [()]:
    """
    Reads all point clouds from the specified directory
    :param dir_name: Directory name to read from, relative to this python file
    :returns: List of tuples containing (point cloud file name, point cloud data)
    """
    ply_files = glob.glob(dir_name + "*.ply")
    point_clouds = [(ply_file, o3d.io.read_point_cloud(ply_file))
                    for ply_file in ply_files]
    # Return the point clouds in order by FILENAME!
    return sorted(point_clouds)


def voxel_downsample_point_clouds(point_clouds, downsample_factor) -> [()]:
    """
    Downsamples a list of point clouds by a downsample factor that determines how bit the voxel cube is.
    0.1 corresponds to a 0.1cm X 0.1cm X 0.1cm voxel.

    :param point_clouds: List of point clouds to downsample.
    Note that this list is NOT modified!
    :param downsample_factor: Percent to downsample by.
    Note that the closer to 1 the factor is, the MORE downsampling that occurs.
    :return: List of (original point cloud file name, PointCloud object) pairs.
    """
    return [(file_name, pcd.voxel_down_sample(voxel_size=downsample_factor))
            for (file_name, pcd) in point_clouds]


def main():
    args = arg_parser.parse_args()
    og_point_clouds = read_all_point_cloud_files(args.pcd_directory)
    downsampled_point_clouds = voxel_downsample_point_clouds(og_point_clouds, args.voxel_downsample_factor)

    visualization_processes = []
    # Which point clouds to view. If -1, then ALL of them.
    pcds_to_view = -1
    if args.pcds_to_view == -1:
       pcds_to_view = [i for i in range(0, len(downsampled_point_clouds), 1)]
    else:
        # Actually given a list. Need to assign and subtract 1 for indexing
        pcds_to_view = [i-1 for i in args.pcds_to_view]

    # Fork a new child process to visualize the point clouds.
    # This makes EACH of the viewers independent of one another, and independent
    # of the Python process that started the viewers.
    for pcd_to_view in pcds_to_view:
        showing_pcd = [downsampled_point_clouds[pcd_to_view][1]]
        cpid = mp.Process(target=o3d.visualization.draw_geometries, args=(showing_pcd,))
        visualization_processes.append(cpid)
        cpid.start()
        # o3d.visualization.draw_geometries(showing_pcd)

    # The Python interpreter that started the viewers MUST wait for ALL viewers
    # to be closed before exiting itself.
    for child in visualization_processes:
        child.join()

    print("All point cloud viewers closed! Exiting!")


if __name__ == "__main__":
    main()
