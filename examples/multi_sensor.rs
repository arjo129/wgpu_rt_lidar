use std::{borrow::Cow, iter, time::Instant};

use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3, Vec4};
use wgpu::util::DeviceExt;
use wgpu_rt_lidar::{
    depth_camera::DepthCamera,
    lidar::Lidar,
    utils::{create_cube, get_raytracing_gpu},
    vertex, AssetMesh, Instance, RayTraceScene, Vertex,
};

fn get_vlp16_spinning_beam_directions(azimuth_resolution_deg: f32) -> Vec<Vec3> {
    // Fixed vertical angles for the 16 Velodyne VLP-16 lasers
    let vertical_angles_deg: [f32; 16] = [
        -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0,
    ];

    let mut beam_directions: Vec<Vec3> = Vec::new();

    // Determine the number of horizontal steps to cover 360 degrees
    // We use `ceil` to ensure we cover the full 360 degrees even if it means
    // slightly exceeding it on the last step, or adjusting the last step.
    let num_azimuth_steps = (360.0 / azimuth_resolution_deg).ceil() as usize;

    // Iterate through each horizontal (azimuth) step
    for i in 0..num_azimuth_steps {
        // Ensure the angle doesn't exceed 360 degrees (or wraps around)
        let azimuth_angle_deg = (i as f32 * azimuth_resolution_deg) % 360.0;
        let azimuth_angle_rad = azimuth_angle_deg.to_radians();

        // Create a rotation quaternion for the current azimuth angle around the Z-axis (upwards)
        // Assuming Z is the axis of rotation for the LiDAR.
        let rotation_quat = Quat::from_rotation_y(azimuth_angle_rad);

        // For each azimuth step, all 16 lasers fire simultaneously
        for &vertical_angle_deg in vertical_angles_deg.iter() {
            let vertical_angle_rad = vertical_angle_deg.to_radians();

            // Define the initial direction of the beam in the sensor's local frame
            // Assuming X-axis is forward, Y-axis is vertical (up/down), Z-axis is sideways.
            let initial_beam_direction = Vec3::new(
                vertical_angle_rad.cos(), // Component along the forward (X) axis
                vertical_angle_rad.sin(), // Component along the vertical (Y) axis
                0.0,
            );

            // Rotate the initial beam direction by the current azimuth angle
            let rotated_beam_direction = rotation_quat * initial_beam_direction;

            beam_directions.push(rotated_beam_direction);
        }
    }

    beam_directions
}


#[tokio::main]
async fn main() {
    // Set up a wgpu instance and device
    let instance = wgpu::Instance::default();
    let (_, device, queue) = get_raytracing_gpu(&instance).await;

    let rec = rerun::RecordingStreamBuilder::new("depth_camera_vis")
        .spawn()
        .unwrap();

    // Lets add a cube as an asset
    let cube = create_cube(1.0);

    // Build Scene. Spawn 16 cubes.
    let side_count = 8;
    let mut instances = vec![];
    for x in 0..side_count {
        for y in 0..side_count {
            instances.push(Instance {
                asset_mesh_index: 0,
                transform: Affine3A::from_rotation_translation(
                    Quat::from_rotation_y(45.9_f32.to_radians()),
                    Vec3 {
                        x: x as f32 * 3.0,
                        y: y as f32 * 3.0,
                        z: -30.0,
                    },
                ),
            });
        }
    }

    let mut scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;

    /// Set the camera frame size
    let mut depth_camera = DepthCamera::new(&device, 1024, 1024, 59.0, 50.0).await;

    /// Set the lidar beams
    let lidar_beams =  get_vlp16_spinning_beam_directions(0.5);
    
    /*let lidar_beams = (0..2040)
        .map(|f| {
            let angle = 3.14 * f as f32 / 2040.0;
            Vec3::new(0.0, angle.sin(), angle.cos())
        })
        .collect::<Vec<_>>();*/
    
    let mut lidar = Lidar::new(&device, lidar_beams).await;

    scene.visualize(&rec);

    // Move the camera back, the cubes are at -30
    for i in 0..3 {
        let start_time = Instant::now();
        let res = depth_camera
            .render_depth_camera(
                &scene,
                &device,
                &queue,
                Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5 + i as f32), Vec3::ZERO, Vec3::Y),
            )
            .await;
        //println!("{:?}", res);
        println!("Took {:?} to render a depth frame", start_time.elapsed());
        use ndarray::ShapeBuilder;

        let mut image = ndarray::Array::<u16, _>::from_elem(
            (
                depth_camera.width() as usize,
                depth_camera.height() as usize,
            )
                .f(),
            65535,
        );
        for (i, x) in res.iter().enumerate() {
            let x = (x * 1000.0) as u16;
            image[(
                i / depth_camera.width() as usize,
                i % depth_camera.width() as usize,
            )] = x;
        }
        let depth_image = rerun::DepthImage::try_from(image)
            .unwrap()
            .with_meter(1000.0)
            .with_colormap(rerun::components::Colormap::Viridis);

        rec.log("depth_cloud", &depth_image);

        println!("Rendering lidar beams");
        let start_time = Instant::now();
        let lidar_pose = Affine3A::from_translation(Vec3::new(2.0, 0.0, i as f32));
        let res = lidar
            .render_lidar_beams(&scene, &device, &queue, &lidar_pose)
            .await;
        println!("Took {:?} to render a lidar frame", start_time.elapsed());
        //println!("{:?}", res);
    }

    let mut updated_instances = vec![];
    // Move instances forward
    for i in 0..instances.len() {
        instances[i].transform.translation.z += -5.0;
        updated_instances.push(i);
    }

    scene
        .set_transform(&device, &queue, &instances, &updated_instances)
        .await
        .unwrap();

    println!(
        "Rendering after moving cubes {:?}",
        Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.5), Vec3::ZERO, Vec3::Y)
    );

    let res = depth_camera
        .render_depth_camera(
            &scene,
            &device,
            &queue,
            Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.5), Vec3::ZERO, Vec3::Y),
        )
        .await;
    println!("{:?}", res.iter().fold(0.0, |acc, x| x.max(acc)));

    println!("Rendering pointcloud");
    let start_time = Instant::now();
    let lidar_pose = Affine3A::from_translation(Vec3::new(2.0, 0.0, 3.0));
    let res = lidar
        .render_lidar_pointcloud(&scene, &device, &queue, &lidar_pose)
        .await;
    println!(
        "Took {:?} to render a lidar pointcloud",
        start_time.elapsed()
    );
    let p = res
        .chunks(4)
        .filter(|p| p[3] < Lidar::no_hit_const())
        .map(|p| {
            lidar_pose
                .transform_point3(Vec3::new(p[0], p[1], p[2]))
                .to_array()
        });
    lidar.visualize_rays(&rec, &lidar_pose, "lidar_beams");
    rec.log("points", &rerun::Points3D::new(p)).unwrap();
}
