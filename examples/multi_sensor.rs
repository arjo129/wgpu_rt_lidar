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

#[tokio::main]
async fn main() {
    // Set up a wgpu instance and device
    let instance = wgpu::Instance::default();
    let (adapter, device, queue) = get_raytracing_gpu(&instance).await;

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

    let mut depth_camera = DepthCamera::new(&device, 1024, 1024, 59.0, 50.0).await;

    let lidar_beams = (0..256)
        .map(|f| {
            let angle = 3.14 * f as f32 / 256.0;
            Vec3::new(0.0, angle.sin(), angle.cos())
        })
        .collect::<Vec<_>>();
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
        //println!("{:?}", res.iter().fold(0.0, |acc, x| x.w.max(acc)));

        //let positions: Vec<_> = res.iter().map(|x| rerun::Position3D::new(x.x, x.y, x.z)).collect();
        rec.log("depth_cloud", &depth_image);
        //rec.log("depth_cloud", &rerun::Points3D::new(positions));

        println!("Rendering lidar beams");
        let start_time = Instant::now();
        let lidar_pose = Affine3A::from_translation(Vec3::new(2.0, 0.0, i as f32));
        let res = lidar
            .render_lidar_beams(&scene, &device, &queue, &lidar_pose)
            .await;
        println!("Took {:?} to render a lidar frame", start_time.elapsed());
        println!("{:?}", res); //res.iter().fold(0.0, |acc, x| x.max(acc)));
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

    let lidar_pose = Affine3A::from_translation(Vec3::new(2.0, 0.0, 3.0));
    let res = lidar
        .render_lidar_pointcloud(&scene, &device, &queue, &lidar_pose)
        .await;
    println!("{:?}", res); //res.iter().fold(0.0, |acc, x| x.max(acc)));
}
