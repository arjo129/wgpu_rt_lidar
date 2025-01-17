use std::{borrow::Cow, iter, time::Instant};

use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3, Vec4};
use wgpu::util::DeviceExt;
use wgpu_rt_lidar::{
    depth_camera::DepthCamera, lidar::Lidar, vertex, AssetMesh, Instance, RayTraceScene, Vertex,
};

/// Lets create a cube with 6 faces
fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1.0, -1.0, 1.0]),
        vertex([1.0, -1.0, 1.0]),
        vertex([1.0, 1.0, 1.0]),
        vertex([-1.0, 1.0, 1.0]),
        // bottom (0, 0, -1)
        vertex([-1.0, 1.0, -1.0]),
        vertex([1.0, 1.0, -1.0]),
        vertex([1.0, -1.0, -1.0]),
        vertex([-1.0, -1.0, -1.0]),
        // right (1, 0, 0)
        vertex([1.0, -1.0, -1.0]),
        vertex([1.0, 1.0, -1.0]),
        vertex([1.0, 1.0, 1.0]),
        vertex([1.0, -1.0, 1.0]),
        // left (-1, 0, 0)
        vertex([-1.0, -1.0, 1.0]),
        vertex([-1.0, 1.0, 1.0]),
        vertex([-1.0, 1.0, -1.0]),
        vertex([-1.0, -1.0, -1.0]),
        // front (0, 1, 0)
        vertex([1.0, 1.0, -1.0]),
        vertex([-1.0, 1.0, -1.0]),
        vertex([-1.0, 1.0, 1.0]),
        vertex([1.0, 1.0, 1.0]),
        // back (0, -1, 0)
        vertex([1.0, -1.0, 1.0]),
        vertex([-1.0, -1.0, 1.0]),
        vertex([-1.0, -1.0, -1.0]),
        vertex([1.0, -1.0, -1.0]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

/// If the environment variable `WGPU_ADAPTER_NAME` is set, this function will attempt to
/// initialize the adapter with that name. If it is not set, it will attempt to initialize
/// the adapter which supports the required features.
async fn get_adapter_with_capabilities_or_from_env(
    instance: &wgpu::Instance,
    required_features: &wgpu::Features,
    required_downlevel_capabilities: &wgpu::DownlevelCapabilities,
) -> wgpu::Adapter {
    use wgpu::Backends;
    if std::env::var("WGPU_ADAPTER_NAME").is_ok() {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, None)
            .await
            .expect("No suitable GPU adapters found on the system!");

        let adapter_info = adapter.get_info();
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(*required_features),
            "Adapter does not support required features for this example: {:?}",
            *required_features - adapter_features
        );

        let downlevel_capabilities = adapter.get_downlevel_capabilities();
        assert!(
            downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
            "Adapter does not support the minimum shader model required to run this example: {:?}",
            required_downlevel_capabilities.shader_model
        );
        assert!(
                downlevel_capabilities
                    .flags
                    .contains(required_downlevel_capabilities.flags),
                "Adapter does not support the downlevel capabilities required to run this example: {:?}",
                required_downlevel_capabilities.flags - downlevel_capabilities.flags
            );
        adapter
    } else {
        let adapters = instance.enumerate_adapters(Backends::all());

        let mut chosen_adapter = None;
        for adapter in adapters {
            let required_features = *required_features;
            let adapter_features = adapter.features();
            if !adapter_features.contains(required_features) {
                continue;
            } else {
                chosen_adapter = Some(adapter);
                break;
            }
        }

        chosen_adapter.expect("No suitable GPU adapters found on the system!")
    }
}

#[tokio::main]
async fn main() {
    // Set up a wgpu instance and device
    let instance = wgpu::Instance::default();
    let required_features = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
        | wgpu::Features::VERTEX_WRITABLE_STORAGE
        | wgpu::Features::EXPERIMENTAL_RAY_QUERY
        | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE;
    let required_downlevel_capabilities = wgpu::DownlevelCapabilities::default();
    let adapter = get_adapter_with_capabilities_or_from_env(
        &instance,
        &required_features,
        &required_downlevel_capabilities,
    )
    .await;

    let Ok((device, queue)) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
    else {
        panic!("Failed to create device");
    };

    // Lets add a cube as an asset
    let (vert_buf, indices) = create_vertices();
    let cube = AssetMesh {
        vertex_buf: vert_buf,
        index_buf: indices,
    };

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

    let mut depth_camera = DepthCamera::new(&device, 256, 256, 59.0).await;

    let lidar_beams = (0..256)
        .map(|f| {
            let angle = 3.14 * f as f32 / 256.0;
            Vec3::new(0.0, angle.sin(), angle.cos())
        })
        .collect::<Vec<_>>();
    let mut lidar = Lidar::new(&device, lidar_beams).await;

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
        println!("{:?}", res.iter().fold(0.0, |acc, x| x.max(acc)));

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

    let res = depth_camera
        .render_depth_camera(
            &scene,
            &device,
            &queue,
            Mat4::look_at_rh(Vec3::new(0.0, 0.0, 4.5), Vec3::ZERO, Vec3::Y),
        )
        .await;
    println!("{:?}", res.iter().fold(0.0, |acc, x| x.max(acc)));
}
