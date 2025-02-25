use glam::{Affine3A, Quat, Vec3};
use wgpu_rt_lidar::{vertex, AssetMesh, Instance, RayTraceScene, Vertex};

/// Lets create a cube with 6 faces
fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-0.5, -0.5, 0.5]),
        vertex([0.5, -0.5, 0.5]),
        vertex([0.5, 0.5, 0.5]),
        vertex([-0.5, 0.5, 0.5]),
        // bottom (0, 0, -1)
        vertex([-0.5, 0.5, -0.5]),
        vertex([0.5, 0.5, -0.5]),
        vertex([0.5, -0.5, -0.5]),
        vertex([-0.5, -0.5, -0.5]),
        // right (1, 0, 0)
        vertex([0.5, -0.5, -0.5]),
        vertex([0.5, 0.5, -0.5]),
        vertex([0.5, 0.5, 0.5]),
        vertex([0.5, -0.5, 0.5]),
        // left (-1, 0, 0)
        vertex([-0.5, -0.5, 0.5]),
        vertex([-0.5, 0.5, 0.5]),
        vertex([-0.5, 0.5, -0.5]),
        vertex([-0.5, -0.5, -0.5]),
        // front (0, 1, 0)
        vertex([0.5, 0.5, -0.5]),
        vertex([-0.5, 0.5, -0.5]),
        vertex([-0.5, 0.5, 0.5]),
        vertex([0.5, 0.5, 0.5]),
        // back (0, -1, 0)
        vertex([0.5, -0.5, 0.5]),
        vertex([-0.5, -0.5, 0.5]),
        vertex([-0.5, -0.5, -0.5]),
        vertex([0.5, -0.5, -0.5]),
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
pub async fn main() {
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

    let (vert_buf, indices) = create_vertices();
    let cube = AssetMesh {
        vertex_buf: vert_buf,
        index_buf: indices,
    };
    let side_count = 4;
    let mut instances = vec![];
    for x in 0..side_count {
        for y in 0..side_count {
            instances.push(Instance {
                asset_mesh_index: 0,
                transform: Affine3A::from_rotation_translation(
                    Quat::from_rotation_y(45.9_f32.to_radians()),
                    Vec3 {
                        x: x as f32,
                        y: y as f32,
                        z: 0.0,
                    },
                ),
            });
        }
    }

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
    let mut scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;
}
