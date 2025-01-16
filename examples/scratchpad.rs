use std::{borrow::Cow, iter, time::Instant};

use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3, Vec4};
use wgpu::util::DeviceExt;

#[inline]
fn affine_to_rows(mat: &Affine3A) -> [f32; 12] {
    let row_0 = mat.matrix3.row(0);
    let row_1 = mat.matrix3.row(1);
    let row_2 = mat.matrix3.row(2);
    let translation = mat.translation;
    [
        row_0.x,
        row_0.y,
        row_0.z,
        translation.x,
        row_1.x,
        row_1.y,
        row_1.z,
        translation.y,
        row_2.x,
        row_2.y,
        row_2.z,
        translation.z,
    ]
}
#[inline]
fn affine_to_4x4rows(mat: &Affine3A) -> [f32; 16] {
    let row_0 = mat.matrix3.row(0);
    let row_1 = mat.matrix3.row(1);
    let row_2 = mat.matrix3.row(2);
    let translation = mat.translation;
    [
        row_0.x,
        row_0.y,
        row_0.z,
        translation.x,
        row_1.x,
        row_1.y,
        row_1.z,
        translation.y,
        row_2.x,
        row_2.y,
        row_2.z,
        translation.z,
        0.0,
        0.0,
        0.0,
        0.1,
    ]
}

// from cube
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthCameraUniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    width: u32,
    height: u32,
    padding: [f32; 2],
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

struct Lidar {
    pipeline: wgpu::ComputePipeline,
    ray_directions: Vec<Vec4>,
    ray_direction_gpu_buf: wgpu::Buffer,
    shader: wgpu::ShaderModule,
}

impl Lidar {
async fn new(device: &wgpu::Device, ray_directions: Vec<Vec3>) -> Self {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let ray_directions: Vec<_>= ray_directions.iter().map(|v| Vec4::new(v.x, v.y, v.z, 0.0)).collect();
    let ray_direction_gpu_buf = device.create_buffer_init( &wgpu::util::BufferInitDescriptor {
        label: Some("Lidar Buffer"),
        contents: bytemuck::cast_slice(&ray_directions),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    println!("Lidar buffer size: {:?}", ray_directions.len());
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lidar_computer"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("lidar_shader.wgsl"))),
    });
    Self {
        ray_directions,
        ray_direction_gpu_buf,
        pipeline: {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lidar"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        },
        shader
    }
}

    async fn render_lidar_beams(
        &mut self,
        scene: &RayTraceScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pose: &Affine3A,
    ) -> Vec<f32> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let compute_bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let mut lidar_positions = affine_to_4x4rows(pose);

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&lidar_positions),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.ray_directions.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let ray_direction_gpu_buf = device.create_buffer_init( &wgpu::util::BufferInitDescriptor {
            label: Some("Lidar Buffer"),
            contents: bytemuck::cast_slice(&self.ray_directions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST| wgpu::BufferUsages::COPY_SRC,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::AccelerationStructure(
                        &scene.tlas_package.tlas(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ray_direction_gpu_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: raw_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(iter::empty(), iter::once(&scene.tlas_package));

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, Some(&compute_bind_group), &[]);
        cpass.dispatch_workgroups(self.ray_directions.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&raw_buf, 0, &staging_buffer, 0, staging_buffer.size());

    queue.submit(Some(encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    receiver.recv().unwrap().unwrap();

    {
        let view = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();

        drop(view);
        staging_buffer.unmap();
        return result;
    }
    }
}


struct DepthCamera {
    pipeline: wgpu::ComputePipeline,
    uniforms: DepthCameraUniforms,
    width: u32,
    height: u32,
}

impl DepthCamera {
    async fn new(device: &wgpu::Device, width: u32, height: u32, fov_y: f32) -> Self {
        let uniforms = {
            let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(
                fov_y.to_radians(),
                width as f32 / height as f32,
                0.001,
                1000.0,
            );

            DepthCameraUniforms {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
                width: width,
                height: height,
                padding: [0.0; 2],
            }
        };

        let camera_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_computer"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rt"),
                layout: None,
                module: &camera_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            uniforms,
            width,
            height,
        }
    }

    async fn render_depth_camera(
        &mut self,
        scene: &RayTraceScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_matrix: Mat4,
    ) -> Vec<f32> {
        self.uniforms.view_inverse = view_matrix.inverse();


        let compute_bind_group_layout = self.pipeline.get_bind_group_layout(0);

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[self.uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.width * self.height * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::AccelerationStructure(
                        &scene.tlas_package.tlas(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: raw_buf.as_entire_binding(),
                },
            ],
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: raw_buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(iter::empty(), iter::once(&scene.tlas_package));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, Some(&compute_bind_group), &[]);
            cpass.dispatch_workgroups(self.width / 8, self.height / 8, 1);
        }
        encoder.copy_buffer_to_buffer(&raw_buf, 0, &staging_buffer, 0, staging_buffer.size());

        queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        receiver.recv().unwrap().unwrap();

        {
            let view = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();

            drop(view);
            staging_buffer.unmap();
            return result;
        }
    }
}

struct AssetMesh {
    vertex_buf: Vec<Vertex>,
    index_buf: Vec<u16>,
}

struct Instance {
    asset_mesh_index: usize,
    transform: Affine3A,
}

struct RayTraceScene {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    blas: Vec<wgpu::Blas>,
    tlas_package: wgpu::TlasPackage,
}

impl RayTraceScene {
    async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        assets: &Vec<AssetMesh>,
        instances: &Vec<Instance>,
    ) -> Self {
        let (vertex_data, index_data, start_vertex_address, start_indices_address): (
            Vec<_>,
            Vec<u16>,
            Vec<usize>,
            Vec<usize>,
        ) = assets.iter().fold(
            (vec![], vec![], vec![0], vec![0]),
            |(vertex_buf, index_buf, start_buf, indices_buf), asset| {
                // TODO
                let mut start_vertex_buf = start_buf.clone();
                if let Some(last) = start_vertex_buf.last() {
                    start_vertex_buf.push(*last + vertex_buf.len());
                }

                let mut start_indices_buf = indices_buf.clone();
                if let Some(last) = start_indices_buf.last() {
                    start_indices_buf.push(*last + indices_buf.len());
                }

                (
                    vertex_buf
                        .iter()
                        .chain(asset.vertex_buf.iter())
                        .cloned()
                        .collect::<Vec<Vertex>>(),
                    index_buf
                        .iter()
                        .chain(asset.index_buf.iter())
                        .cloned()
                        .collect::<Vec<u16>>(),
                    start_vertex_buf,
                    start_indices_buf,
                )
            },
        ); //create_vertices();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let geometry_desc_sizes = assets
            .iter()
            .map(|asset| wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_count: asset.vertex_buf.len() as u32,
                vertex_format: wgpu::VertexFormat::Float32x3,
                index_count: Some(asset.index_buf.len() as u32),
                index_format: Some(wgpu::IndexFormat::Uint16),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            })
            .collect::<Vec<_>>();

        let blas = vec![device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: geometry_desc_sizes.clone(),
            },
        )];

        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: instances.len() as u32,
        });

        let mut tlas_package = wgpu::TlasPackage::new(tlas);

        for (idx, instance) in instances.iter().enumerate() {
            tlas_package[idx] = Some(wgpu::TlasInstance::new(
                &blas[instance.asset_mesh_index],
                affine_to_rows(&instance.transform),
                0,
                0xff,
            ));
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let blas_iter: Vec<_> = blas
            .iter()
            .enumerate()
            .map(|(index, blas)| wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &geometry_desc_sizes[index],
                        vertex_buffer: &vertex_buf,
                        first_vertex: start_vertex_address[index] as u32,
                        vertex_stride: std::mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(&index_buf),
                        index_buffer_offset: Some(start_indices_address[index] as u64),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            })
            .collect();
        encoder.build_acceleration_structures(blas_iter.iter(), iter::once(&tlas_package));

        queue.submit(Some(encoder.finish()));
        device.push_error_scope(wgpu::ErrorFilter::Validation);

        Self {
            vertex_buf,
            index_buf,
            blas,
            tlas_package,
        }
    }

    async fn set_transform(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        update_instance: &Vec<Instance>,
        idx: &Vec<usize>,
    ) -> Result<(), String> {
        if update_instance.len() != idx.len() {
            return Err("Instance and index length mismatch".to_string());
        }

        for (i, instance) in update_instance.iter().enumerate() {
            self.tlas_package[idx[i]] = Some(wgpu::TlasInstance::new(
                &self.blas[instance.asset_mesh_index],
                affine_to_rows(&instance.transform),
                0,
                0xff,
            ));
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.build_acceleration_structures(iter::empty(), iter::once(&self.tlas_package));

        Ok(())
    }
}

#[tokio::main]
async fn main() {
    let side_count = 8;
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
    device.on_uncaptured_error(Box::new(move |err| {
        panic!("{:?}",err);
    }));
    let (vert_buf, indices) = create_vertices();
    let cube = AssetMesh {
        vertex_buf: vert_buf,
        index_buf: indices,
    };

    let mut instances = vec![];
    // Build Scene
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
        let lidar_pose= Affine3A::from_translation(Vec3::new(2.0, 0.0,i  as f32));
        let res = lidar.render_lidar_beams(&scene, &device, &queue, &lidar_pose).await;
        println!("Took {:?} to render a lidar frame", start_time.elapsed());
        println!("{:?}", res);//res.iter().fold(0.0, |acc, x| x.max(acc)));
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
