use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Quat, Vec3};
use wgpu::{util::DeviceExt, Blas, Instance};

/// Handle for individual rigid geometries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectHandle(usize);

/// Handle for individual instances of rigid geometries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceHandle(usize);

/// Handle for individual lidar sensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LidarDescriptionHandle(usize);

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

/// A scene that can be rendered with a ray tracer
struct BLASScene {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    blases: Vec<wgpu::Blas>,
    blas_geo_size_descs: Vec<wgpu::BlasTriangleGeometrySizeDescriptor>,
}

struct RaytracePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    tlas_package: wgpu::TlasPackage,
    staging_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    lidar_position_buf: wgpu::Buffer,
    lidar_beam_buf: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BeamDirection {
    pub directions: [f32; 3],
}

impl BeamDirection {
    pub fn new(directions: [f32; 3]) -> Self {
        Self { directions }
    }
}
pub struct LidarDescription {
    pub vectors: Vec<BeamDirection>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct GPULidarBeamId {
    direction: BeamDirection,
    lidar_id: u32,
}

#[derive(Default)]
pub struct LiDARRenderScene {
    objects: Vec<(Vec<Vertex>, Vec<u16>)>,
    need_tlas_rebuild: bool,
    tlas_obj_to_update: HashSet<usize>,
    need_blas_rebuild: Option<BLASScene>,
    raytrace_pipeline: Option<RaytracePipeline>,
    instances: Vec<(ObjectHandle, glam::Affine3A)>,
    lidars: Vec<(LidarDescription, glam::Affine3A)>,
    lidar_pose_buff_needs_update: bool,
}

impl LiDARRenderScene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_object(&mut self, vertices: &Vec<Vertex>, indices: &Vec<u16>) -> ObjectHandle {
        let handle = ObjectHandle(self.objects.len());
        self.objects.push((vertices.clone(), indices.clone()));
        self.need_blas_rebuild = None;
        self.need_tlas_rebuild = true;
        handle
    }

    pub fn add_instance(&mut self, object: ObjectHandle, pose: glam::Affine3A) -> InstanceHandle {
        let inst = InstanceHandle(self.instances.len());
        self.instances.push((object, pose));
        self.need_tlas_rebuild = true;
        inst
    }

    pub fn add_lidar(&mut self, desc: LidarDescription) -> LidarDescriptionHandle {
        let desc_handle = LidarDescriptionHandle(self.lidars.len());
        self.lidars.push((desc, Affine3A::IDENTITY));
        self.lidar_pose_buff_needs_update = true;
        desc_handle
    }

    pub fn set_lidar_pose(&mut self, handle: LidarDescriptionHandle, pose: glam::Affine3A) {
        self.lidars[handle.0].1 = pose;
        self.lidar_pose_buff_needs_update = true;
    }

    pub fn set_instance_pose(&mut self, handle: InstanceHandle, pose: glam::Affine3A) {
        self.instances[handle.0].1 = pose;
        self.tlas_obj_to_update.insert(handle.0);
    }

    pub async fn get_lidar_returns(&mut self, rc: &RenderContext) {
        let numbers = vec![0f32; 256];
        let size = size_of_val(numbers.as_slice()) as wgpu::BufferAddress;
        if self.need_blas_rebuild.is_none() {
            self.need_blas_rebuild = Some(self.build_blas(rc));
        }
        let Some(blas_scene) = self.need_blas_rebuild.as_ref() else {
            panic!("BLAS not built");
        };
        if !self.need_tlas_rebuild {
            if let Some(ref mut pipeline) = self.raytrace_pipeline {
                let mut encoder = rc
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                if self.tlas_obj_to_update.len() != 0 {
                    for tlas_id in &self.tlas_obj_to_update {
                        pipeline.tlas_package[*tlas_id] = Some(wgpu::TlasInstance::new(
                            &blas_scene.blases[self.instances[*tlas_id].0 .0],
                            affine_to_rows(&self.instances[*tlas_id].1),
                            0,
                            0xff,
                        ));
                    }
                    self.tlas_obj_to_update.clear();

                    encoder.build_acceleration_structures(
                        std::iter::empty(),
                        std::iter::once(&pipeline.tlas_package),
                    );
                }

                if self.lidar_pose_buff_needs_update {
                    let mut lidar_positions = self
                        .lidars
                        .iter()
                        .map(|(_, pose)| affine_to_rows(pose))
                        .collect::<Vec<[f32; 12]>>();
                    // Padding
                    while lidar_positions.len() < 2 {
                        lidar_positions.push([2.0; 12]);
                    }

                    let lidar_position_buf =
                        rc.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Lidar Position Buffer"),
                                contents: bytemuck::cast_slice(&lidar_positions),
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                            });
                    pipeline.lidar_position_buf = lidar_position_buf;
                    self.lidar_pose_buff_needs_update = false;
                    pipeline.bind_group = rc.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipeline.pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: pipeline.storage_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::AccelerationStructure(
                                    pipeline.tlas_package.tlas(),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: pipeline.lidar_beam_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: pipeline.lidar_position_buf.as_entire_binding(),
                            },
                        ],
                    });
                }

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&pipeline.pipeline);
                    cpass.set_bind_group(0, Some(&pipeline.bind_group), &[]);
                    cpass.dispatch_workgroups(256, 1, 1);
                }

                // Sets adds copy operation to command encoder.
                // Will copy data from storage buffer on GPU to staging buffer on CPU.
                encoder.copy_buffer_to_buffer(
                    &pipeline.storage_buffer,
                    0,
                    &pipeline.staging_buffer,
                    0,
                    size,
                );

                // Submits command encoder for processing
                rc.queue.submit(Some(encoder.finish()));

                // Note that we're not calling `.await` here.
                let buffer_slice = pipeline.staging_buffer.slice(..);
                // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
                let (sender, receiver) = flume::bounded(1);
                buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

                // Poll the device in a blocking manner so that our future resolves.
                // In an actual application, `device.poll(...)` should
                // be called in an event loop or on another thread.
                rc.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

                // Awaits until `buffer_future` can be read from
                if let Ok(Ok(())) = receiver.recv_async().await {
                    // Gets contents of buffer
                    let data = buffer_slice.get_mapped_range();
                    // Since contents are got in bytes, this converts these bytes back to u32
                    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

                    // With the current interface, we have to make sure all mapped views are
                    // dropped before we unmap the buffer.
                    drop(data);
                    pipeline.staging_buffer.unmap(); // Unmaps buffer from memory
                                                     // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                                     //   delete myPointer;
                                                     //   myPointer = NULL;
                                                     // It effectively frees the memory

                    // Returns data from buffer
                    println!("{:?}", result);
                }
                return;
            }
        }

        let tlas = rc.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: self.instances.len() as u32,
        });

        let shader = rc
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rt_computer"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

        let compute_pipeline =
            rc.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rt"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        //   The source of a copy.
        let storage_buffer = rc
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&numbers),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = rc.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // TODO(arjo): Only recompute when needed.
        let lidar_beams = self
            .lidars
            .iter()
            .enumerate()
            .map(|(lidar_id, (desc, pose))| {
                let mut beams = Vec::new();
                for beam in desc.vectors.iter() {
                    // Do this in the shader itself.
                    // let direction = pose.transform_vector3(Vec3::new(beam.directions[0], beam.directions[1], beam.directions[2]));
                    beams.push(GPULidarBeamId {
                        direction: *beam,
                        lidar_id: lidar_id as u32,
                    });
                }
                beams
            })
            .flatten()
            .collect::<Vec<GPULidarBeamId>>();

        //println!("Lidar beams: {:?}", lidar_beams);

        // Create a buffer of lidar beams
        let lidar_buffer = rc
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lidar Buffer"),
                contents: bytemuck::cast_slice(&lidar_beams),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let mut lidar_positions = self
            .lidars
            .iter()
            .map(|(_, pose)| affine_to_rows(pose))
            .collect::<Vec<[f32; 12]>>();
        // Padding
        while lidar_positions.len() < 2 {
            lidar_positions.push([2.0; 12]);
        }

        let lidar_position_buf = rc
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lidar Position Buffer"),
                contents: bytemuck::cast_slice(&lidar_positions),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let compute_bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let compute_bind_group = rc.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::AccelerationStructure(&tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lidar_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lidar_position_buf.as_entire_binding(),
                },
            ],
        });

        let mut tlas_package = wgpu::TlasPackage::new(tlas);

        // TODO(arjo): Rewrite the folloring to use the instancing API
        for x in 0..self.instances.len() {
            tlas_package[x] = Some(wgpu::TlasInstance::new(
                &blas_scene.blases[self.instances[x].0 .0],
                affine_to_rows(&self.instances[x].1),
                0,
                0xff,
            ));
        }

        let mut encoder = rc
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let ve: Vec<_> = blas_scene
            .blases
            .iter()
            .map(|b| wgpu::BlasBuildEntry {
                blas: b,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &blas_scene.blas_geo_size_descs[0],
                        vertex_buffer: &blas_scene.vertex_buf,
                        first_vertex: 0,
                        vertex_stride: std::mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(&blas_scene.index_buf),
                        index_buffer_offset: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            })
            .collect();
        encoder.build_acceleration_structures(ve.iter(), std::iter::once(&tlas_package));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &compute_bind_group, &[]);
            cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }

        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

        // Submits command encoder for processing
        rc.queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        rc.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = receiver.recv_async().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            staging_buffer.unmap(); // Unmaps buffer from memory
                                    // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                    //   delete myPointer;
                                    //   myPointer = NULL;
                                    // It effectively frees the memory

            // Returns data from buffer
            println!("{:?}", result);
        }

        self.lidar_pose_buff_needs_update = false;
        self.need_tlas_rebuild = false;
        self.raytrace_pipeline = Some(RaytracePipeline {
            bind_group: compute_bind_group,
            pipeline: compute_pipeline,
            tlas_package,
            staging_buffer,
            storage_buffer,
            lidar_position_buf,
            lidar_beam_buf: lidar_buffer,
        });
    }

    /// Internal function used to build the BLAS for the scene
    fn build_blas(&self, rc: &RenderContext) -> BLASScene {
        let mut blases = Vec::new();
        let mut blas_geo_size_descs = Vec::new();

        let mut vertex_data = vec![];
        let mut index_data: Vec<u16> = vec![];
        for (vertices, indices) in self.objects.iter() {
            let mut remapping = HashMap::new();
            for v in 0..vertices.len() {
                let vert_id = vertex_data.len();
                remapping.insert(v as u16, vert_id as u16);
                vertex_data.push(vertices[v]);
            }

            for i in 0..indices.len() {
                index_data.push(remapping[&indices[i]]);
            }

            let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: vertices.len() as u32,
                index_format: Some(wgpu::IndexFormat::Uint16),
                index_count: Some(indices.len() as u32),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };
            blas_geo_size_descs.push(blas_geo_size_desc.clone());

            blases.push(rc.device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![blas_geo_size_desc.clone()],
                },
            ));
        }

        let vertex_buf = rc
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
            });

        let index_buf = rc
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&index_data),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
            });

        BLASScene {
            vertex_buf,
            index_buf,
            blases,
            blas_geo_size_descs,
        }
    }
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

pub struct RenderContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl RenderContext {
    pub async fn new(instance: Instance) -> Self {
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

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

pub fn vertex(pos: [f32; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}
