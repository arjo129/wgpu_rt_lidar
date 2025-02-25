use glam::Vec3;
use std::{mem::size_of_val, result, str::FromStr};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{vertex, AssetMesh, RayTraceScene, Vertex};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct VoxelItem {
    position: Vec3,
    occupied: u32,
}

pub struct DenseVoxel {
    /// Size of the voxel grid.
    top_right: Vec3,
    /// Bottom left corner of the voxel grid.
    bottom_left: Vec3,
    /// Resolution of each cell
    resolution: f32,
    /// Max number of items in each cell.
    max_density: u32,
    /// Voxel data
    data_on_cpu: Vec<VoxelItem>,
}

impl DenseVoxel {
    pub fn new(top_right: Vec3, bottom_left: Vec3, resolution: f32, max_density: u32) -> Self {
        if top_right.x < bottom_left.x || top_right.y < bottom_left.y || top_right.z < bottom_left.z
        {
            panic!("Invalid voxel grid bounds");
        }
        let num_voxel_grids = (top_right - bottom_left) / resolution;
        let num_voxel_grids = (num_voxel_grids.x.ceil() as usize
            * num_voxel_grids.y.ceil() as usize
            * num_voxel_grids.z.ceil() as usize
            * max_density as usize);
        let data_on_cpu = (0..num_voxel_grids)
            .map(|_| VoxelItem {
                position: Vec3::new(0.0, 0.0, 0.0),
                occupied: 0,
            })
            .collect();
        Self {
            top_right,
            bottom_left,
            resolution,
            max_density,
            data_on_cpu,
        }
    }

    pub fn width(&self) -> f32 {
        self.top_right.x - self.bottom_left.x
    }

    pub fn width_steps(&self) -> usize {
        (self.width() / self.resolution).ceil() as usize
    }

    pub fn height(&self) -> f32 {
        self.top_right.z - self.bottom_left.z
    }

    pub fn height_steps(&self) -> usize {
        (self.height() / self.resolution).ceil() as usize
    }

    pub fn length(&self) -> f32 {
        self.top_right.y - self.bottom_left.y
    }

    pub fn length_steps(&self) -> usize {
        (self.length() / self.resolution).ceil() as usize
    }

    pub fn capacity(&self) -> usize {
        self.width_steps() * self.height_steps() * self.length_steps() * self.max_density as usize
    }

    pub fn add_item(&mut self, item: VoxelItem) -> Result<usize, String> {
        if item.position.x < self.bottom_left.x
            || item.position.y < self.bottom_left.y
            || item.position.z < self.bottom_left.z
        {
            return Err("Out of voxel bounds".to_string());
        }
        if item.position.x > self.top_right.x
            || item.position.y > self.top_right.y
            || item.position.z > self.top_right.z
        {
            return Err("Out of voxel bounds".to_string());
        }

        let x = ((item.position.x - self.bottom_left.x) / self.resolution) as usize;
        let y = ((item.position.y - self.bottom_left.y) / self.resolution) as usize;
        let z = ((item.position.z - self.bottom_left.z) / self.resolution) as usize;

        let index = x + y * self.width_steps() + z * self.height_steps() * self.width_steps();
        let index = index * self.max_density as usize;

        for i in 0..self.max_density as usize {
            if self.data_on_cpu[index + i].occupied == 0 {
                self.data_on_cpu[index + i] = item;
                self.data_on_cpu[index + i].occupied = 1;
                return Ok(index + i);
            }
        }
        return Err("No space in voxel grid".to_string());
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (x + y * self.width_steps() + z * self.height_steps() * self.width_steps())
            * self.max_density as usize
    }

    fn from_index(&self, index: usize) -> (usize, usize, usize) {
        let z = index / (self.height_steps() * self.width_steps() * self.max_density as usize);
        let index =
            index - z * self.height_steps() * self.width_steps() * self.max_density as usize;
        let y = index / (self.width_steps() * self.max_density as usize);
        let index = index - y * self.width_steps() * self.max_density as usize;
        let x = index / self.max_density as usize;
        (x, y, z)
    }

    pub fn get_items_in_cell(&self, x: usize, y: usize, z: usize) -> Vec<VoxelItem> {
        let index = x + y * self.width_steps() + z * self.height_steps() * self.width_steps();
        let index = index * self.max_density as usize;
        let mut items = vec![];
        for i in 0..self.max_density as usize {
            if self.data_on_cpu[index + i].occupied == 1 {
                items.push(self.data_on_cpu[index + i]);
            }
        }
        items
    }

    pub fn get_items_in_cell_position(&self, position: Vec3) -> Vec<VoxelItem> {
        let x = ((position.x - self.bottom_left.x) / self.resolution) as usize;
        let y = ((position.y - self.bottom_left.y) / self.resolution) as usize;
        let z = ((position.z - self.bottom_left.z) / self.resolution) as usize;
        self.get_items_in_cell(x, y, z)
    }

    pub fn to_gpu_buffers(&self, device: &wgpu::Device) -> DenseVoxelGpuRepresentation {
        let data_on_gpu = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Voxel Grid Data"),
            contents: bytemuck::cast_slice(&self.data_on_cpu),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let dense_parameters = DenseVoxelGpuParams {
            top_right: self.top_right,
            width_steps: self.width_steps() as u32,
            bottom_left: self.bottom_left,
            height_steps: self.height_steps() as u32,
            max_density: self.max_density,
            resolution: self.resolution,
            _padding: 0.0,
            _padding2: 0.0,
        };
        let parameters = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Voxel Grid Parameters"),
            contents: bytemuck::cast_slice(&[dense_parameters]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
        DenseVoxelGpuRepresentation {
            data_on_gpu,
            parameters,
            cpu_parameters: dense_parameters,
            height_steps: self.height_steps() as u32,
            width_steps: self.width_steps() as u32,
            length_steps: self.length_steps() as u32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct DenseVoxelGpuParams {
    // Word 1
    top_right: Vec3,
    width_steps: u32,

    // Word 2
    bottom_left: Vec3,
    height_steps: u32,

    // Word 3
    max_density: u32,
    resolution: f32,
    _padding: f32,
    _padding2: f32,
}

struct DenseVoxelGpuRepresentation {
    data_on_gpu: wgpu::Buffer,
    parameters: wgpu::Buffer,
    cpu_parameters: DenseVoxelGpuParams,
    height_steps: u32,
    width_steps: u32,
    length_steps: u32,
}

impl DenseVoxelGpuRepresentation {
    fn prepare_query_points(&self, query_points: &Vec<Vec3>) -> DenseVoxel {
        let mut query_voxel = DenseVoxel::new(
            self.cpu_parameters.top_right,
            self.cpu_parameters.bottom_left,
            self.cpu_parameters.resolution,
            self.cpu_parameters.max_density,
        );
        query_points.iter().for_each(|point| {
            query_voxel
                .add_item(VoxelItem {
                    position: *point,
                    occupied: 1,
                })
                .unwrap();
        });
        query_voxel
    }
}

pub async fn query_nearest_neighbours(voxel: &DenseVoxel, points: Vec<Vec3>) -> Option<Vec<u32>> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    dense_voxel_nearest_neighbor(&device, &queue, voxel, &points).await
}

struct DenseVoxelNearestNeighbors {
    pipeline: wgpu::ComputePipeline,
    result_buffer: wgpu::Buffer,
}

impl DenseVoxelNearestNeighbors {
    fn new(device: &wgpu::Device, voxel: &DenseVoxel) -> Self {
        let cs_module = device.create_shader_module(wgpu::include_wgsl!("nn.wgsl"));
        let results = vec![0xFFFFu32; voxel.capacity()];
        let result_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Result"),
            contents: bytemuck::cast_slice(&results),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline: compute_pipeline,
            result_buffer,
        }
    }
}

async fn dense_voxel_nearest_neighbor(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    voxel: &DenseVoxel,
    query_points: &Vec<Vec3>,
) -> Option<Vec<u32>> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::include_wgsl!("nn.wgsl"));

    // Gets the size in bytes of the buffer.
    let size = (voxel.capacity() * 4) as wgpu::BufferAddress;

    let results = vec![0xFFFFu32; voxel.capacity()];
    let result_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Result"),
        contents: bytemuck::cast_slice(&results),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let base = voxel.to_gpu_buffers(device);
    let other = base
        .prepare_query_points(query_points)
        .to_gpu_buffers(device);
    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    println!("Bind group layout: {:?}", bind_group_layout);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: base.data_on_gpu.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: base.parameters.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: other.data_on_gpu.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch_workgroups(
            voxel.length_steps() as u32,
            voxel.width_steps() as u32,
            voxel.height_steps() as u32,
        ); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

async fn execute_gpu_rrt_one_iter_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    voxel: &DenseVoxel,
    query_points: &Vec<Vec3>,
    lidar: &RayTraceScene,
) -> Option<Vec<u32>> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::include_wgsl!("rrt.wgsl"));

    // Gets the size in bytes of the buffer.
    let size = (voxel.capacity() * 4) as wgpu::BufferAddress;

    let results = vec![0xFFFFu32; voxel.capacity()];
    let result_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Result"),
        contents: bytemuck::cast_slice(&results),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).
    let base = voxel.to_gpu_buffers(device);
    let other = base
        .prepare_query_points(query_points)
        .to_gpu_buffers(device);
    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    println!("Bind group layout: {:?}", bind_group_layout);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: base.data_on_gpu.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: base.parameters.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: other.data_on_gpu.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: result_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::AccelerationStructure(&lidar.tlas_package.tlas()),
            },
        ],
    });

    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch_workgroups(
            voxel.length_steps() as u32,
            voxel.width_steps() as u32,
            voxel.height_steps() as u32,
        ); // Number of cells to run, the (x,y,z) size of item being processed
    }
    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, size);

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_voxel_nn() {
    let mut voxel_grid =
        DenseVoxel::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 0.0), 0.5, 10);

    voxel_grid
        .add_item(VoxelItem {
            position: Vec3::new(0.5, 0.5, 0.5),
            occupied: 0,
        })
        .unwrap();

    voxel_grid
        .add_item(VoxelItem {
            position: Vec3::new(1.55, 1.55, 1.55),
            occupied: 0,
        })
        .unwrap();

    let target = voxel_grid
        .add_item(VoxelItem {
            position: Vec3::new(1.6, 1.6, 1.6),
            occupied: 0,
        })
        .unwrap();

    let items = voxel_grid.get_items_in_cell(1, 1, 1);
    assert_eq!(items.len(), 1);

    let items = voxel_grid.get_items_in_cell_position(Vec3::new(1.6, 1.6, 1.6));
    assert_eq!(items.len(), 2);

    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let internal_index = voxel_grid.index(i, j, k);
                let (x, y, z) = voxel_grid.from_index(internal_index);
                assert_eq!(i, x);
                assert_eq!(j, y);
                assert_eq!(k, z);
            }
        }
    }

    let queries = vec![Vec3::new(1.65, 1.65, 1.65)];
    let times_now = std::time::Instant::now();
    let result = query_nearest_neighbours(&voxel_grid, queries)
        .await
        .unwrap();
    println!("Time taken: {:?}", times_now.elapsed());
    let result: Vec<_> = result.iter().filter(|p| **p != 0xFFFFu32).collect();
    assert_eq!(result.len(), 1);
    assert_eq!(*result[0], target as u32);
    //run().await;
}

#[cfg(test)]
#[tokio::test]
async fn test_voxel_rrt() {
    use crate::utils::{create_cube, get_raytracing_gpu};

    let mut voxel_grid =
        DenseVoxel::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 0.0), 0.5, 10);

    voxel_grid
        .add_item(VoxelItem {
            position: Vec3::new(0.5, 0.5, 0.5),
            occupied: 0,
        })
        .unwrap();

    voxel_grid
        .add_item(VoxelItem {
            position: Vec3::new(1.55, 1.55, 1.55),
            occupied: 0,
        })
        .unwrap();

    let target = voxel_grid
        .add_item(VoxelItem {
            position: Vec3::new(1.6, 1.6, 1.6),
            occupied: 0,
        })
        .unwrap();

    let items = voxel_grid.get_items_in_cell(1, 1, 1);
    assert_eq!(items.len(), 1);

    let items = voxel_grid.get_items_in_cell_position(Vec3::new(1.6, 1.6, 1.6));
    assert_eq!(items.len(), 2);

    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let internal_index = voxel_grid.index(i, j, k);
                let (x, y, z) = voxel_grid.from_index(internal_index);
                assert_eq!(i, x);
                assert_eq!(j, y);
                assert_eq!(k, z);
            }
        }
    }

    let instance = wgpu::Instance::default();

    let (_, device, queue) = get_raytracing_gpu(&instance).await;
    let cube = create_cube(0.1);
    let instances = vec![crate::Instance {
        asset_mesh_index: 0,
        transform: glam::Affine3A::from_rotation_translation(
            glam::Quat::from_rotation_y(0.0),
            glam::Vec3 {
                x: 1.75,
                y: 1.75,
                z: 1.75,
            },
        ),
    }];

    let scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;

    let query_points = vec![
        Vec3::new(1.65, 1.65, 1.65),
        Vec3::new(1.85, 1.85, 1.85),
        Vec3::new(1.58, 1.58, 1.58),
    ];
    let one = execute_gpu_rrt_one_iter_inner(&device, &queue, &voxel_grid, &query_points, &scene)
        .await
        .unwrap();
    let result: Vec<_> = one.iter().filter(|p| **p != 0xFFFFu32).collect();
    assert_eq!(result.len(), 1);
    assert_eq!(*result[0], target as u32);
    //run().await;
}
