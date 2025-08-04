use std::{borrow::Cow, iter};

use glam::{Affine3A, Quat, Vec3, Vec4};
use rand::rand_core::le;
use wgpu::util::DeviceExt;

use crate::{affine_to_4x4rows, RayTraceScene};

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct WorkGroupParameters {
    width: u32,
    height: u32,
    depth: u32,
    num_lidar_beams: u32,
}

/// Represents a LiDAR sensor.
///
/// This struct manages the compute pipelines and buffers required for simulating a LiDAR sensor.
pub struct Lidar {
    pipeline: wgpu::ComputePipeline,
    pointcloud_pipeline: wgpu::ComputePipeline,
    ray_directions: Vec<Vec4>,
    ray_direction_gpu_buf: wgpu::Buffer,
}

impl Lidar {
    pub fn visualize_rays(&self, rec: &rerun::RecordingStream, lidar_pose: &Affine3A, name: &str) {
        let (_scale, rot, translation) = lidar_pose.to_scale_rotation_translation();
        let vectors: Vec<[f32; 3]> = self
            .ray_directions
            .iter()
            .map(|v| (rot * Vec3::new(v.x, v.y, v.z)).to_array())
            .collect();
        let origins = vec![translation.to_array(); self.ray_directions.len()];
        rec.log(
            name,
            &rerun::Arrows3D::from_vectors(vectors).with_origins(origins),
        )
        .unwrap();
    }

    /// Returns the constant value used to indicate a "no hit" from the LiDAR sensor.
    pub fn no_hit_const() -> f32 {
        10000.0
    }
    /// Creates a new LiDAR sensor.
    ///
    /// # Arguments
    ///
    /// * `device` - The `wgpu::Device` to use for creating GPU resources.
    /// * `ray_directions` - A list of `Vec3` representing the direction of each LiDAR beam.
    pub async fn new(device: &wgpu::Device, ray_directions: Vec<Vec3>) -> Self {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let ray_directions: Vec<_> = ray_directions
            .iter()
            .map(|v| Vec4::new(v.x, v.y, v.z, 0.0))
            .collect();
        let ray_direction_gpu_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lidar Buffer"),
            contents: bytemuck::cast_slice(&ray_directions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        println!("Lidar buffer size: {:?}", ray_directions.len());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lidar_computer"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        let pc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lidar_computer"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.pointcloud.wgsl"))),
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
            pointcloud_pipeline: {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("lidar"),
                    layout: None,
                    module: &pc_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            },
        }
    }

    /// Calculate the best distribution for
    fn distribute_workgroup(&self, num_points: u32, device: &wgpu::Device) -> WorkGroupParameters {
        if num_points == 0 {
            panic!("no points");
        }
        let limits = device.limits();
        // Assume these are the maximum allowed workgroup dimensions for your target GPU
        let max_workgroup_x: u32 = limits.max_compute_workgroup_size_x;
        let max_workgroup_y: u32 = limits.max_compute_workgroup_size_y;
        let max_workgroup_z: u32 = limits.max_compute_workgroup_size_z;

        if num_points > max_workgroup_x * max_workgroup_y * max_workgroup_z {
            panic!(
                "Too many points to render in a single GPU call {:?}, GPU only supports {:?}",
                num_points,
                max_workgroup_x * max_workgroup_y * max_workgroup_z
            );
        }

        let mut width = 1;
        let mut height = 1;
        let mut depth = 1;

        let num_lidar_beams = num_points;

        // Distribute across X first
        width = num_lidar_beams.min(max_workgroup_x);
        let mut remaining_beams = (num_lidar_beams + width - 1) / width; // Ceiling division

        // If there are still beams left, distribute across Y
        if remaining_beams > 1 {
            height = remaining_beams.min(max_workgroup_x);
            remaining_beams = (remaining_beams + height - 1) / height; // Ceiling division
        }

        // If there are still beams left, distribute across Z
        if remaining_beams > 1 {
            depth = remaining_beams.min(max_workgroup_x);
            // At this point, if remaining_beams > 1 after this,
            // it means total_beams cannot be covered by a single workgroup
            // within the max dimension limits. For dispatching multiple workgroups,
            // you'd typically calculate the number of workgroups needed in each dimension
            // based on a fixed workgroup size. This function focuses on *one* workgroup's dimensions.
        }

        WorkGroupParameters {
            width,
            height,
            depth,
            num_lidar_beams,
        }
    }

    /// Renders a LiDAR point cloud.
    ///
    /// This function dispatches a compute shader to trace the LiDAR beams and returns a point cloud.
    ///
    /// # Arguments
    ///
    /// * `scene` - The `RayTraceScene` to render.
    /// * `device` - The `wgpu::Device` to use.
    /// * `queue` - The `wgpu::Queue` to use for submitting commands.
    /// * `pose` - The `Affine3A` transform of the LiDAR sensor.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` containing the point cloud data, where each point is represented by 4 floats (x, y, z, intensity).
    pub async fn render_lidar_pointcloud(
        &mut self,
        scene: &RayTraceScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pose: &Affine3A,
    ) -> Vec<f32> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let compute_bind_group_layout = self.pointcloud_pipeline.get_bind_group_layout(0);
        let lidar_positions = affine_to_4x4rows(pose);

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&lidar_positions),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let work_group_params = self.distribute_workgroup(self.ray_directions.len() as u32, device);
        let work_group_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Work Group Parameters Buffer"),
            contents: bytemuck::cast_slice(&[work_group_params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.ray_directions.len() * 4 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
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
                        &scene.tlas_package,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ray_direction_gpu_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: work_group_params_buf.as_entire_binding(),
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
            cpass.set_pipeline(&self.pointcloud_pipeline);
            cpass.set_bind_group(0, Some(&compute_bind_group), &[]);
            cpass.dispatch_workgroups(self.ray_directions.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&raw_buf, 0, &staging_buffer, 0, staging_buffer.size());

        queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.poll(wgpu::PollType::wait()).unwrap();

        receiver.recv().unwrap().unwrap();

        {
            let view = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();

            drop(view);
            staging_buffer.unmap();
            return result;
        }
    }

    /// Renders the LiDAR beams and returns the hit distances.
    ///
    /// This function dispatches a compute shader to trace the LiDAR beams and returns the distance to the first hit for each beam.
    ///
    /// # Arguments
    ///
    /// * `scene` - The `RayTraceScene` to render.
    /// * `device` - The `wgpu::Device` to use.
    /// * `queue` - The `wgpu::Queue` to use for submitting commands.
    /// * `pose` - The `Affine3A` transform of the LiDAR sensor.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` containing the hit distance for each LiDAR beam.
    pub async fn render_lidar_beams(
        &mut self,
        scene: &RayTraceScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pose: &Affine3A,
    ) -> Vec<f32> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let compute_bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let lidar_positions = affine_to_4x4rows(pose);

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
                    resource: wgpu::BindingResource::AccelerationStructure(&scene.tlas_package),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ray_direction_gpu_buf.as_entire_binding(),
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

        device.poll(wgpu::PollType::wait()).unwrap();

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
