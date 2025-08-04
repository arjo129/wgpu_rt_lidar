use std::{borrow::Cow, iter};

use bytemuck_derive::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::RayTraceScene;

/// Depth camera uniforms.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthCameraUniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    width: u32,
    height: u32,
    padding: [f32; 2],
}

/// Represents a depth camera sensor.
///
/// This struct manages the compute pipelines and uniforms required for simulating a depth camera.
pub struct DepthCamera {
    pipeline: wgpu::ComputePipeline,
    pointcloud_pipeline: wgpu::ComputePipeline,
    uniforms: DepthCameraUniforms,
    width: u32,
    height: u32,
}

impl DepthCamera {
    /// Creates a new depth camera sensor.
    ///
    /// # Arguments
    ///
    /// * `device` - The `wgpu::Device` to use for creating GPU resources.
    /// * `width` - The width of the depth camera image in pixels.
    /// * `height` - The height of the depth camera image in pixels.
    /// * `fov_y` - The vertical field of view in degrees.
    /// * `_max_depth` - The maximum depth value.
    pub async fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        fov_y: f32,
        _max_depth: f32,
    ) -> Self {
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

        let pointcloud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt_computer"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.pointcloud.wgsl"))),
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
            pointcloud_pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rt"),
                layout: None,
                module: &pointcloud_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            uniforms,
            width,
            height,
        }
    }

    /// Renders a depth image from the camera's perspective.
    ///
    /// This function dispatches a compute shader to trace rays from the camera and returns a depth image.
    ///
    /// # Arguments
    ///
    /// * `scene` - The `RayTraceScene` to render.
    /// * `device` - The `wgpu::Device` to use.
    /// * `queue` - The `wgpu::Queue` to use for submitting commands.
    /// * `view_matrix` - The `Mat4` view matrix of the camera.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` containing the depth image data.
    pub async fn render_depth_camera(
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
                    resource: wgpu::BindingResource::AccelerationStructure(&scene.tlas_package),
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

    /// Renders a point cloud from the camera's perspective.
    ///
    /// This function dispatches a compute shader to trace rays and generate a point cloud.
    ///
    /// # Arguments
    ///
    /// * `scene` - The `RayTraceScene` to render.
    /// * `device` - The `wgpu::Device` to use.
    /// * `queue` - The `wgpu::Queue` to use for submitting commands.
    /// * `view_matrix` - The `Mat4` view matrix of the camera.
    ///
    /// # Returns
    ///
    /// A `Vec<Vec4>` containing the point cloud data, where each point is represented by a `Vec4` (x, y, z, w).
    pub async fn render_depth_camera_pointcloud(
        &mut self,
        scene: &RayTraceScene,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_matrix: Mat4,
    ) -> Vec<Vec4> {
        self.uniforms.view_inverse = view_matrix.inverse();

        let compute_bind_group_layout = self.pipeline.get_bind_group_layout(0);

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[self.uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.width * self.height * 4 * 4) as u64,
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
                    resource: wgpu::BindingResource::AccelerationStructure(&scene.tlas_package),
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
            cpass.set_pipeline(&self.pointcloud_pipeline);
            cpass.set_bind_group(0, Some(&compute_bind_group), &[]);
            cpass.dispatch_workgroups(self.width / 8, self.height / 8, 1);
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
            let result: Vec<Vec4> = bytemuck::cast_slice(&view).to_vec();

            drop(view);
            staging_buffer.unmap();
            return result;
        }
    }

    /// Returns the width of the depth camera image.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height of the depth camera image.
    pub fn height(&self) -> u32 {
        self.height
    }
}
