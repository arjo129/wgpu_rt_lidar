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

/// Representation for a depth camera sensor
pub struct DepthCamera {
    pipeline: wgpu::ComputePipeline,
    pointcloud_pipeline: wgpu::ComputePipeline,
    uniforms: DepthCameraUniforms,
    width: u32,
    height: u32,
}

impl DepthCamera {
    /// Create a new depth camera sensor
    pub async fn new(device: &wgpu::Device, width: u32, height: u32, fov_y: f32) -> Self {
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

    /// Render the depth camera sensor. Returns the depth image as a vector of f32.
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

    /// Render the depth camera sensor. Returns the depth image as a vector of f32.
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
            cpass.set_pipeline(&self.pointcloud_pipeline);
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
            let result: Vec<Vec4> = bytemuck::cast_slice(&view).to_vec();

            drop(view);
            staging_buffer.unmap();
            return result;
        }
    }
}
