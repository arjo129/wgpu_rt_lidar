use std::{borrow::Cow, iter};

use glam::{Affine3A, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::{affine_to_4x4rows, RayTraceScene};

pub struct Lidar {
    pipeline: wgpu::ComputePipeline,
    ray_directions: Vec<Vec4>,
    ray_direction_gpu_buf: wgpu::Buffer,
}

impl Lidar {
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
        }
    }

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
                    resource: wgpu::BindingResource::AccelerationStructure(
                        &scene.tlas_package.tlas(),
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
