use std::{collections::HashMap, iter};

use bytemuck_derive::{Pod, Zeroable};
use glam::Affine3A;
use rerun::components::RotationAxisAngle;
use wgpu::util::DeviceExt;
pub mod depth_camera;
pub mod lidar;
pub mod utils;

/// Helper function to convert an affine matrix to a 4x3 row matrix.
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

/// Helper function to convert an affine matrix to a 4x4 row matrix.
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
/// A simple vertex with a position.
/// This is used for loading mesh data into the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

/// A simple function to create a vertex with a position.
pub fn vertex(pos: [f32; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1.0],
        _tex_coord: [0.0, 0.0],
    }
}

/// Representation of a mesh asset.
#[derive(Clone, Debug)]
pub struct AssetMesh {
    pub vertex_buf: Vec<Vertex>,
    pub index_buf: Vec<u16>,
}

/// Representation of an instance of a mesh asset.
#[derive(Clone, Debug)]
pub struct Instance {
    pub asset_mesh_index: usize,
    pub transform: Affine3A,
}

/// A ray tracing scene. Use this struct to create a scene which can be raytraced
/// using any hardware accelerated raytracing backend. You can load meshes and instances
/// to create a scene.
pub struct RayTraceScene {
    pub(crate) vertex_buf: wgpu::Buffer,
    pub(crate) index_buf: wgpu::Buffer,
    pub(crate) blas: Vec<wgpu::Blas>,
    pub(crate) tlas_package: wgpu::TlasPackage,
    assets: Vec<AssetMesh>,
    instances: Vec<Instance>,
}

impl RayTraceScene {
    /// Create a new ray tracing scene. Requires a device, queue, list of assets, and list of instances.
    pub async fn new(
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

        let mut geometry_desc_sizes = vec![];
        let mut blas = vec![];
        for asset in assets {
            geometry_desc_sizes.push(wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_count: asset.vertex_buf.len() as u32,
                vertex_format: wgpu::VertexFormat::Float32x3,
                index_count: Some(asset.index_buf.len() as u32),
                index_format: Some(wgpu::IndexFormat::Uint16),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            });

            blas.push(device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: geometry_desc_sizes.clone(),
                },
            ));
        }

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
                        first_index: Some(start_indices_address[index] as u32),
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
            assets: assets.clone(),
            instances: instances.clone(),
        }
    }

    /// Set the transform of instances within a scene.
    pub async fn set_transform(
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
        // Warning: SLOW!
        self.instances = update_instance.clone();

        Ok(())
    }

    /// Visualize the scene in rerun.
    pub fn visualize(&self, rerun: &rerun::RecordingStream) {
        // TODO
        for (idx, mesh) in self.assets.iter().enumerate() {
            let vertex: Vec<_> = mesh
                .vertex_buf
                .iter()
                .map(|a| [a._pos[0], a._pos[1], a._pos[2]])
                .collect();
            let indices: Vec<_> = mesh
                .index_buf
                .chunks(3)
                .map(|a| [a[0] as u32, a[1] as u32, a[2] as u32])
                .collect();
            rerun.log(
                format!("mesh_{}", idx),
                &rerun::Mesh3D::new(vertex).with_triangle_indices(indices),
            );
        }

        let mut instance_map = HashMap::new();
        for (idx, instance) in self.instances.iter().enumerate() {
            let translations = [
                instance.transform.translation.x,
                instance.transform.translation.y,
                instance.transform.translation.z,
            ];
            let rotation = glam::Quat::from_mat3a(&instance.transform.matrix3);
            let rotation =
                rerun::Quaternion::from_xyzw([rotation.x, rotation.y, rotation.z, rotation.w]);
            let Some(mesh_idx) = instance_map.get_mut(&instance.asset_mesh_index) else {
                instance_map.insert(idx, vec![(translations, rotation)]);
                continue;
            };
            mesh_idx.push((translations, rotation));
        }

        for (idx, transform) in instance_map.iter() {
            let translations = transform.iter().map(|f| f.0);
            let rotations = transform.iter().map(|f| f.1);
            rerun.log(
                format!("mesh_{}", idx),
                &rerun::InstancePoses3D::new()
                    .with_translations(translations)
                    .with_quaternions(rotations),
            );
        }
    }
}
