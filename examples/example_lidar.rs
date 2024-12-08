use std::borrow::Cow;

use bytemuck_derive::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3};
use wgpu::{core::device::{self, queue}, util::DeviceExt};
use wgpu_rt_lidar::{vertex, LiDARRenderScene, RenderContext, Vertex};


fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1.0, -1.0, 1.0], [0, 0]),
        vertex([1.0, -1.0, 1.0], [1, 0]),
        vertex([1.0, 1.0, 1.0], [1, 1]),
        vertex([-1.0, 1.0, 1.0], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1.0, 1.0, -1.0], [1, 0]),
        vertex([1.0, 1.0, -1.0], [0, 0]),
        vertex([1.0, -1.0, -1.0], [0, 1]),
        vertex([-1.0, -1.0, -1.0], [1, 1]),
        // right (1, 0, 0)
        vertex([1.0, -1.0, -1.0], [0, 0]),
        vertex([1.0, 1.0, -1.0], [1, 0]),
        vertex([1.0, 1.0, 1.0], [1, 1]),
        vertex([1.0, -1.0, 1.0], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1.0, -1.0, 1.0], [1, 0]),
        vertex([-1.0, 1.0, 1.0], [0, 0]),
        vertex([-1.0, 1.0, -1.0], [0, 1]),
        vertex([-1.0, -1.0, -1.0], [1, 1]),
        // front (0, 1, 0)
        vertex([1.0, 1.0, -1.0], [1, 0]),
        vertex([-1.0, 1.0, -1.0], [0, 0]),
        vertex([-1.0, 1.0, 1.0], [0, 1]),
        vertex([1.0, 1.0, 1.0], [1, 1]),
        // back (0, -1, 0)
        vertex([1.0, -1.0, 1.0], [0, 0]),
        vertex([-1.0, -1.0, 1.0], [1, 0]),
        vertex([-1.0, -1.0, -1.0], [1, 1]),
        vertex([1.0, -1.0, -1.0], [0, 1]),
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

#[tokio::main]
async fn main() {
    let mut scene = LiDARRenderScene::new();
    let (vertex_data, index_data) = create_vertices();
    
    // Adding an object creates a handle
    let handle = scene.add_object(&vertex_data, &index_data);

    // We need to instantiate said handle. This allows you to store the same geometry
    // accross existing instances. So for incance if your instance is a car, you can
    // have multiple cars in the scene without duplicating the geometry. In this case we
    // are rendering the same cube.
    
    
    let side_count = 8;
    let dist = 3.0;
    for x in 0..side_count {
        for y in 0..side_count {

            let pose = Affine3A::from_rotation_translation(
                    Quat::from_rotation_y(45.9_f32.to_radians()),
                    Vec3 {
                        x: x as f32 * dist,
                        y: y as f32 * dist,
                        z: -30.0,
                    },
                );

            scene.add_instance(handle, pose);
        }
    }

    // Render context creates a new GPU instance and a new device.
    let render_context = RenderContext::new(wgpu::Instance::default()).await;
    
    // This will render the scene and return the lidar returns.
    scene.get_lidar_returns(&render_context).await;
}