use glam::Vec3;
use wgpu_rt_lidar::{utils::{create_cube, dense_voxel::{execute_experimental_gpu_rrt, DenseVoxel, VoxelItem}, get_raytracing_gpu}, RayTraceScene};
use wgpu_rt_lidar::Instance;


/// Super Super experimental RRT implementation
/// No idea if its correct.
#[tokio::main]
pub async fn main() {
    let rec = rerun::RecordingStreamBuilder::new("rrt_vis")
    .spawn()
    .unwrap();

let mut voxel_grid =
DenseVoxel::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 0.0), 0.5, 10);

voxel_grid
.add_item(VoxelItem {
    position: Vec3::new(0.5, 0.5, 0.5),
    occupied: 0,
})
.unwrap();


let instance = wgpu::Instance::default();

let (_, device, queue) = get_raytracing_gpu(&instance).await;
let cube = create_cube(0.2);
let instances = (0..4).map(|x| 
    (0..4).map(move |y|
        Instance {
            asset_mesh_index: 0,
            transform: glam::Affine3A::from_rotation_translation(
                glam::Quat::from_rotation_y(0.0),
                glam::Vec3 {
                    x: 2.0,
                    y: y as f32 + 0.5,
                    z: x as f32 + 0.5,
                },
            ),
        }
)    
).flatten().collect();

let scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;
scene.visualize(&rec);
let one = execute_experimental_gpu_rrt(&device, &queue, &voxel_grid, &scene)
.await
.unwrap();
println!("{:?}", one.len());
let result: Vec<_> = one.iter().filter(|p| p.parent > 50000).collect();
println!("Valid end states: {:?}", result.len());

let result: Vec<_> = one.iter().filter(|p| p.parent != 50000).collect();
println!("Valid  {:?}", result.len());
}