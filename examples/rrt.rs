use core::f32;

use glam::Vec3;
use wgpu_rt_lidar::Instance;
use wgpu_rt_lidar::{
    utils::{
        create_cube,
        dense_voxel::{execute_experimental_gpu_rrt, DenseVoxel, VoxelItem},
        get_raytracing_gpu,
    },
    RayTraceScene,
};

fn to_tuple(val: glam::Vec3) -> [f32;3] {
    [val.x, val.y, val.z]
}
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
    let instances = (0..4)
        .map(|x| {
            (0..4).map(move |y| Instance {
                asset_mesh_index: 0,
                transform: glam::Affine3A::from_rotation_translation(
                    glam::Quat::from_rotation_y(0.0),
                    glam::Vec3 {
                        x: 2.0,
                        y: y as f32 + 0.5,
                        z: x as f32 + 0.5,
                    },
                ),
            })
        })
        .flatten()
        .collect();

    let scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;
    scene.visualize(&rec);
    let one = execute_experimental_gpu_rrt(&device, &queue, &voxel_grid, &scene)
        .await
        .unwrap();
    let mut points = vec![];
    let mut leaves = vec![];
    let goal = Vec3::new(4.0, 4.0, 3.0);
    let start = Vec3::new(0.5, 0.5, 0.5);
    let mut lowest_cost = f32::INFINITY;
    let mut pt = Vec3::new(0.0,0.0,0.0);
    for i in &one {
        if i.parent == 50000  {
            continue;
        }
        
        if (i.parent as usize) < one.len() {
            println!("{:?}", i.parent);
            println!("{:?}", one[i.parent as usize]);
            leaves.push((i.position.x, i.position.y, i.position.z));
        }
        else {
            println!("{:?}", i.parent & !0xf0000000);
            let cost = (i.position - goal).length() + (i.position-start).length();
            if cost < lowest_cost {
                lowest_cost = cost;
                pt = i.position;
            }
            points.push((i.position.x, i.position.y, i.position.z));
            //println!("{:?}", one[(i.parent & ~0XF0000000) as usize]);
            
        }
    }
    rec.log("path", &rerun::LineStrips3D::new([[
        to_tuple(start), to_tuple(pt), to_tuple(goal)]]));
    rec.log(
        "points",
        &rerun::Points3D::new(points.iter()).with_colors([rerun::Color::from_rgb(0, 255, 0)]),
    );

    rec.log(
        "shadow_points",
        &rerun::Points3D::new(leaves.iter()).with_colors([rerun::Color::from_rgb(255, 0, 0)]),
    );

    rec.log(
        "goal",
        &rerun::Points3D::new([(4.0, 4.0, 3.0)]).with_colors([rerun::Color::from_rgb(0, 255, 0)]).with_radii([0.2]),
    );
    println!("{:?}", one.len());
    let result: Vec<_> = one.iter().filter(|p| p.parent > 50000).collect();
    println!("Valid end states: {:?}", result.len());

    let result: Vec<_> = one.iter().filter(|p| p.parent != 50000).collect();

    //let actual_end = one.iter().filter(|p| p.parent != 50000).map(|p| if p.parent > 50000).collect::<Vec<_>>();

    println!("Valid  {:?}", result.len());
}
