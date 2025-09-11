use glam::*;
use rand::seq::IndexedMutRandom;
use rand::Rng;
use rerun::dataframe::ColumnSelector;
use rerun::external::crossbeam::queue;
use rerun::Vec4D;
use wgpu_rt_lidar::utils::dense_voxel::collision_check_step;
use wgpu_rt_lidar::utils::dense_voxel::DenseVoxel;
use wgpu_rt_lidar::utils::*;
use wgpu_rt_lidar::Instance;
use wgpu_rt_lidar::RayTraceScene;

#[tokio::main]
async fn main() {
    let instance = wgpu::Instance::default();
    let (_, device, queue) = get_raytracing_gpu(&instance).await;

    let rec = rerun::RecordingStreamBuilder::new("rrt_vis")
        .spawn()
        .unwrap();

    // Lets add a cube as an asset
    let cube = create_cube(1.0);
    let mut instances = vec![];
    let num_cubes = 30;
    let mut rng = rand::thread_rng();
    for _ in 0..num_cubes {
        let x = rng.random_range(0.1f32..10.0);
        let y = rng.random_range(0.1f32..10.0);
        let z = rng.random_range(0.1f32..10.0);
        let rot = rng.random_range(0.0f32..90.0);

        instances.push(Instance {
            asset_mesh_index: 0,
            transform: Affine3A::from_rotation_translation(
                Quat::from_rotation_y(rot.to_radians()),
                Vec3 { x, y, z },
            ),
        });
    }

    let mut scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;
    scene.visualize(&rec);

    let num_intial_random = 100;
    let mut random_pool = vec![];
    let mut initial_map = vec![0; num_intial_random];

    let goal = Vec4::new(
        rng.random_range(0.1f32..10.0),
        rng.random_range(0.1f32..10.0),
        rng.random_range(0.1f32..10.0),
        0.0,
    );
    for i in 0..num_intial_random {
        let x = rng.random_range(0.1f32..10.0);
        let y = rng.random_range(0.1f32..10.0);
        let z = rng.random_range(0.1f32..10.0);
        random_pool.push(Vec4::new(x, y, z, 0.0));
    }

    let start_time = std::time::Instant::now();
    // Initial Tree Expansion
    let result = collision_check_step(
        &device,
        &queue,
        &scene,
        &random_pool,
        &vec![Vec4::new(0.0, 0.0, 0.0, 0.0)],
        &initial_map,
    )
    .await
    .unwrap();

    println!("1st collision: {:?}", start_time.elapsed());

    // Visuallization
    let p: Vec<_> = random_pool
        .iter()
        .enumerate()
        .filter(|(index, p)| result[*index] == 1)
        .map(|(_, p)| (p.x, p.y, p.z))
        .collect();
    rec.log(
        "safe_points",
        &rerun::Points3D::new(p.iter()).with_radii((0..random_pool.len()).map(|_| 0.1)),
    );

    rec.log(
        "goal",
        &rerun::Points3D::new([rerun::Vec3D::new(goal.x, goal.y, goal.z)])
            .with_radii([0.5])
            .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
    );

    /*rec.log(
        "tree_branches",
        &rerun::Arrows3D::from_vectors(p.iter().map(|p| rerun::Vector3D::from(p)))
            .with_origins(p.iter().map(|_| (0.0, 0.0, 0.0))),
    );*/

    // Target check
    let res = collision_check_step(
        &device,
        &queue,
        &scene,
        &p.iter().map(|f| Vec4::new(f.0, f.1, f.2, 0.0)).collect(),
        &vec![goal],
        &vec![0; p.len()],
    )
    .await
    .unwrap();

    // If a target is found We Are Done
    let r = res
        .iter()
        .enumerate()
        .fold((false, 0), |res, p| ((((*p.1 == 1) || res.0), p.0)));

    if r.0 {
        let idx = r.1;

        let final_dir = (goal.x - p[idx].0, goal.y - p[idx].1, goal.z - p[idx].2);

        rec.log(
            "tree_branches",
            &rerun::Arrows3D::from_vectors([p[idx], final_dir])
                .with_origins([(0.0, 0.0, 0.0), p[idx]]),
        );
        return;
    }

    // Nearest Neighbour
    let mut existing_points = DenseVoxel::new(
        Vec3::new(11.0, 11.0, 11.0),
        Vec3::new(-1.0, -1.0, -1.0),
        0.5,
        20,
    );

    let start_index = existing_points
        .add_item(dense_voxel::VoxelItem {
            position: Vec3::new(0.0, 0.0, 0.0),
            occupied: 0,
        })
        .unwrap();

    let mut pt_indices = vec![];
    for pt in p {
        pt_indices.push(
            existing_points
                .add_item(dense_voxel::VoxelItem {
                    position: Vec3::new(pt.0, pt.1, pt.2),
                    occupied: 0,
                })
                .unwrap(),
        );
    }
}
