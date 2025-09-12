use std::collections::hash_map;
use std::collections::HashMap;
use std::io::Write;

use glam::*;
use rand::seq::SliceRandom;
use rand::Rng;
use wgpu_rt_lidar::utils::dense_voxel::collision_check_step;
use wgpu_rt_lidar::utils::dense_voxel::dense_voxel_nearest_neighbor;
use wgpu_rt_lidar::utils::dense_voxel::query_nearest_neighbours;
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

    let mut rng = rand::thread_rng();

    let maze_dim = 11;
    let mut maze = vec![vec![vec![1; maze_dim]; maze_dim]; maze_dim];

    generate_maze(&mut maze, (1, 1, 1));

    // Randomly remove some interior walls to make the maze more porous
    let wall_removal_probability = 0.90;
    for x in 0..maze_dim {
        for y in 0..maze_dim {
            for z in 0..maze_dim {
                if maze[x][y][z] == 1 && rng.gen::<f32>() < wall_removal_probability {
                    maze[x][y][z] = 0;
                }
            }
        }
    }

    // Create an entrance
    maze[0][1][1] = 0;

    // Create an exit
    let mut exit_found = false;
    for y in (1..maze_dim - 1).rev() {
        for z in (1..maze_dim - 1).rev() {
            if maze[maze_dim - 2][y][z] == 0 {
                maze[maze_dim - 1][y][z] = 0;
                exit_found = true;
                break;
            }
        }
        if exit_found {
            break;
        }
    }

    let goal = Vec4::new(
        rng.random_range(0.1f32..10.0),
        rng.random_range(0.1f32..10.0),
        rng.random_range(0.1f32..10.0),
        0.0,
    );
    let mut file = std::fs::File::create("obstacles.txt").unwrap();
    file.write_all("start: 0.0 0.0 0.0\n".as_bytes());

    file.write_all(format!("goal: {} {} {}\n", goal.x, goal.y, goal.z).as_bytes());
    for x in 1..maze_dim {
        for y in 1..maze_dim {
            for z in 1..maze_dim {
                if maze[x][y][z] == 1 {
                    instances.push(Instance {
                        asset_mesh_index: 0,
                        transform: Affine3A::from_translation(Vec3::new(
                            x as f32, y as f32, z as f32,
                        )),
                    });
                    file.write_all(format!("obstacle: {} {} {}\n", x, y, z).as_bytes());
                }
            }
        }
    }

    let scene = RayTraceScene::new(&device, &queue, &vec![cube], &instances).await;
    scene.visualize(&rec);

    let num_intial_random = 1000;
    let mut random_pool = vec![];
    let initial_map = vec![0; num_intial_random];

    for _ in 0..num_intial_random {
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
        .filter(|(index, _p)| result[*index] == 1)
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

    let _start_index = existing_points
        .add_item(dense_voxel::VoxelItem {
            position: Vec3::new(0.0, 0.0, 0.0),
            occupied: 0,
        })
        .unwrap();

    let mut pt_indices = HashMap::new();
    for (id, pt) in p.iter().enumerate() {
        let assigned_index = existing_points
            .add_item(dense_voxel::VoxelItem {
                position: Vec3::new(pt.0, pt.1, pt.2),
                occupied: 0,
            })
            .unwrap();
        pt_indices.insert(assigned_index, id);
        //println!("{}", pt_indices[pt_indices.len() - 1])
    }

    // Point generation
    let mut random_pool2 = vec![];
    for _ in 0..100 {
        let x = rng.random_range(0.1f32..10.0);
        let y = rng.random_range(0.1f32..10.0);
        let z = rng.random_range(0.1f32..10.0);
        random_pool2.push(Vec3::new(x, y, z));
    }

    let x = query_nearest_neighbours(&existing_points, random_pool2)
        .await
        .unwrap();
    let found: Vec<_> = x.iter().filter(|p| **p != 0xFFFFu32).collect();
    println!("p {:?}", found)
}

fn generate_maze(maze: &mut Vec<Vec<Vec<i32>>>, start: (usize, usize, usize)) {
    let mut stack = vec![start];
    let mut rng = rand::thread_rng();

    while let Some((x, y, z)) = stack.pop() {
        maze[x][y][z] = 0;

        let mut neighbors = vec![];
        if x > 1 && maze[x - 2][y][z] == 1 {
            neighbors.push((x - 2, y, z, x - 1, y, z));
        }
        if x < maze.len() - 2 && maze[x + 2][y][z] == 1 {
            neighbors.push((x + 2, y, z, x + 1, y, z));
        }
        if y > 1 && maze[x][y - 2][z] == 1 {
            neighbors.push((x, y - 2, z, x, y - 1, z));
        }
        if y < maze[0].len() - 2 && maze[x][y + 2][z] == 1 {
            neighbors.push((x, y + 2, z, x, y + 1, z));
        }
        if z > 1 && maze[x][y][z - 2] == 1 {
            neighbors.push((x, y, z - 2, x, y, z - 1));
        }
        if z < maze[0][0].len() - 2 && maze[x][y][z + 2] == 1 {
            neighbors.push((x, y, z + 2, x, y, z + 1));
        }

        if !neighbors.is_empty() {
            neighbors.shuffle(&mut rng);
            for (nx, ny, nz, wx, wy, wz) in neighbors {
                if maze[nx][ny][nz] == 1 {
                    maze[wx][wy][wz] = 0;
                    stack.push((x, y, z));
                    stack.push((nx, ny, nz));
                    break;
                }
            }
        }
    }
}
