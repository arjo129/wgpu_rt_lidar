struct DenseVoxelGpuParams {
    // Word 1
    top_right: vec3<f32>,
    width_steps: u32,
    
    // Word 2
    bottom_left: vec3<f32>,
    height_steps: u32,

    // Word 3
    max_density: u32,
    resolution: f32,
    _padding: f32,
    _padding2: f32
}


struct VoxelNode {
    position: vec3<f32>,
    occupied: u32
}

struct SearchResult {
    found: u32,
    index: u32
}

struct Tree {
    position: vec3<f32>,
    parent: u32,
}

struct State {
  x: u32,
  y: u32,
  z: u32,
  w: u32,
};

struct RandomResult {
    state: State,
    value: u32,
}

fn xorshift(inp: State) -> RandomResult {
  var state = inp; 
  var t = state.x ^ (state.x << 11);
  state.x = state.y;
  state.y = state.z;
  state.z = state.w;
  state.w = (state.w ^ (state.w >> 19)) ^ (t ^ (t >> 8));
  return RandomResult(state, state.w);
}

struct RandomPointResult {
    state: State,
    point: vec3<f32>,
}

fn random_point(inp: State) -> RandomPointResult {
    var state = xorshift(inp);
    let x = state.value;
    state = xorshift(state.state);
    let y = state.value;
    state = xorshift(state.state);
    let z = state.value;
    RandomPointResult(state.state, vec3<f32>(f32(x) , f32(y) , f32(z)));
}

@group(0)
@binding(0)
var<storage, read_write> base_grid: array<VoxelNode>;

@group(0) 
@binding(1)
var<uniform> uniforms_base: DenseVoxelGpuParams;

@group(0)
@binding(2)
var<storage, read_write> query_grid: array<State>; 

@group(0)
@binding(3)
var<storage, read_write> query_matches: array<Tree>; 

@group(0)
@binding(4)
var acc_struct: acceleration_structure;

@group(0)
@binding(5)
var<storage, read_write> found: atomic<u32>;

fn to_index(pos: vec3<u32>) -> u32 {
    return (pos.x + pos.y * uniforms_base.width_steps + pos.z * uniforms_base.width_steps * uniforms_base.height_steps) * uniforms_base.max_density;
}

/// Check in a cell for the closest point to a given position
fn get_closest_point_in_cell(pos: vec3<f32>, index: u32) -> SearchResult{
    var i: u32 = 0;
    var nearest_distance: f32 = 10000.0;
    var nearest_index: u32 = uniforms_base.max_density;
    while i < uniforms_base.max_density {
        let node = base_grid[index + i];
        if (node.occupied == 0) {
            if uniforms_base.max_density == nearest_index {
                return SearchResult(0, 0);
            }
            else {
                return SearchResult(1, nearest_index + index);
            }
        }
        let distance = length(node.position - pos);
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest_index = i;
        }
        i = i + 1;
    }
    return SearchResult(0, 0);
}



/// Very inefficient way to find approximate closest point in the grid.
/// We should be looking for a KDTree or some other data structure
/// 
fn get_closest_point(pos: vec3<f32>, starting_cell: vec3<u32>) -> SearchResult {
    var step: u32 = 0;
    var found: u32 = 0;
    let index = to_index(starting_cell + vec3<u32>(step, 0, 0));
    let result = get_closest_point_in_cell(pos, index);
    if result.found == 1 {
        return result;
    }
    step = step + 1;
    while found == 0 {
        var start_x: u32 = 0;
        if starting_cell.x > step {
            start_x = starting_cell.x - step;
        }

        var end_x: u32 = uniforms_base.width_steps;
        if starting_cell.x + step < uniforms_base.width_steps {
            end_x = starting_cell.x + step;
        }

        var start_y: u32 = 0;
        if starting_cell.y > step {
            start_y = starting_cell.y - step;
        }

        var end_y: u32 = uniforms_base.width_steps;
        if starting_cell.y + step < uniforms_base.width_steps {
            end_y = starting_cell.y + step;
        }

        var start_z: u32 = 0;
        if starting_cell.z > step {
            start_z = starting_cell.z - step;
        }

        var end_z: u32 = uniforms_base.width_steps;
        if starting_cell.y + step < uniforms_base.width_steps {
            end_z = starting_cell.z + step;
        }



        var i = start_x;
        var j = start_y;
        var k = start_z;
        while i <= end_x { 
            j = start_y;
            while j <= end_y {
                k = start_z;
                while k <= end_z {    
                    let index = to_index(vec3<u32>(i, j, k));
                    let result = get_closest_point_in_cell(pos, index);
                    if result.found == 1 {
                        return result;
                    }
                    k += 1;
                }
                j += 1;
            }
            i += step;
        }

        j = start_y;
        while j <= end_y {
            i = start_x;
            while i <= end_x {
                k = start_z;
                while k <= end_z {     
                    let index = to_index(vec3<u32>(i, j, k));
                    let result = get_closest_point_in_cell(pos, index);
                    if result.found == 1 {
                        return result;
                    }
                    k += 1;
                }
                i += 1;
            }
            j += step;
        }

        k = start_z;
        while k <= end_z {
            i = start_x;
            while i <= end_x {
                j = start_y;
                while j <= end_y {    
                    let index = to_index(vec3<u32>(i, j, k));
                    let result = get_closest_point_in_cell(pos, index);
                    if result.found == 1 {
                        return result;
                    }
                    j += 1;
                }
                i += 1;
            }
            k += step;
        }
        step = step + 1;
        if step == uniforms_base.width_steps {
            return SearchResult(0, 0);
        }
    }
    return SearchResult(0, 0);
}

fn to_f(inp: u32) -> f32 {
    return f32(inp) / 4294967296.0;
}

@compute
@workgroup_size(5,5,5)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let goal = vec3<f32>(0.0, 4.0, 3.0);
  
    var i: u32 = 0;
    let base_index = to_index(global_id);
    let query_index = to_index(global_id);
    var insert_at: u32 = 0;
    while i < uniforms_base.max_density {
        if base_grid[base_index + i].occupied == 0 {
            insert_at = i;
        }
        i+=1;
    }
    
    i = 0;
    while i < uniforms_base.max_density {
        let random = query_grid[base_index + i];
        let scale =  (uniforms_base.top_right - uniforms_base.bottom_left);
        let query_point =  vec3<f32>(to_f(random.x) * scale.x, to_f(random.y) * scale.y, to_f(random.z) * scale.z)  + uniforms_base.bottom_left;
        //query_grid[query_index + i] = random.state;
        let result = get_closest_point(query_point, global_id);
        if result.found == 1 {
            var rq: ray_query;
            let size = length(query_point - base_grid[result.index].position);
            let direction =  (query_point - base_grid[result.index].position) / size;
            rayQueryInitialize(&rq, acc_struct, RayDesc(0x0u, 0xFFu, 0.0, size, query_point, direction));
            rayQueryProceed(&rq);
            ///query_matches[query_index + i] = result.index;
            let intersection = rayQueryGetCommittedIntersection(&rq);
            if (intersection.kind == RAY_QUERY_INTERSECTION_NONE) 
            {
                //Check goal
              var rq2: ray_query;
                let size = length(query_point - goal);
                let direction =  (query_point - goal) / size;
                rayQueryInitialize(&rq2, acc_struct, RayDesc(0x0u, 0xFFu, 0.0, size, query_point, direction));
                rayQueryProceed(&rq2);
                let intersection2 = rayQueryGetCommittedIntersection(&rq2);
                if (intersection2.kind == RAY_QUERY_INTERSECTION_NONE) {
                    query_matches[base_index + insert_at] = Tree(query_point, result.index | 0XF0000000);
                    base_grid[base_index + insert_at] = VoxelNode(query_point, 1);
                    insert_at += 1;
                    atomicStore(&found, u32(1));
                    storageBarrier();
                    return;
                }
              query_matches[base_index + insert_at] = Tree(query_point, result.index);
              base_grid[base_index + insert_at] = VoxelNode(query_point, 1);
              insert_at += 1;
              storageBarrier();
              //return;
            }
        }

        let p = atomicLoad(&found);
        if p == 1 {
            return;
        }
        i = i + 1;
    }
}