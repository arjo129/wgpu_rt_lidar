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

@group(0)
@binding(0)
var<storage, read_write> base_grid: array<VoxelNode>;

@group(0) 
@binding(1)
var<uniform> uniforms_base: DenseVoxelGpuParams;

@group(0)
@binding(2)
var<storage, read_write> query_grid: array<VoxelNode>; 

@group(0)
@binding(3)
var<storage, read_write> query_matches: array<u32>; 

fn to_index(pos: vec3<u32>) -> u32 {
    return (pos.x + pos.y * uniforms_base.width_steps + pos.z * uniforms_base.width_steps * uniforms_base.height_steps) * uniforms_base.max_density;
}


/// Very inefficient way to find approximate closest point in the grid
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


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var i: u32 = 0;
    let base_index = to_index(global_id);
    let query_index = to_index(global_id);
    while i < uniforms_base.max_density {
        let query_point = query_grid[query_index + i];
        if query_point.occupied == 0 {
            return;
        }
        let result = get_closest_point(query_point.position, global_id);
        if result.found == 1 {
            query_matches[query_index + i] = result.index;
        }
        i = i + 1;
    }
}