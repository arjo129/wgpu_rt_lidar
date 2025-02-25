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

@group(0)
@binding(4)
var acc_struct: acceleration_structure;

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
        let result = get_closest_point_in_cell(query_point.position, base_index);
        if result.found == 1 {
            var rq: ray_query;
            let size = length(query_point.position - base_grid[result.index].position);
            let direction =  (query_point.position - base_grid[result.index].position) / size;
            rayQueryInitialize(&rq, acc_struct, RayDesc(0x0u, 0xFFu, 0.1, 50.0, query_point.position, direction));
            rayQueryProceed(&rq);
            ///query_matches[query_index + i] = result.index;
            let intersection = rayQueryGetCommittedIntersection(&rq);
           
            if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
              
              if (intersection.t < size) {
                query_matches[query_index + i] = result.index;
              }
            }
            else {
              query_matches[query_index + i] = result.index;
            }
        }
        i = i + 1;
    }
}