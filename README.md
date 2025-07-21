# Raytracing LiDAR models

This crate provides a set of simple LiDAR models that can be used for raytracing on windows and linux as long as your graphics card support raytracing.

This includes graphics cards like those on the Steam Deck, the intel battlemage (untested) and any NVidia RTX card. Support for raytracing is achieved through
the use of the experimental API within `wgpu` which in turn wraps Vulkan and Direct3D.

The API for using the raytracing models is really simple and we provide integration for gazebo here:
- [Gazebo integration](https://github.com/arjo129/gz_wgpu_rt_lidar)
- [TODO] Bevy integration

If you'd like to use this take a look at `multi_sensor.rs`

## Future improvements

Currently this relies on a seperate GPU call per sensor. It should be possible to Batch multiple sensor calls from the same scene. 
However, there will need to be some work done to accomodate the different 

## API Stability Guarantees

NONE. Currently in active development.