# Raytracing LiDAR models

This crate provides a set of simple LiDAR models that can be used for raytracing on windows and linux as long as your graphics card support raytracing.

This includes graphics cards like those on the Steam Deck, the intel battlemage (untested) and any NVidia RTX card. Support for raytracing is achieved through
the use of the experimental API within wgpu.

The API for using the raytracing models is really simple and we provide integration for gazebo here:
- [TODO] Gaxebo integration