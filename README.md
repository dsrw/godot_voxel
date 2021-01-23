Voxel Tools for Godot
=========================

A C++ module for creating volumetric worlds in Godot Engine.

![Blocky screenshot](doc/source/images/blocky_screenshot.png)
![Smooth screenshot](doc/source/images/smooth_screenshot.png)
![Textured screenshot](doc/source/images/textured-terrain.jpg)

Features
---------------------------

- Realtime editable, 3D based terrain (Unlike a heightmap based terrain, this allows for overhangs, tunnels, and user creation/destruction)
- Physics based collision and raycast support
- Infinite terrains made by paging sections in and out
- Voxel data is streamed from a variety of sources, which includes the ability to write your own generators
- Minecraft-style blocky voxel terrain, with multiple materials and baked ambient occlusion
- Smooth terrain using Transvoxel
- Levels of detail for smooth terrain
- Voxel storage using 8-bit or 16-bit channels for any general purpose


What This Module Doesn't Provide
-----------------------------------

- Levels of detail for blocky terrain
- Game specific features such as cave generation or procedural trees (though it might include tools to help doing them)
- Editor tools (only a few things are exposed)
- Import and export of voxel formats


How To Install And Use
-------------------------

Voxel Tools is a custom C++ module for Godot 3.1+. It must be compiled into the engine to work.

### Prebuilt binaries

Ready-to-use versions of Godot including the module exist, though they might not be 100% up to date.

- Tokisan Games (editor and export templates)
    - [Tokisan Games binaries](http://tokisan.com/godot-binaries)

- Master builds (editor only)
    - Windows: https://github.com/Zylann/godot_voxel/actions?query=workflow%3A%22%F0%9F%9A%AA+Windows+Builds%22
    - Linux: https://github.com/Zylann/godot_voxel/actions?query=workflow%3A%22%F0%9F%90%A7+Linux+Builds%22
    - Warning, these can be unstable, as they come from the very latest commits. Click on one of the recent commits and a build artifact should be available.


### Compiling

Compiling the source yourself is the best way to get your own version and export template.
Please see the [documentation](https://voxel-tools.readthedocs.io/en/latest/getting_the_module/) for instructions.


### Documentation

- [Main documentation](https://voxel-tools.readthedocs.io/en/latest/)

- Zylann's demos: [Zylann's demos](https://github.com/Zylann/voxelgame)
- TinmanJuggernaut's demos: [TinmanJuggernaut's demo](https://github.com/tinmanjuggernaut/voxelgame)


Roadmap
---------

These are some ideas that may or may not be implemented in the future:

* Instancing for foliage and rocks
* Texturing on smooth terrain
* Editor preview and authoring
* Improving LOD performance
* Other meshing algorithms (e.g. dual contouring)
* GPU offloading (Maybe when Godot 4+ supports compute shaders)


Supporters
-----------

This module is a non-profit project developped by voluntary contributors. The following is the list of the current donors.
Thanks for your support :)

### Supporters

```
wacyym
Sergey Lapin (slapin)
Jonas (NoFr1ends)
lenis0012
Phyronnaz
RonanZe
furtherorbit
```

