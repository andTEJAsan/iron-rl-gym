"""
OpenGL 3.3+ / ModernGL Volumetric Dust Particle & LiDAR Backscatter Renderer (`sim/opengl_dust_renderer.py`)
============================================================================================================
Renders 50,000+ GPU-accelerated 3D airborne dust particles on top of MuJoCo's OpenGL RGB & Depth buffers:
  1. Turbulent curl-noise advection + Stokes settling drag + diffusive cloud expansion
  2. GLSL Vertex & Fragment Shaders (`#version 330 core`) with 4-Octave 3D Fractional Brownian Motion (fBM)
  3. OpenGL Depth-Buffer Softening ("Soft Particles") against MuJoCo's hardware Z-buffer
  4. Directional sun-light Beer-Lambert self-shadowing (Mie forward/backward scattering)
  5. 905 nm Livox Mid360S LiDAR Dust Backscatter & Voxel De-Dusting Filter

Usage:
------
```python
from sim.opengl_dust_renderer import OpenGLVolumetricDustEngine, sample_livox_lidar_with_dust_backscatter

dust_engine = OpenGLVolumetricDustEngine(width=1280, height=720, max_particles=50000)

# Inside your 50 Hz / 25 Hz simulation loop:
dust_engine.emit_dust_burst(
    source_xyz=bucket_tip_xyz,
    jet_velocity_xyz=np.array([0.4, 0.2, 1.1], dtype=np.float32),
    intensity_kg_s=poured_soil_kg_per_sec,
)
dust_engine.step_physics(dt=0.02)

# Composite OpenGL GLSL volumetric dust over MuJoCo RGB + Depth buffers:
composited_rgb = dust_engine.render_over_mujoco_frame(
    mujoco_rgb_u8=rgb_frame,
    mujoco_depth_f32=depth_frame,
    view_matrix_4x4=cam_view_mat,
    proj_matrix_4x4=cam_proj_mat,
)
```
"""

from typing import Optional
import numpy as np


# ==============================================================================
# 1. GLSL VERTEX SHADER (Projects 3D Dust Particles into Camera Clip Space)
# ==============================================================================
DUST_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_world_pos;   // 3D particle center (m)
layout(location = 1) in float in_radius;     // World-space puff radius (m)
layout(location = 2) in float in_alpha;      // Base opacity [0..1]
layout(location = 3) in float in_phase_seed; // Random per-particle turbulence seed

uniform mat4 u_view_matrix;
uniform mat4 u_proj_matrix;
uniform float u_viewport_height;

out vec3 v_world_pos;
out float v_view_depth;
out float v_alpha;
out float v_seed;
out float v_radius;

void main() {
    vec4 view_pos = u_view_matrix * vec4(in_world_pos, 1.0);
    gl_Position = u_proj_matrix * view_pos;

    v_world_pos = in_world_pos;
    v_view_depth = -view_pos.z;  // Linear eye-space depth (meters)
    v_alpha = in_alpha;
    v_seed = in_phase_seed;
    v_radius = in_radius;

    // Perspective-correct OpenGL point-sprite size in pixels:
    // size_px = (world_radius / eye_depth) * focal_scale
    float focal_len = u_proj_matrix[1][1] * (u_viewport_height * 0.5);
    gl_PointSize = clamp((in_radius * focal_len) / max(0.1, v_view_depth), 1.0, 256.0);
}
"""


# ==============================================================================
# 2. GLSL FRAGMENT SHADER (3D fBM Turbulence + Soft-Depth Blending + Mie Lighting)
# ==============================================================================
DUST_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_world_pos;
in float v_view_depth;
in float v_alpha;
in float v_seed;
in float v_radius;

uniform sampler2D u_scene_depth_tex; // MuJoCo OpenGL Hardware Depth Buffer
uniform vec2 u_resolution;
uniform vec3 u_sun_dir;              // Normalized sun direction e.g. normalize(vec3(0.3, -0.4, 0.85))
uniform vec3 u_dust_albedo;          // Dry earth/sand color e.g. vec3(0.74, 0.61, 0.45)
uniform float u_time;
uniform float u_z_near;
uniform float u_z_far;

out vec4 frag_color;

// --- 3D Hash & Value Noise for Volumetric Billowing ---
float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i + vec3(0,0,0)), hash31(i + vec3(1,0,0)), f.x),
            mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
            mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y),
        f.z
    );
}

// 4-Octave Fractional Brownian Motion (fBM) for realistic dust cloud wisps
float fbm3D(vec3 p) {
    float val = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
        val += amp * noise3D(p);
        p = p * 2.03 + vec3(17.1, 31.7, 11.3);
        amp *= 0.5;
    }
    return val;
}

// Convert hardware non-linear Z-buffer [0..1] to linear eye depth (meters)
float linearize_depth(float d) {
    float z_ndc = 2.0 * d - 1.0;
    return (2.0 * u_z_near * u_z_far) / (u_z_far + u_z_near - z_ndc * (u_z_far - u_z_near));
}

void main() {
    // 1. Map gl_PointCoord [0..1]^2 to unit sphere normal (-1..+1)
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard; // Outside spherical dust puff

    // Reconstruct 3D normal on the dust puff hemisphere
    float z_sphere = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(uv.x, -uv.y, z_sphere));

    // 2. Sample 3D fBM Turbulence to erode edges into organic dust plumes
    vec3 sample_p = v_world_pos * 4.5 + normal * 1.8 + vec3(0.0, 0.0, -u_time * 0.9) + v_seed;
    float wisp = fbm3D(sample_p);
    float radial_falloff = smoothstep(1.0, 0.15, sqrt(r2));
    float density = clamp(radial_falloff * (wisp * 1.35 - 0.25), 0.0, 1.0);
    if (density < 0.01) discard;

    // 3. Soft-Particle Depth Intersection (Prevents hard clipping with bucket/ground!)
    vec2 screen_uv = gl_FragCoord.xy / u_resolution;
    float scene_raw_z = texture(u_scene_depth_tex, screen_uv).r;
    float scene_linear_z = linearize_depth(scene_raw_z);
    float particle_frag_z = v_view_depth - z_sphere * v_radius;

    // If behind solid excavator steel or ground, discard; if close, fade smoothly
    float depth_diff = scene_linear_z - particle_frag_z;
    if (depth_diff <= 0.0) discard;
    float soft_fade = clamp(depth_diff / max(0.05, v_radius * 0.85), 0.0, 1.0);

    // 4. Directional Sun Lighting + Beer-Lambert Self-Shadowing (Mie Scattering)
    float n_dot_l = max(0.0, dot(normal, u_sun_dir));
    float transmittance = exp(-density * 1.6); // Beer's Law extinction inside thick dust
    vec3 lit_color = u_dust_albedo * (0.42 + 0.68 * n_dot_l) * mix(0.65, 1.15, transmittance);

    float final_alpha = clamp(v_alpha * density * soft_fade, 0.0, 0.88);
    frag_color = vec4(lit_color, final_alpha);
}
"""


# ==============================================================================
# 3. 3D CURL-NOISE & STOKES-DRAG DUST EMITTER + OPENGL COMPOSITOR
# ==============================================================================
class OpenGLVolumetricDustEngine:
    """
    Simulates and renders up to `max_particles` (e.g. 50,000) airborne 3D dust puffs
    spawned when the JCB 8026 bucket cuts dry earth or pours soil into the dump truck.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        max_particles: int = 50000,
        init_opengl_context: bool = True,
    ):
        self.width = width
        self.height = height
        self.max_particles = max_particles

        # Particle State Arrays (Structure-of-Arrays for fast vectorized physics updates)
        self.pos = np.zeros((max_particles, 3), dtype=np.float32)
        self.vel = np.zeros((max_particles, 3), dtype=np.float32)
        self.radius = np.zeros(max_particles, dtype=np.float32)
        self.alpha = np.zeros(max_particles, dtype=np.float32)
        self.age = np.zeros(max_particles, dtype=np.float32)
        self.lifetime = np.ones(max_particles, dtype=np.float32)
        self.seed = np.random.uniform(0.0, 100.0, size=max_particles).astype(np.float32)
        self.alive = np.zeros(max_particles, dtype=bool)
        self.write_cursor = 0
        self.time_sec = 0.0

        self.ctx = None
        if init_opengl_context:
            import moderngl

            self._mgl = moderngl
            self.ctx = moderngl.create_standalone_context(require=330)
            self.ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            self.prog = self.ctx.program(
                vertex_shader=DUST_VERTEX_SHADER,
                fragment_shader=DUST_FRAGMENT_SHADER,
            )

            # Allocate GPU Vertex Buffer Object (6 floats per particle: x, y, z, radius, alpha, seed)
            self.vbo = self.ctx.buffer(reserve=max_particles * 6 * 4)
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo, "3f 1f 1f 1f", "in_world_pos", "in_radius", "in_alpha", "in_phase_seed")],
            )

            # Offscreen Framebuffer & Scene Depth Texture (copied from MuJoCo renderer)
            self.color_tex = self.ctx.texture((width, height), 4)
            self.scene_depth_tex = self.ctx.texture((width, height), 1, dtype="f4")
            self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])

    def emit_dust_burst(
        self,
        source_xyz: np.ndarray,
        jet_velocity_xyz: np.ndarray,
        intensity_kg_s: float,
        spread_m: float = 0.25,
    ):
        """
        Spawns airborne dust particles at the bucket cutting edge or truck bed impact zone.
        Rate scales with soil cutting force (kN) or pouring mass rate (kg/s).
        """
        num_spawn = int(np.clip(intensity_kg_s * 18.0, 0, 800))
        if num_spawn <= 0:
            return

        idxs = (np.arange(num_spawn) + self.write_cursor) % self.max_particles
        self.write_cursor = (self.write_cursor + num_spawn) % self.max_particles

        self.pos[idxs] = source_xyz + np.random.normal(0.0, spread_m, size=(num_spawn, 3)).astype(np.float32)
        self.pos[idxs, 2] = np.maximum(0.03, self.pos[idxs, 2])

        radial_v = np.random.normal(0.0, 0.85, size=(num_spawn, 3)).astype(np.float32)
        radial_v[:, 2] = np.abs(radial_v[:, 2]) * 0.9 + 0.35
        self.vel[idxs] = jet_velocity_xyz + radial_v

        self.radius[idxs] = np.random.uniform(0.06, 0.14, size=num_spawn).astype(np.float32)
        self.alpha[idxs] = np.random.uniform(0.28, 0.55, size=num_spawn).astype(np.float32)
        self.age[idxs] = 0.0
        self.lifetime[idxs] = np.random.uniform(1.8, 4.2, size=num_spawn).astype(np.float32)
        self.alive[idxs] = True

    def step_physics(
        self,
        dt: float,
        wind_xyz: Optional[np.ndarray] = None,
    ):
        """
        Advances all alive dust particles using Stokes air drag, gravity settling,
        turbulent curl-noise vorticity, and diffusive cloud expansion (`dr/dt > 0`).
        """
        if wind_xyz is None:
            wind_xyz = np.array([0.35, -0.20, 0.0], dtype=np.float32)

        self.time_sec += dt
        active = np.where(self.alive)[0]
        if len(active) == 0:
            return

        self.age[active] += dt
        expired = active[self.age[active] >= self.lifetime[active]]
        self.alive[expired] = False
        active = np.where(self.alive)[0]
        if len(active) == 0:
            return

        p = self.pos[active]
        v = self.vel[active]

        # 1. Stokes Air Drag
        drag_coeff = 2.4
        v += (wind_xyz - v) * min(1.0, drag_coeff * dt)

        # 2. Gravitational Stokes Settling (~0.12 m/s downward for 30-50 um silt/dust)
        v[:, 2] -= 0.12 * dt

        # 3. Divergence-Free Turbulent Curl Vorticity
        phase = self.seed[active] + self.time_sec * 2.2
        curl_x = np.sin(p[:, 1] * 3.5 + phase) * 0.45
        curl_y = np.cos(p[:, 0] * 3.5 - phase) * 0.45
        curl_z = np.sin((p[:, 0] + p[:, 1]) * 2.5 + phase) * 0.30
        v += np.stack([curl_x, curl_y, curl_z], axis=-1).astype(np.float32) * dt

        p += v * dt
        ground_hit = p[:, 2] < 0.04
        p[ground_hit, 2] = 0.04
        v[ground_hit, 2] = np.abs(v[ground_hit, 2]) * 0.2

        self.pos[active] = p
        self.vel[active] = v

        # 4. Turbulent Diffusion: Dust puffs expand as they dissipate
        life_frac = self.age[active] / self.lifetime[active]
        self.radius[active] += 0.22 * dt
        self.alpha[active] *= (1.0 - 0.35 * dt) * np.clip(1.0 - life_frac ** 2, 0.0, 1.0)

    def render_over_mujoco_frame(
        self,
        mujoco_rgb_u8: np.ndarray,
        mujoco_depth_f32: np.ndarray,
        view_matrix_4x4: np.ndarray,
        proj_matrix_4x4: np.ndarray,
        z_near: float = 0.1,
        z_far: float = 25.0,
    ) -> np.ndarray:
        """
        Uploads the active dust particles (sorted back-to-front along camera view ray)
        to the OpenGL VBO and renders the GLSL volumetric dust shader over `mujoco_rgb_u8`.
        """
        active = np.where(self.alive)[0]
        if len(active) == 0 or self.ctx is None:
            return mujoco_rgb_u8

        # Sort active dust particles back-to-front in camera eye space for accurate alpha blending
        p_alive = self.pos[active]
        ones = np.ones((len(active), 1), dtype=np.float32)
        eye_z = (np.hstack([p_alive, ones]) @ view_matrix_4x4.T)[:, 2]
        order = active[np.argsort(eye_z)]

        vertex_data = np.column_stack(
            [
                self.pos[order],
                self.radius[order],
                self.alpha[order],
                self.seed[order],
            ]
        ).astype(np.float32)

        self.vbo.orphan(len(order) * 6 * 4)
        self.vbo.write(vertex_data.tobytes())

        rgba_bg = np.dstack([mujoco_rgb_u8, np.full((self.height, self.width), 255, dtype=np.uint8)])
        self.color_tex.write(np.flipud(rgba_bg).tobytes())
        self.scene_depth_tex.write(np.flipud(mujoco_depth_f32.astype(np.float32)).tobytes())

        self.fbo.use()
        self.scene_depth_tex.use(location=0)
        self.prog["u_scene_depth_tex"].value = 0
        self.prog["u_view_matrix"].write(view_matrix_4x4.T.astype(np.float32).tobytes())
        self.prog["u_proj_matrix"].write(proj_matrix_4x4.T.astype(np.float32).tobytes())
        self.prog["u_viewport_height"].value = float(self.height)
        self.prog["u_resolution"].value = (float(self.width), float(self.height))
        self.prog["u_sun_dir"].value = (0.31, -0.41, 0.86)
        self.prog["u_dust_albedo"].value = (0.74, 0.61, 0.45)
        self.prog["u_time"].value = float(self.time_sec)
        self.prog["u_z_near"].value = float(z_near)
        self.prog["u_z_far"].value = float(z_far)

        self.vao.render(mode=self._mgl.POINTS, vertices=len(order))

        raw_bytes = self.fbo.read(components=3)
        out_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.flipud(out_rgb).copy()


# ==============================================================================
# 4. LIVOX MID360S 905nm LIDAR DUST BACKSCATTER & VOXEL DE-DUSTING FILTER
# ==============================================================================
def sample_livox_lidar_with_dust_backscatter(
    cab_lidar_origin_xyz: np.ndarray,
    true_terrain_points_xyz: np.ndarray,
    dust_engine: OpenGLVolumetricDustEngine,
) -> np.ndarray:
    """
    Simulates 905 nm laser Mie backscatter off dense airborne dust plumes:
    When optical density (alpha / dist^2) exceeds the Livox Mid360S return threshold,
    a fraction of LiDAR rays return a false early hit on the floating dust cloud.
    """
    active = np.where(dust_engine.alive)[0]
    if len(active) == 0:
        return true_terrain_points_xyz

    dust_xyz = dust_engine.pos[active]
    dust_alpha = dust_engine.alpha[active]
    dist_to_lidar = np.linalg.norm(dust_xyz - cab_lidar_origin_xyz, axis=-1)

    backscatter_prob = np.clip((dust_alpha * 1.8) / (0.5 + dist_to_lidar ** 2), 0.0, 0.65)
    hits = np.random.uniform(0.0, 1.0, size=len(active)) < backscatter_prob
    if not np.any(hits):
        return true_terrain_points_xyz

    ghost_dust_points = dust_xyz[hits] + np.random.normal(0.0, 0.03, size=(int(np.sum(hits)), 3)).astype(np.float32)
    return np.vstack([true_terrain_points_xyz, ghost_dust_points]).astype(np.float32)


def filter_airborne_dust_from_lidar_pointcloud(
    raw_points_xyz: np.ndarray,
    min_voxel_support: int = 4,
    voxel_size_m: float = 0.12,
) -> np.ndarray:
    """
    Production Spatial Voxel Density Filter for the Livox Mid360S:
    Removes sparse airborne dust backscatter returns (which have low spatial neighbor density
    above the continuous ground surface) before projecting to `livox_elevation`.
    """
    if raw_points_xyz is None or len(raw_points_xyz) == 0:
        return raw_points_xyz

    voxel_coords = np.floor(raw_points_xyz / voxel_size_m).astype(np.int32)
    _, inverse_idx, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    solid_mask = (counts[inverse_idx] >= min_voxel_support) | (raw_points_xyz[:, 2] <= 0.38)
    return raw_points_xyz[solid_mask]
