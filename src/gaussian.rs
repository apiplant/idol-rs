//! The activated Gaussians and the standard 3D Gaussian Splatting PLY writer.
//!
//! `xyz` positions, `features_dc` colour (degree-0 spherical harmonics), `scaling`, `rotation` unit quaternions
//! `(w, x, y, z)` and `opacity`. The object is z-up, inside `[-0.5, 0.5]^3`.

use std::io::Write;
use std::path::Path;

use anyhow::Result;

#[derive(Default, Clone)]
pub struct Gaussians {
    pub xyz: Vec<[f32; 3]>,
    pub features_dc: Vec<[f32; 3]>,
    pub scaling: Vec<[f32; 3]>,
    pub rotation: Vec<[f32; 4]>,
    pub opacity: Vec<f32>,
}

impl Gaussians {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            xyz: Vec::with_capacity(n),
            features_dc: Vec::with_capacity(n),
            scaling: Vec::with_capacity(n),
            rotation: Vec::with_capacity(n),
            opacity: Vec::with_capacity(n),
        }
    }

    pub fn push(&mut self, xyz: [f32; 3], dc: [f32; 3], scaling: [f32; 3], rotation: [f32; 4], opacity: f32) {
        self.xyz.push(xyz);
        self.features_dc.push(dc);
        self.scaling.push(scaling);
        self.rotation.push(rotation);
        self.opacity.push(opacity);
    }

    /// Turn the object over: a proper rotation by 180 degrees about the unit `axis` (in the model's z-up frame, before
    /// the PLY's y-up conversion). About the camera's view axis it puts an upside-down object upright while the same
    /// face stays toward the input camera, so the predicted camera remains valid. Not a mirror: handedness is kept.
    pub fn turn_over(&mut self, axis: [f32; 3]) {
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt().max(1e-12);
        let a = [axis[0] / n, axis[1] / n, axis[2] / n];
        for p in &mut self.xyz {
            let d = a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
            *p = [2.0 * a[0] * d - p[0], 2.0 * a[1] * d - p[1], 2.0 * a[2] * d - p[2]];
        }
        // q' = (0, a) * q
        for q in &mut self.rotation {
            let v = [q[1], q[2], q[3]];
            let cross = [a[1] * v[2] - a[2] * v[1], a[2] * v[0] - a[0] * v[2], a[0] * v[1] - a[1] * v[0]];
            *q = [-(a[0] * v[0] + a[1] * v[1] + a[2] * v[2]), q[0] * a[0] + cross[0], q[0] * a[1] + cross[1], q[0] * a[2] + cross[2]];
        }
    }

    pub fn len(&self) -> usize {
        self.xyz.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xyz.is_empty()
    }

    /// The PLY file as bytes: z-up object turned to the y-up convention viewers use, with the inverse activations
    /// of opacity and scale clamped so saturated values never become ±inf.
    pub fn to_ply(&self) -> Vec<u8> {
        const NAMES: [&str; 17] = [
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let n = self.len();
        let mut out = Vec::with_capacity(256 + n * NAMES.len() * 4);
        write!(out, "ply\nformat binary_little_endian 1.0\nelement vertex {n}\n").unwrap();
        for name in NAMES {
            writeln!(out, "property float {name}").unwrap();
        }
        out.extend_from_slice(b"end_header\n");
        let eps = 1e-6f32;
        let h = std::f32::consts::FRAC_1_SQRT_2;
        for i in 0..n {
            let p = self.xyz[i];
            let c = self.features_dc[i];
            let s = self.scaling[i];
            let q = self.rotation[i];
            let o = self.opacity[i].clamp(eps, 1.0 - eps);
            // +90 degrees about x: (x, y, z) -> (x, -z, y); quaternion product (h, h, 0, 0) * q
            let r = [h * q[0] - h * q[1], h * q[1] + h * q[0], h * q[2] - h * q[3], h * q[3] + h * q[2]];
            let row = [
                p[0], -p[2], p[1], 0.0, 0.0, 0.0, c[0], c[1], c[2], (o / (1.0 - o)).ln(), s[0].max(eps).ln(), s[1].max(eps).ln(), s[2].max(eps).ln(), r[0], r[1], r[2], r[3],
            ];
            for v in row {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    pub fn save_ply(&self, path: &Path) -> Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(path, self.to_ply())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turning_over_is_a_rotation_of_order_two() {
        let mut g = Gaussians::default();
        g.push([0.2, -0.1, 0.4], [0.0; 3], [0.01; 3], [0.5, 0.5, 0.5, 0.5], 0.9);
        let before = g.clone();
        let axis = [0.6, 0.0, 0.8];
        g.turn_over(axis);
        // points on the axis stay, the rest are reflected through it: z-up -> upside down for a horizontal axis
        let mut up = Gaussians::default();
        up.push([0.0, 0.0, 0.5], [0.0; 3], [0.01; 3], [1.0, 0.0, 0.0, 0.0], 1.0);
        up.turn_over([1.0, 0.0, 0.0]);
        assert!((up.xyz[0][2] + 0.5).abs() < 1e-6, "{:?}", up.xyz[0]);
        // the rotation is unit length and distances from the origin are kept
        let q = g.rotation[0];
        assert!((q.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
        let r = |p: [f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((r(g.xyz[0]) - r(before.xyz[0])).abs() < 1e-6);
        // twice is the identity (q and -q are the same rotation)
        g.turn_over(axis);
        for i in 0..3 {
            assert!((g.xyz[0][i] - before.xyz[0][i]).abs() < 1e-6);
        }
        let same = g.rotation[0].iter().zip(&before.rotation[0]).all(|(a, b)| (a - b).abs() < 1e-5) || g.rotation[0].iter().zip(&before.rotation[0]).all(|(a, b)| (a + b).abs() < 1e-5);
        assert!(same, "{:?} vs {:?}", g.rotation[0], before.rotation[0]);
    }

    #[test]
    fn ply_header_and_size() {
        let mut g = Gaussians::default();
        g.push([0.1, 0.2, 0.3], [0.0; 3], [0.01; 3], [1.0, 0.0, 0.0, 0.0], 1.0);
        g.push([-0.1, 0.0, 0.5], [1.0; 3], [0.02; 3], [0.0, 1.0, 0.0, 0.0], 0.5);
        let ply = g.to_ply();
        let end = ply.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let head = std::str::from_utf8(&ply[..end]).unwrap();
        assert!(head.starts_with("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"));
        assert_eq!(head.matches("property float").count(), 17);
        assert_eq!(ply.len() - end, 2 * 17 * 4);
        let f = |i: usize| f32::from_le_bytes(ply[end + 4 * i..end + 4 * i + 4].try_into().unwrap());
        // z-up -> y-up: (x, y, z) -> (x, -z, y)
        assert_eq!((f(0), f(1), f(2)), (0.1, -0.3, 0.2));
        // a fully opaque Gaussian must not become +inf
        assert!(f(9).is_finite());
    }
}
