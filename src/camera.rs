//! The input camera the denoiser predicts alongside the object.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Camera {
    /// radians
    pub yaw: f64,
    /// radians
    pub pitch: f64,
    pub radius: f64,
    /// vertical field of view, degrees
    pub fov: f64,
    /// unit vector from the object to the camera in the model's (z-up) frame
    #[serde(skip)]
    pub dir: [f64; 3],
}

/// `camera_latent [5]` → yaw and pitch (radians), radius and vertical field of view (degrees).
pub fn decode_camera(l: &[f32]) -> Camera {
    let (d, inv_dist, half_width) = ([l[0] as f64, l[1] as f64, l[2] as f64], l[3] as f64, l[4] as f64);
    let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let [dx, dy, dz] = if norm > 1e-6 { [d[0] / norm, d[1] / norm, d[2] / norm] } else { [0.0, 0.0, 1.0] };
    let (radius, fov) = if inv_dist < 1e-3 {
        (1000.0, 2.0 * (half_width / 1000.0).atan().to_degrees())
    } else {
        (1.0 / inv_dist, 2.0 * (half_width * inv_dist).atan().to_degrees())
    };
    Camera { yaw: dx.atan2(dy), pitch: dz.clamp(-1.0, 1.0).asin(), radius, fov, dir: [dx, dy, dz] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_and_orthographic() {
        let c = decode_camera(&[0.0, 1.0, 0.0, 0.5, 0.25]);
        assert!((c.yaw).abs() < 1e-9 && c.pitch.abs() < 1e-9);
        assert!((c.radius - 2.0).abs() < 1e-9);
        assert!((c.fov - 2.0 * (0.25f64 * 0.5).atan().to_degrees()).abs() < 1e-6);
        let o = decode_camera(&[0.0, 0.0, 1.0, 0.0, 0.25]);
        assert_eq!(o.radius, 1000.0);
        assert!((o.pitch - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }
}
