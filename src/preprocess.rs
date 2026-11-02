//! Image preparation: the RGB image the encoders see.
//!
//! Resize so the short side is `image_size`; if the image has no real alpha, BiRefNet predicts one; erode the alpha
//! by 1 px so no background-coloured fringe survives; crop a square of side `crop_margin` × the object's bounding
//! box; resize to `image_size`; composite on black.

use anyhow::{bail, Result};
use image::imageops::{self, FilterType};
use image::{RgbImage, RgbaImage};
use serde::{Deserialize, Serialize};

/// How much room the square crop leaves around the object: crop side = object bounding box × margin, so the object
/// spans 1 / margin of the image. Renders of single objects work best tight, photos with a looser crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// computer generated images of one object (crop 1.2)
    #[default]
    Synthetic,
    /// real-world photographs (crop 1.7)
    Real,
}

impl Mode {
    pub fn crop_margin(self) -> f32 {
        match self {
            Mode::Synthetic => 1.2,
            Mode::Real => 1.7,
        }
    }
}

pub fn has_real_alpha(img: &RgbaImage) -> bool {
    img.pixels().any(|p| p[3] < 255)
}

fn premultiply(img: &mut RgbaImage) {
    for p in img.pixels_mut() {
        let a = p[3] as u32;
        for c in 0..3 {
            p[c] = ((p[c] as u32 * a + 127) / 255) as u8;
        }
    }
}

fn unpremultiply(img: &mut RgbaImage) {
    for p in img.pixels_mut() {
        let a = p[3] as u32;
        if a > 0 && a < 255 {
            for c in 0..3 {
                p[c] = ((p[c] as u32 * 255 + a / 2) / a).min(255) as u8;
            }
        }
    }
}

/// Lanczos resize that weights colours by alpha (what Pillow does for RGBA images).
pub fn resize_rgba(img: &RgbaImage, w: u32, h: u32) -> RgbaImage {
    let mut p = img.clone();
    premultiply(&mut p);
    let mut r = imageops::resize(&p, w, h, FilterType::Lanczos3);
    unpremultiply(&mut r);
    r
}

/// Step 1: scale so the short side is `size`.
pub fn resize_short_side(img: &RgbaImage, size: u32) -> RgbaImage {
    let (w, h) = img.dimensions();
    let scale = size as f64 / w.min(h) as f64;
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);
    resize_rgba(img, nw, nh)
}

/// `ImageFilter.MinFilter(3)` on one channel.
fn erode3(a: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut tmp = vec![0u8; a.len()];
    for y in 0..h {
        for x in 0..w {
            let mut m = 255u8;
            for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                m = m.min(a[y * w + xx]);
            }
            tmp[y * w + x] = m;
        }
    }
    let mut out = vec![0u8; a.len()];
    for y in 0..h {
        for x in 0..w {
            let mut m = 255u8;
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                m = m.min(tmp[yy * w + x]);
            }
            out[y * w + x] = m;
        }
    }
    out
}

/// Steps 3–6, given an image whose alpha is the object matte: erode, crop around the object, resize, composite on
/// black. Returns the `size × size` RGB image.
pub fn crop_and_composite(img: &RgbaImage, size: u32, crop_margin: f32) -> Result<RgbImage> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let alpha: Vec<u8> = img.pixels().map(|p| p[3]).collect();
    let alpha = erode3(&alpha, w, h);
    let mut img = img.clone();
    for (p, a) in img.pixels_mut().zip(&alpha) {
        p[3] = *a;
    }
    let (mut l, mut t, mut r, mut b) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if alpha[y * w + x] != 0 {
                l = l.min(x);
                r = r.max(x);
                t = t.min(y);
                b = b.max(y);
            }
        }
    }
    if l == usize::MAX {
        bail!("no object found in the image (the alpha channel is empty)");
    }
    let (cx, cy) = ((l + r) as f64 / 2.0, (t + b) as f64 / 2.0);
    let half = (r - l).max(b - t) as f64 / 2.0 * crop_margin as f64;
    // PIL: int() truncates; areas outside the image are transparent
    let (x0, y0, x1, y1) = ((cx - half) as i64, (cy - half) as i64, (cx + half) as i64, (cy + half) as i64);
    let (cw, ch) = ((x1 - x0).max(1) as u32, (y1 - y0).max(1) as u32);
    let mut crop = RgbaImage::new(cw, ch);
    for y in 0..ch as i64 {
        for x in 0..cw as i64 {
            let (sx, sy) = (x + x0, y + y0);
            if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                crop.put_pixel(x as u32, y as u32, *img.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    let mut p = crop;
    premultiply(&mut p);
    let r = imageops::resize(&p, size, size, FilterType::Lanczos3);
    // premultiplied colour over black is the composite
    Ok(RgbImage::from_fn(size, size, |x, y| {
        let p = r.get_pixel(x, y);
        image::Rgb([p[0], p[1], p[2]])
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn crop_frames_the_object_and_composites_on_black() {
        // a 40 x 20 opaque red box on a transparent 100 x 100 canvas
        let mut img = RgbaImage::new(100, 100);
        for y in 40..60 {
            for x in 30..70 {
                img.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        let out = crop_and_composite(&img, 64, 1.2).unwrap();
        assert_eq!(out.dimensions(), (64, 64));
        // the middle is the object, the corners are black
        assert!(out.get_pixel(32, 32)[0] > 200);
        assert_eq!(out.get_pixel(0, 0).0, [0, 0, 0]);
        // the long side of the box spans about 1 / margin of the frame
        let row: Vec<bool> = (0..64).map(|x| out.get_pixel(x, 32)[0] > 128).collect();
        let span = row.iter().filter(|&&b| b).count() as f32 / 64.0;
        assert!((span - 1.0 / 1.2).abs() < 0.08, "span {span}");
    }

    #[test]
    fn empty_alpha_is_an_error() {
        assert!(crop_and_composite(&RgbaImage::new(16, 16), 32, 1.2).is_err());
    }

    #[test]
    fn real_alpha_is_detected() {
        let mut img = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));
        assert!(!has_real_alpha(&img));
        img.put_pixel(1, 1, Rgba([1, 2, 3, 200]));
        assert!(has_real_alpha(&img));
    }
}
