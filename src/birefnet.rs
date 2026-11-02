//! BiRefNet (`ZhengPeng7/BiRefNet`, Swin-L backbone) background removal: `forward` returns the alpha map in [0, 1].
//! The deformable convolutions of its ASPP blocks are written out here (bilinear taps gathered one kernel
//! position at a time), as candle has no such op.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::ops::{Conv2d, LayerNorm, Linear};

/// `F.interpolate(x, size, mode="bilinear", align_corners=True)`.
fn interp(x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (_, _, xh, xw) = x.dims4()?;
    if xh == h && xw == w {
        return Ok(x.clone());
    }
    let dt = x.dtype();
    let y = x.to_dtype(DType::F32)?.contiguous()?.upsample_bilinear2d(h, w, true)?;
    Ok(y.to_dtype(dt)?)
}

fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.relu()?)
}

/// Inference-mode `BatchNorm2d` folded to a scale and a shift.
struct Bn {
    scale: Tensor,
    shift: Tensor,
}

impl Bn {
    fn load(vb: &VarBuilder, c: usize, dt: DType) -> Result<Self> {
        let g = |n: &str| -> Result<Tensor> { Ok(vb.get(c, n)?.to_dtype(DType::F32)?) };
        let scale = (g("weight")? / (g("running_var")? + 1e-5)?.sqrt()?)?;
        let shift = (g("bias")? - (g("running_mean")?.mul(&scale))?)?;
        Ok(Self { scale: scale.reshape((1, c, 1, 1))?.to_dtype(dt)?, shift: shift.reshape((1, c, 1, 1))?.to_dtype(dt)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

// -- Swin Transformer (Swin-Large preset) ------------------------------------------------------------------------

const WINDOW: usize = 12;

struct SwinBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    bias: Tensor, // [heads, N, N]
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    shift: usize,
    heads: usize,
}

impl SwinBlock {
    fn load(vb: &VarBuilder, dim: usize, heads: usize, shift: usize, index: &Tensor, dt: DType, dev: &Device) -> Result<Self> {
        let a = vb.pp("attn");
        let n = WINDOW * WINDOW;
        let table = a.get(((2 * WINDOW - 1) * (2 * WINDOW - 1), heads), "relative_position_bias_table")?.to_dtype(DType::F32)?;
        let bias = table.index_select(index, 0)?.reshape((n, n, heads))?.permute((2, 0, 1))?.contiguous()?.to_dtype(dt)?;
        Ok(Self {
            norm1: LayerNorm::load(&vb.pp("norm1"), dim, 1e-5, true, dev)?,
            qkv: Linear::load(&a.pp("qkv"), dim, 3 * dim, true, dt)?,
            proj: Linear::load(&a.pp("proj"), dim, dim, true, dt)?,
            bias,
            norm2: LayerNorm::load(&vb.pp("norm2"), dim, 1e-5, true, dev)?,
            fc1: Linear::load(&vb.pp("mlp").pp("fc1"), dim, 4 * dim, true, dt)?,
            fc2: Linear::load(&vb.pp("mlp").pp("fc2"), 4 * dim, dim, true, dt)?,
            shift,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, _, c) = x.dims3()?;
        let ws = WINDOW;
        let shortcut = x;
        let mut t = self.norm1.forward(x)?.reshape((b, h, w, c))?;
        let (pr, pb) = ((ws - w % ws) % ws, (ws - h % ws) % ws);
        if pr > 0 {
            t = t.pad_with_zeros(2, 0, pr)?;
        }
        if pb > 0 {
            t = t.pad_with_zeros(1, 0, pb)?;
        }
        let (hp, wp) = (h + pb, w + pr);
        if self.shift > 0 {
            t = t.roll(-(self.shift as i32), 1)?.roll(-(self.shift as i32), 2)?;
        }
        let (nh, nw) = (hp / ws, wp / ws);
        let nwin = nh * nw;
        let n = ws * ws;
        let win = t.reshape((b, nh, ws, nw, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b * nwin, n, c))?;

        let hd = c / self.heads;
        let qkv = self.qkv.forward(&win)?.reshape((b * nwin, n, 3, self.heads, hd))?.permute((2, 0, 3, 1, 4))?.contiguous()?;
        let (q, k, v) = (qkv.narrow(0, 0, 1)?.squeeze(0)?, qkv.narrow(0, 1, 1)?.squeeze(0)?, qkv.narrow(0, 2, 1)?.squeeze(0)?);
        let q = (q * (hd as f64).powf(-0.5))?;
        let mut attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.broadcast_add(&self.bias.unsqueeze(0)?)?;
        if let Some(m) = mask {
            attn = attn.reshape((b, nwin, self.heads, n, n))?.broadcast_add(&m.reshape((1, nwin, 1, n, n))?)?.reshape((b * nwin, self.heads, n, n))?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let o = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((b * nwin, n, c))?;
        let o = self.proj.forward(&o)?;
        let mut t = o.reshape((b, nh, nw, ws, ws, c))?.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((b, hp, wp, c))?;
        if self.shift > 0 {
            t = t.roll(self.shift as i32, 1)?.roll(self.shift as i32, 2)?;
        }
        if pr > 0 || pb > 0 {
            t = t.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()?;
        }
        let x = shortcut.add(&t.reshape((b, h * w, c))?)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        Ok(x.add(&m)?)
    }
}

/// The attention mask of the shifted-window blocks for an `h × w` token grid: `[nW, N, N]`, 0 or -100.
fn shift_mask(h: usize, w: usize, dt: DType, dev: &Device) -> Result<Tensor> {
    let ws = WINDOW;
    let shift = ws / 2;
    let (hp, wp) = (h.div_ceil(ws) * ws, w.div_ceil(ws) * ws);
    let region = |i: usize, size: usize| -> usize {
        if i < size - ws {
            0
        } else if i < size - shift {
            1
        } else {
            2
        }
    };
    let (nh, nw) = (hp / ws, wp / ws);
    let n = ws * ws;
    let mut ids = vec![0u8; nh * nw * n];
    for y in 0..hp {
        for x in 0..wp {
            let id = (region(y, hp) * 3 + region(x, wp)) as u8;
            let win = (y / ws) * nw + x / ws;
            ids[win * n + (y % ws) * ws + x % ws] = id;
        }
    }
    let mut m = vec![0f32; nh * nw * n * n];
    for w_i in 0..nh * nw {
        for i in 0..n {
            for j in 0..n {
                if ids[w_i * n + i] != ids[w_i * n + j] {
                    m[(w_i * n + i) * n + j] = -100.0;
                }
            }
        }
    }
    Ok(Tensor::from_vec(m, (nh * nw, n, n), dev)?.to_dtype(dt)?)
}

struct SwinStage {
    blocks: Vec<SwinBlock>,
    merge: Option<(LayerNorm, Linear)>,
    norm: LayerNorm,
}

struct Swin {
    patch: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<SwinStage>,
    dims: [usize; 4],
    dt: DType,
    dev: Device,
}

impl Swin {
    fn load(vb: &VarBuilder, dt: DType, dev: &Device) -> Result<Self> {
        let depths = [2, 2, 18, 2];
        let heads = [6, 12, 24, 48];
        let embed = 192;
        // relative_position_index: pairwise offsets of the tokens of one window
        let mut idx = Vec::with_capacity(WINDOW.pow(4));
        for a in 0..WINDOW * WINDOW {
            for b in 0..WINDOW * WINDOW {
                let dy = (a / WINDOW) as i64 - (b / WINDOW) as i64 + WINDOW as i64 - 1;
                let dx = (a % WINDOW) as i64 - (b % WINDOW) as i64 + WINDOW as i64 - 1;
                idx.push((dy * (2 * WINDOW as i64 - 1) + dx) as u32);
            }
        }
        let index = Tensor::from_vec(idx, WINDOW.pow(4), dev)?;
        let mut stages = Vec::new();
        let mut dims = [0; 4];
        for i in 0..4 {
            let dim = embed << i;
            dims[i] = dim;
            let l = vb.pp("layers").pp(i.to_string());
            let blocks = (0..depths[i])
                .map(|j| SwinBlock::load(&l.pp("blocks").pp(j.to_string()), dim, heads[i], if j % 2 == 0 { 0 } else { WINDOW / 2 }, &index, dt, dev))
                .collect::<Result<Vec<_>>>()?;
            let merge = if i < 3 {
                Some((LayerNorm::load(&l.pp("downsample").pp("norm"), 4 * dim, 1e-5, true, dev)?, Linear::load(&l.pp("downsample").pp("reduction"), 4 * dim, 2 * dim, false, dt)?))
            } else {
                None
            };
            stages.push(SwinStage { blocks, merge, norm: LayerNorm::load(&vb.pp(format!("norm{i}")), dim, 1e-5, true, dev)? });
        }
        Ok(Self {
            patch: Conv2d::load(&vb.pp("patch_embed").pp("proj"), 3, embed, 4, 4, 0, true, dt)?,
            patch_norm: LayerNorm::load(&vb.pp("patch_embed").pp("norm"), embed, 1e-5, true, dev)?,
            stages,
            dims,
            dt,
            dev: dev.clone(),
        })
    }

    /// `[B, 3, H, W]` → the four stage outputs (NCHW), strides 4, 8, 16, 32.
    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let x = self.patch.forward(x)?;
        let (b, c, mut h, mut w) = x.dims4()?;
        let mut x = self.patch_norm.forward(&x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?)?;
        let mut outs = Vec::new();
        for (i, st) in self.stages.iter().enumerate() {
            let mask = shift_mask(h, w, self.dt, &self.dev)?;
            for blk in &st.blocks {
                x = blk.forward(&x, h, w, if blk.shift > 0 { Some(&mask) } else { None })?;
            }
            let dim = self.dims[i];
            outs.push(st.norm.forward(&x)?.reshape((b, h, w, dim))?.permute((0, 3, 1, 2))?.contiguous()?);
            if let Some((norm, reduction)) = &st.merge {
                let t = x.reshape((b, h / 2, 2, w / 2, 2, dim))?;
                let part = |hs: usize, ws_: usize| -> Result<Tensor> { Ok(t.narrow(2, hs, 1)?.narrow(4, ws_, 1)?.reshape((b, h / 2, w / 2, dim))?) };
                let m = Tensor::cat(&[&part(0, 0)?, &part(1, 0)?, &part(0, 1)?, &part(1, 1)?], 3)?.reshape((b, (h / 2) * (w / 2), 4 * dim))?;
                x = reduction.forward(&norm.forward(&m)?)?;
                h /= 2;
                w /= 2;
            }
            crate::progress::check()?;
        }
        Ok(outs)
    }
}

// -- ASPP with deformable convolutions ---------------------------------------------------------------------------

/// `torchvision.ops.deform_conv2d` for one image, stride 1, one offset group: `x [1, C, H, W]`, `offset
/// [1, 2k², H, W]` (dy, dx per tap), `mask [1, k², H, W]`, `weight [O, C, k, k]`. Computed in f32.
fn deform_conv2d(x: &Tensor, offset: &Tensor, mask: &Tensor, weight: &Tensor, pad: usize) -> Result<Tensor> {
    let dev = x.device();
    let (_, c, h, w) = x.dims4()?;
    let (o, _, k, _) = weight.dims4()?;
    let hw = h * w;
    let xf = x.to_dtype(DType::F32)?.reshape((c, hw))?.contiguous()?;
    let offset = offset.to_dtype(DType::F32)?.reshape((k * k, 2, hw))?;
    let mask = mask.to_dtype(DType::F32)?.reshape((k * k, hw))?;
    let grid_y = Tensor::from_vec((0..hw).map(|i| (i / w) as f32).collect::<Vec<_>>(), hw, dev)?;
    let grid_x = Tensor::from_vec((0..hw).map(|i| (i % w) as f32).collect::<Vec<_>>(), hw, dev)?;
    let (hmax, wmax) = ((h - 1) as f64, (w - 1) as f64);
    let mut acc: Option<Tensor> = None;
    for i in 0..k {
        for j in 0..k {
            let t = i * k + j;
            let py = (grid_y.clone() + (i as f64 - pad as f64))?.add(&offset.narrow(0, t, 1)?.narrow(1, 0, 1)?.reshape(hw)?)?;
            let px = (grid_x.clone() + (j as f64 - pad as f64))?.add(&offset.narrow(0, t, 1)?.narrow(1, 1, 1)?.reshape(hw)?)?;
            let (y0, x0) = (py.floor()?, px.floor()?);
            let (ly, lx) = (py.sub(&y0)?, px.sub(&x0)?);
            let (hy, hx) = ((1.0 - &ly)?, (1.0 - &lx)?);
            let mut idxs = Vec::with_capacity(4);
            let mut wts = Vec::with_capacity(4);
            for (dy, dx) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
                let yy = (y0.clone() + dy)?;
                let xx = (x0.clone() + dx)?;
                let valid = yy.ge(0.0)?.to_dtype(DType::F32)?.mul(&yy.le(hmax)?.to_dtype(DType::F32)?)?.mul(&xx.ge(0.0)?.to_dtype(DType::F32)?)?.mul(&xx.le(wmax)?.to_dtype(DType::F32)?)?;
                let idx = (yy.clamp(0.0, hmax)? * w as f64)?.add(&xx.clamp(0.0, wmax)?)?.to_dtype(DType::U32)?;
                let wy = if dy == 0.0 { &hy } else { &ly };
                let wx = if dx == 0.0 { &hx } else { &lx };
                idxs.push(idx);
                wts.push(wy.mul(wx)?.mul(&valid)?);
            }
            let idx = Tensor::cat(&idxs, 0)?; // [4 HW]
            let wt = Tensor::stack(&wts, 0)?.unsqueeze(0)?; // [1, 4, HW]
            let g = xf.index_select(&idx, 1)?.reshape((c, 4, hw))?;
            let sampled = g.broadcast_mul(&wt)?.sum(1)?.broadcast_mul(&mask.narrow(0, t, 1)?)?; // [C, HW]
            let wtap = weight.narrow(2, i, 1)?.narrow(3, j, 1)?.reshape((o, c))?.to_dtype(DType::F32)?;
            let y = wtap.matmul(&sampled)?;
            acc = Some(match acc {
                Some(a) => a.add(&y)?,
                None => y,
            });
        }
    }
    Ok(acc.unwrap().reshape((1, o, h, w))?)
}

struct DeformConv {
    offset: Conv2d,
    modulator: Conv2d,
    weight: Tensor,
    pad: usize,
}

impl DeformConv {
    fn load(vb: &VarBuilder, i: usize, o: usize, k: usize, pad: usize, dt: DType) -> Result<Self> {
        Ok(Self {
            offset: Conv2d::load(&vb.pp("offset_conv"), i, 2 * k * k, k, 1, pad, true, dt)?,
            modulator: Conv2d::load(&vb.pp("modulator_conv"), i, k * k, k, 1, pad, true, dt)?,
            weight: vb.pp("regular_conv").get((o, i, k, k), "weight")?.to_dtype(DType::F32)?,
            pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let offset = self.offset.forward(x)?;
        let modulator = (candle_nn::ops::sigmoid(&self.modulator.forward(x)?)? * 2.0)?;
        Ok(deform_conv2d(x, &offset, &modulator, &self.weight, self.pad)?.to_dtype(x.dtype())?)
    }
}

struct AsppModule {
    conv: DeformConv,
    bn: Bn,
}

impl AsppModule {
    fn load(vb: &VarBuilder, i: usize, planes: usize, k: usize, pad: usize, dt: DType) -> Result<Self> {
        Ok(Self { conv: DeformConv::load(&vb.pp("atrous_conv"), i, planes, k, pad, dt)?, bn: Bn::load(&vb.pp("bn"), planes, dt)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        relu(&self.bn.forward(&self.conv.forward(x)?)?)
    }
}

struct Aspp {
    aspp1: AsppModule,
    deforms: Vec<AsppModule>,
    pool_conv: Conv2d,
    pool_bn: Bn,
    conv1: Conv2d,
    bn1: Bn,
}

impl Aspp {
    fn load(vb: &VarBuilder, c: usize, dt: DType) -> Result<Self> {
        let inter = 256;
        let sizes = [1usize, 3, 7];
        Ok(Self {
            aspp1: AsppModule::load(&vb.pp("aspp1"), c, inter, 1, 0, dt)?,
            deforms: sizes.iter().enumerate().map(|(i, &k)| AsppModule::load(&vb.pp("aspp_deforms").pp(i.to_string()), c, inter, k, k / 2, dt)).collect::<Result<_>>()?,
            pool_conv: Conv2d::load(&vb.pp("global_avg_pool").pp("1"), c, inter, 1, 1, 0, false, dt)?,
            pool_bn: Bn::load(&vb.pp("global_avg_pool").pp("2"), inter, dt)?,
            conv1: Conv2d::load(&vb.pp("conv1"), inter * (2 + sizes.len()), c, 1, 1, 0, false, dt)?,
            bn1: Bn::load(&vb.pp("bn1"), c, dt)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let mut parts = vec![self.aspp1.forward(x)?];
        for d in &self.deforms {
            parts.push(d.forward(x)?);
        }
        let pooled = x.to_dtype(DType::F32)?.mean_keepdim(2)?.mean_keepdim(3)?.to_dtype(x.dtype())?;
        let p = relu(&self.pool_bn.forward(&self.pool_conv.forward(&pooled)?)?)?;
        parts.push(p.broadcast_as((1, p.dim(1)?, h, w))?.contiguous()?);
        let y = Tensor::cat(&parts, 1)?;
        relu(&self.bn1.forward(&self.conv1.forward(&y)?)?)
    }
}

struct DecBlock {
    conv_in: Conv2d,
    bn_in: Bn,
    att: Aspp,
    conv_out: Conv2d,
    bn_out: Bn,
}

impl DecBlock {
    fn load(vb: &VarBuilder, i: usize, o: usize, dt: DType) -> Result<Self> {
        let inter = 64;
        Ok(Self {
            conv_in: Conv2d::load(&vb.pp("conv_in"), i, inter, 3, 1, 1, true, dt)?,
            bn_in: Bn::load(&vb.pp("bn_in"), inter, dt)?,
            att: Aspp::load(&vb.pp("dec_att"), inter, dt)?,
            conv_out: Conv2d::load(&vb.pp("conv_out"), inter, o, 3, 1, 1, true, dt)?,
            bn_out: Bn::load(&vb.pp("bn_out"), o, dt)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = relu(&self.bn_in.forward(&self.conv_in.forward(x)?)?)?;
        let x = self.att.forward(&x)?;
        self.bn_out.forward(&self.conv_out.forward(&x)?)
    }
}

struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    fn load(vb: &VarBuilder, i: usize, o: usize, dt: DType) -> Result<Self> {
        Ok(Self { conv1: Conv2d::load(&vb.pp("conv1"), i, 64, 3, 1, 1, true, dt)?, conv_out: Conv2d::load(&vb.pp("conv_out"), 64, o, 3, 1, 1, true, dt)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.conv1.forward(x)?)
    }
}

/// `einops: b c (hg h) (wg w) -> b (c hg wg) h w`, with `h, w` taken from `reference`.
fn image2patches(image: &Tensor, rh: usize, rw: usize) -> Result<Tensor> {
    let (b, c, hf, wf) = image.dims4()?;
    let (hg, wg) = (hf / rh, wf / rw);
    Ok(image.reshape((b, c, hg, rh, wg, rw))?.permute((0, 1, 2, 4, 3, 5))?.contiguous()?.reshape((b, c * hg * wg, rh, rw))?)
}

struct Gdt {
    conv: Conv2d,
    bn: Bn,
    attn: Conv2d,
}

impl Gdt {
    fn load(vb: &VarBuilder, name: &str, i: usize, dt: DType) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::load(&vb.pp(format!("gdt_convs_{name}")).pp("0"), i, 16, 3, 1, 1, true, dt)?,
            bn: Bn::load(&vb.pp(format!("gdt_convs_{name}")).pp("1"), 16, dt)?,
            attn: Conv2d::load(&vb.pp(format!("gdt_convs_attn_{name}")).pp("0"), 16, 1, 1, 1, 0, true, dt)?,
        })
    }

    /// `p * sigmoid(attn(relu(bn(conv(p)))))`
    fn gate(&self, p: &Tensor) -> Result<Tensor> {
        let a = self.attn.forward(&relu(&self.bn.forward(&self.conv.forward(p)?)?)?)?;
        Ok(p.broadcast_mul(&candle_nn::ops::sigmoid(&a)?)?)
    }
}

struct Decoder {
    ipt: [SimpleConvs; 5], // ipt_blk1..5
    block: [DecBlock; 4],  // decoder_block1..4
    lateral: [Conv2d; 3],  // lateral_block2..4
    gdt: [Gdt; 3],         // 2..4
    conv_out: Conv2d,
}

impl Decoder {
    fn load(vb: &VarBuilder, dt: DType) -> Result<Self> {
        let c = [3072usize, 1536, 768, 384];
        let ipt = |n: usize, i: usize, o: usize| SimpleConvs::load(&vb.pp(format!("ipt_blk{n}")), i, o, dt);
        let blk = |n: usize, i: usize, o: usize| DecBlock::load(&vb.pp(format!("decoder_block{n}")), i, o, dt);
        let lat = |n: usize, ch: usize| Conv2d::load(&vb.pp(format!("lateral_block{n}")).pp("conv"), ch, ch, 1, 1, 0, true, dt);
        Ok(Self {
            ipt: [ipt(1, 3, c[3] / 8)?, ipt(2, 48, c[2] / 8)?, ipt(3, 192, c[1] / 8)?, ipt(4, 768, c[0] / 8)?, ipt(5, 3072, c[0] / 8)?],
            block: [blk(1, c[3] + c[2] / 8, c[3] / 2)?, blk(2, c[2] + c[1] / 8, c[3])?, blk(3, c[1] + c[0] / 8, c[2])?, blk(4, c[0] + c[0] / 8, c[1])?],
            lateral: [lat(2, c[3])?, lat(3, c[2])?, lat(4, c[1])?],
            gdt: [Gdt::load(vb, "2", c[3], dt)?, Gdt::load(vb, "3", c[2], dt)?, Gdt::load(vb, "4", c[1], dt)?],
            conv_out: Conv2d::load(&vb.pp("conv_out1").pp("0"), c[3] / 2 + c[3] / 8, 1, 1, 1, 0, true, dt)?,
        })
    }

    fn forward(&self, x: &Tensor, x1: &Tensor, x2: &Tensor, x3: &Tensor, x4: &Tensor) -> Result<Tensor> {
        let dims = |t: &Tensor| -> Result<(usize, usize)> {
            let d = t.dims4()?;
            Ok((d.2, d.3))
        };
        let patches = |reference: &Tensor| -> Result<Tensor> {
            let (h, w) = dims(reference)?;
            image2patches(x, h, w)
        };
        let x4 = Tensor::cat(&[x4, &self.ipt[4].forward(&patches(x4)?)?], 1)?;
        let p4 = self.gdt[2].gate(&self.block[3].forward(&x4)?)?;
        let (h, w) = dims(x3)?;
        let p3 = interp(&p4, h, w)?.add(&self.lateral[2].forward(x3)?)?;

        let p3 = Tensor::cat(&[&p3, &self.ipt[3].forward(&patches(&p3)?)?], 1)?;
        let p3 = self.gdt[1].gate(&self.block[2].forward(&p3)?)?;
        let (h, w) = dims(x2)?;
        let p2 = interp(&p3, h, w)?.add(&self.lateral[1].forward(x2)?)?;

        let p2 = Tensor::cat(&[&p2, &self.ipt[2].forward(&patches(&p2)?)?], 1)?;
        let p2 = self.gdt[0].gate(&self.block[1].forward(&p2)?)?;
        let (h, w) = dims(x1)?;
        let p1 = interp(&p2, h, w)?.add(&self.lateral[0].forward(x1)?)?;

        let p1 = Tensor::cat(&[&p1, &self.ipt[1].forward(&patches(&p1)?)?], 1)?;
        let p1 = self.block[0].forward(&p1)?;
        let (h, w) = dims(x)?;
        let p1 = interp(&p1, h, w)?;
        let p1 = Tensor::cat(&[&p1, &self.ipt[0].forward(&patches(&p1)?)?], 1)?;
        self.conv_out.forward(&p1)
    }
}

pub struct BiRefNet {
    bb: Swin,
    squeeze: DecBlock,
    decoder: Decoder,
    dt: DType,
}

impl BiRefNet {
    pub const INPUT: usize = 1024;

    pub fn load(vb: &VarBuilder, dt: DType, dev: &Device) -> Result<Self> {
        Ok(Self {
            bb: Swin::load(&vb.pp("bb"), dt, dev)?,
            squeeze: DecBlock::load(&vb.pp("squeeze_module").pp("0"), 3072 + 384 + 768 + 1536, 3072, dt)?,
            decoder: Decoder::load(&vb.pp("decoder"), dt)?,
            dt,
        })
    }

    /// `x [1, 3, 1024, 1024]` normalised with the ImageNet statistics → the alpha map `[1, 1, 1024, 1024]` f32.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(self.dt)?;
        let (_, _, h, w) = x.dims4()?;
        let f = self.bb.forward(&x)?;
        let half = self.bb.forward(&interp(&x, h / 2, w / 2)?)?;
        let mut cat = Vec::new();
        for (a, b) in f.iter().zip(&half) {
            let (_, _, ah, aw) = a.dims4()?;
            cat.push(Tensor::cat(&[a, &interp(b, ah, aw)?], 1)?);
        }
        let (_, _, h4, w4) = cat[3].dims4()?;
        let x4 = Tensor::cat(&[&interp(&cat[0], h4, w4)?, &interp(&cat[1], h4, w4)?, &interp(&cat[2], h4, w4)?, &cat[3]], 1)?;
        let x4 = self.squeeze.forward(&x4)?;
        let logits = self.decoder.forward(&x, &cat[0], &cat[1], &cat[2], &x4)?;
        Ok(candle_nn::ops::sigmoid(&logits.to_dtype(DType::F32)?)?)
    }
}
