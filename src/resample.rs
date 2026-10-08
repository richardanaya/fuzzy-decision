//! Antialiased bilinear resize, matching `torch.nn.functional.interpolate`
//! with `mode="bilinear"`, `align_corners=False`, and `antialias=True`.

/// Resize an HWC f32 image. Same-size inputs are copied.
pub fn resize_hwc(
    src: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    if height == out_h && width == out_w {
        return src.to_vec();
    }
    let width_filters = axis_weights(width, out_w);
    let mut tmp = vec![0f32; height * out_w * channels];
    for y in 0..height {
        for (out_x, (xmin, weights)) in width_filters.iter().enumerate() {
            for channel in 0..channels {
                let mut acc = 0.0f64;
                for (offset, weight) in weights.iter().enumerate() {
                    let x = xmin + offset;
                    acc += src[(y * width + x) * channels + channel] as f64 * weight;
                }
                tmp[(y * out_w + out_x) * channels + channel] = acc as f32;
            }
        }
    }
    let height_filters = axis_weights(height, out_h);
    let mut out = vec![0f32; out_h * out_w * channels];
    for (out_y, (ymin, weights)) in height_filters.iter().enumerate() {
        for x in 0..out_w {
            for channel in 0..channels {
                let mut acc = 0.0f64;
                for (offset, weight) in weights.iter().enumerate() {
                    let y = ymin + offset;
                    acc += tmp[(y * out_w + x) * channels + channel] as f64 * weight;
                }
                out[(out_y * out_w + x) * channels + channel] = acc as f32;
            }
        }
    }
    out
}

/// Float resize, then round and clamp back to bytes. This can differ by about
/// one level from torchvision's uint8 resize.
pub fn resize_rgb_u8(
    src: &[u8],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let float: Vec<f32> = src.iter().map(|pixel| *pixel as f32).collect();
    resize_hwc(&float, height, width, 3, out_h, out_w)
        .into_iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

fn axis_weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inv_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_interp = support.ceil() as usize * 2 + 1;
    let mut filters = Vec::with_capacity(out_size);
    for index in 0..out_size {
        let center = scale * (index as f64 + 0.5);
        let xmin = (center - support + 0.5).trunc() as i64;
        let xmax = (center + support + 0.5).trunc() as i64;
        let xmin = xmin.max(0) as usize;
        let xmax = (xmax.max(0) as usize).min(in_size);
        let size = xmax.saturating_sub(xmin).min(max_interp);
        let mut weights = Vec::with_capacity(size);
        for offset in 0..size {
            let arg = ((offset + xmin) as f64 - center + 0.5) * inv_scale;
            let distance = arg.abs();
            weights.push(if distance < 1.0 { 1.0 - distance } else { 0.0 });
        }
        let sum: f64 = weights.iter().sum();
        if sum != 0.0 {
            for weight in &mut weights {
                *weight /= sum;
            }
        }
        filters.push((xmin, weights));
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_is_identity() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = resize_hwc(&src, 2, 3, 1, 2, 3);
        assert_eq!(out, src);
    }
}
