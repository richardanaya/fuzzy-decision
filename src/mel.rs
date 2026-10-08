//! NeMo log-mel front end from `audio.py`.
//!
//! 16 kHz mono, pre-emphasis 0.97, 512-point STFT, 400-point Hann window,
//! hop 160, Slaney mel with 128 bins, log, then per-feature sample
//! normalization. Clips are cut to 30 s and padded to 0.5 s before this.

use std::sync::OnceLock;

use realfft::RealFftPlanner;

pub const SAMPLE_RATE: u32 = 16_000;
pub const MIN_SAMPLES: usize = 8_000;
pub const MAX_SAMPLES: usize = 16_000 * 30;
const N_FFT: usize = 512;
const WINDOW: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
const PRE_EMPHASIS: f32 = 0.97;

pub struct LogMel {
    /// `(n_mels, frames)`, feature-major, including the masked final frame.
    pub features: Vec<f32>,
    pub n_mels: usize,
    pub frames: usize,
    pub valid: usize,
}

pub fn prepare_waveform(samples: &[f32]) -> Vec<f32> {
    let mut wave: Vec<f32> = samples.iter().take(MAX_SAMPLES).copied().collect();
    if wave.len() < MIN_SAMPLES {
        wave.resize(MIN_SAMPLES, 0.0);
    }
    wave
}

pub fn log_mel(wave: &[f32]) -> LogMel {
    let n = wave.len();
    let valid = n / HOP;
    let mut emphasized = Vec::with_capacity(n);
    emphasized.push(wave[0]);
    for index in 1..n {
        emphasized.push(wave[index] - PRE_EMPHASIS * wave[index - 1]);
    }
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; n + N_FFT];
    padded[pad..pad + n].copy_from_slice(&emphasized);
    let stft_frames = n / HOP + 1;
    let n_freq = N_FFT / 2 + 1;
    let window = hann();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut spectrum = fft.make_output_vec();
    let mut power = vec![0.0f32; n_freq * stft_frames];
    let mut frame = vec![0.0f32; N_FFT];
    for index in 0..stft_frames {
        let start = index * HOP;
        for bin in 0..N_FFT {
            frame[bin] = padded[start + bin] * window[bin];
        }
        fft.process(&mut frame, &mut spectrum).expect("fft 512");
        for (freq, value) in spectrum.iter().enumerate() {
            power[freq * stft_frames + index] = value.norm_sqr();
        }
    }
    let bank = filterbank();
    let mut mel = vec![0.0f32; N_MELS * stft_frames];
    for bin in 0..N_MELS {
        for frame_index in 0..stft_frames {
            let mut acc = 0.0f32;
            for freq in 0..n_freq {
                acc += bank[bin * n_freq + freq] * power[freq * stft_frames + frame_index];
            }
            mel[bin * stft_frames + frame_index] = (acc + 2f32.powi(-24)).ln();
        }
    }
    let count = valid as f32;
    for bin in 0..N_MELS {
        // PyTorch's CPU sum accumulates in 16-wide lanes. A sequential sum
        // disagrees in the last bits, and a silent clip (every bin equal)
        // turns that into an O(1) difference after dividing by a tiny std.
        let mut samples = Vec::with_capacity(valid);
        for frame_index in 0..valid {
            samples.push(mel[bin * stft_frames + frame_index]);
        }
        let mean = torch_sum(&samples) / count;
        let squares: Vec<f32> = samples
            .into_iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .collect();
        let mut std = (torch_sum(&squares) / (count - 1.0)).sqrt();
        if !std.is_finite() {
            std = 0.0;
        }
        std += 1e-5;
        for frame_index in 0..stft_frames {
            let slot = bin * stft_frames + frame_index;
            if frame_index < valid {
                mel[slot] = (mel[slot] - mean) / std;
            } else {
                mel[slot] = 0.0;
            }
        }
    }
    LogMel {
        features: mel,
        n_mels: N_MELS,
        frames: stft_frames,
        valid,
    }
}

/// Sum the way PyTorch's CPU kernel does: 16-wide lanes, then a scalar tail.
fn torch_sum(values: &[f32]) -> f32 {
    const WIDTH: usize = 16;
    let mut lanes = [0.0f32; WIDTH];
    let blocks = values.len() / WIDTH;
    for block in 0..blocks {
        let base = block * WIDTH;
        for lane in 0..WIDTH {
            lanes[lane] += values[base + lane];
        }
    }
    let mut acc = 0.0f32;
    for lane in lanes {
        acc += lane;
    }
    for value in &values[blocks * WIDTH..] {
        acc += *value;
    }
    acc
}

fn hann() -> &'static [f32] {
    static WINDOW_CACHE: OnceLock<Vec<f32>> = OnceLock::new();
    WINDOW_CACHE.get_or_init(|| {
        let mut window = vec![0.0; N_FFT];
        let offset = (N_FFT - WINDOW) / 2;
        for index in 0..WINDOW {
            let phase = 2.0 * std::f32::consts::PI * index as f32 / (WINDOW - 1) as f32;
            window[offset + index] = 0.5 * (1.0 - phase.cos());
        }
        window
    })
}

fn filterbank() -> &'static [f32] {
    static BANK: OnceLock<Vec<f32>> = OnceLock::new();
    BANK.get_or_init(|| slaney(SAMPLE_RATE as f64, N_FFT, N_MELS))
}

fn slaney(sample_rate: f64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let log_step = 6.4f64.ln() / 27.0;
    let min_log_mel = min_log_hz / f_sp;
    let hz_to_mel = |freq: f64| {
        if freq >= min_log_hz {
            min_log_mel + (freq / min_log_hz).ln() / log_step
        } else {
            freq / f_sp
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel >= min_log_mel {
            min_log_hz * (log_step * (mel - min_log_mel)).exp()
        } else {
            f_sp * mel
        }
    };
    let n_freq = 1 + n_fft / 2;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);
    let mut mel_f = vec![0.0f64; n_mels + 2];
    for (index, slot) in mel_f.iter_mut().enumerate() {
        let mel = mel_min + (mel_max - mel_min) * index as f64 / (n_mels + 1) as f64;
        *slot = mel_to_hz(mel);
    }
    let mut weights = vec![0.0f32; n_mels * n_freq];
    for bin in 0..n_mels {
        let left = mel_f[bin + 1] - mel_f[bin];
        let right = mel_f[bin + 2] - mel_f[bin + 1];
        let scale = 2.0 / (mel_f[bin + 2] - mel_f[bin]);
        for freq in 0..n_freq {
            let hz = freq as f64 * sample_rate / n_fft as f64;
            let lower = (hz - mel_f[bin]) / left;
            let upper = (mel_f[bin + 2] - hz) / right;
            weights[bin * n_freq + freq] = (lower.min(upper).max(0.0) * scale) as f32;
        }
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_is_padded_and_long_audio_is_cut() {
        let short = prepare_waveform(&[0.0; 10]);
        assert_eq!(short.len(), MIN_SAMPLES);
        let long = prepare_waveform(&vec![0.1; MAX_SAMPLES + 50]);
        assert_eq!(long.len(), MAX_SAMPLES);
    }

    #[test]
    fn hann_ends_are_zero() {
        let window = hann();
        assert_eq!(window[0], 0.0);
        assert_eq!(window[N_FFT - 1], 0.0);
        assert!(window[N_FFT / 2] > 0.9);
    }
}
