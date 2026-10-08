//! WAV and PCM loading. Samples are mono f32. Anything that is not 16 kHz is
//! resampled with linear interpolation when a decision runs. Stereo and other
//! multi-channel files are averaged. 16-bit integer samples are divided by
//! 32768; other integer widths use `2^(bits-1)`.

use std::path::Path;

use crate::mel::SAMPLE_RATE;
use crate::Error;

/// One clip. `samples` are floating amplitudes, nominally in `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct AudioClip {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioClip {
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32) -> Result<Self, Error> {
        validate(&samples, sample_rate)?;
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Signed 16-bit PCM. Each sample is divided by 32768.
    pub fn from_pcm16(samples: &[i16], sample_rate: u32) -> Result<Self, Error> {
        let samples = samples
            .iter()
            .map(|sample| *sample as f32 / 32768.0)
            .collect();
        Self::from_samples(samples, sample_rate)
    }

    pub fn from_wav(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let mut reader = hound::WavReader::open(path).map_err(|err| Error::Audio {
            message: format!("read {}: {err}", path.display()),
        })?;
        let spec = reader.spec();
        if spec.channels == 0 || spec.sample_rate == 0 {
            return Err(Error::Audio {
                message: format!("{} has an empty WAV header", path.display()),
            });
        }
        let channels = spec.channels as usize;
        let mono = match spec.sample_format {
            hound::SampleFormat::Float => {
                let decoded = reader
                    .samples::<f32>()
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| Error::Audio {
                        message: format!("decode {}: {err}", path.display()),
                    })?;
                downmix(&decoded, channels)
            }
            hound::SampleFormat::Int => {
                let decoded = reader
                    .samples::<i32>()
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| Error::Audio {
                        message: format!("decode {}: {err}", path.display()),
                    })?;
                let scale = int_scale(spec.bits_per_sample);
                downmix(
                    &decoded
                        .into_iter()
                        .map(|sample| sample as f32 / scale)
                        .collect::<Vec<_>>(),
                    channels,
                )
            }
        };
        Self::from_samples(mono, spec.sample_rate)
    }

    /// Linear resample to 16 kHz. 16 kHz clips are copied.
    pub fn at_16k(&self) -> Vec<f32> {
        if self.sample_rate == SAMPLE_RATE {
            self.samples.clone()
        } else {
            linear_resample(&self.samples, self.sample_rate, SAMPLE_RATE)
        }
    }
}

fn validate(samples: &[f32], sample_rate: u32) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::Audio {
            message: "sample rate must be greater than 0".into(),
        });
    }
    if samples.is_empty() {
        return Err(Error::Audio {
            message: "audio clip is empty".into(),
        });
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(Error::Audio {
            message: "audio samples must be finite".into(),
        });
    }
    Ok(())
}

fn int_scale(bits: u16) -> f32 {
    let shift = bits.saturating_sub(1).min(31);
    (1u32 << shift) as f32
}

fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

pub fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let out_len = (input.len() as f64 * to_rate as f64 / from_rate as f64).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }
    let step = from_rate as f64 / to_rate as f64;
    let mut out = Vec::with_capacity(out_len);
    let last = input.len() - 1;
    for index in 0..out_len {
        let position = index as f64 * step;
        let left = (position.floor() as usize).min(last);
        let right = (left + 1).min(last);
        let frac = (position - left as f64) as f32;
        out.push(input[left] + (input[right] - input[left]) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm16_uses_the_reference_divisor() {
        let clip = AudioClip::from_pcm16(&[16384, -16384], 16_000).unwrap();
        assert!((clip.samples[0] - 0.5).abs() < 1e-6);
        assert!((clip.samples[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn wav_round_trip_downmixes_stereo() {
        let path = std::env::temp_dir().join("fuzzy-decision-stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        writer.write_sample(16_384i16).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.finalize().unwrap();
        let clip = AudioClip::from_wav(&path).unwrap();
        assert_eq!(clip.sample_rate, 8_000);
        assert_eq!(clip.samples.len(), 1);
        assert!((clip.samples[0] - 0.25).abs() < 1e-4);
        let resampled = clip.at_16k();
        assert_eq!(resampled.len(), 2);
    }

    #[test]
    fn empty_audio_is_rejected() {
        let err = AudioClip::from_samples(Vec::new(), 16_000).unwrap_err();
        assert!(matches!(err, Error::Audio { .. }));
    }
}
