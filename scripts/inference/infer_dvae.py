import torch
import torchaudio
import numpy as np
import os
import argparse
from pathlib import Path
import soundfile as sf
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa

from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

def load_trained_dvae(checkpoint_path: str, device: str = 'cuda') -> DiscreteVAE:
    """Load the trained DVAE model from checkpoint."""
    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=1024,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    
    # Load the trained weights with strict=False to ignore missing keys
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dvae.load_state_dict(checkpoint, strict=False)  # Add strict=False here
    dvae.to(device)
    dvae.eval()
    
    print(f"Loaded trained DVAE from {checkpoint_path}")
    return dvae

def mel_to_audio_librosa(mel_spec: torch.Tensor, 
                        sr: int = 22050, 
                        n_fft: int = 1024, 
                        hop_length: int = 256, 
                        n_iter: int = 32) -> torch.Tensor:
    """Convert mel spectrogram back to audio using librosa Griffin-Lim algorithm."""
    # Convert torch tensor to numpy array
    mel_spec_np = mel_spec.cpu().numpy()
    
    # Ensure correct shape (n_mels, time_frames)
    if mel_spec_np.ndim == 3:
        mel_spec_np = mel_spec_np.squeeze(0)  # Remove batch dimension if present
    
    # Convert mel spectrogram to audio using librosa
    audio = librosa.feature.inverse.mel_to_audio(
        M=mel_spec_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter,
        power=2.0  # Assuming power spectrogram
    )
    
    # Convert back to torch tensor
    return torch.from_numpy(audio)




def compute_reconstruction_metrics(original_mel: torch.Tensor, 
                                 reconstructed_mel: torch.Tensor) -> dict:
    """Compute various metrics to evaluate reconstruction quality."""
    # Mean Squared Error
    mse = torch.nn.functional.mse_loss(reconstructed_mel, original_mel).item()
    
    # Mean Absolute Error
    mae = torch.nn.functional.l1_loss(reconstructed_mel, original_mel).item()
    
    # Signal-to-Noise Ratio
    signal_power = torch.mean(original_mel ** 2)
    noise_power = torch.mean((original_mel - reconstructed_mel) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'snr_db': snr
    }

def plot_mel_comparison(original_mel: torch.Tensor, 
                       reconstructed_mel: torch.Tensor, 
                       save_path: str):
    """Plot original vs reconstructed mel spectrograms for visual comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original
    axes[0].imshow(original_mel.cpu().numpy(), aspect='auto', origin='lower')
    axes[0].set_title('Original Mel Spectrogram')
    axes[0].set_ylabel('Mel Bins')
    
    # Reconstructed
    axes[1].imshow(reconstructed_mel.cpu().numpy(), aspect='auto', origin='lower')
    axes[1].set_title('Reconstructed Mel Spectrogram')
    axes[1].set_ylabel('Mel Bins')
    axes[1].set_xlabel('Time Frames')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def inference_single_audio(audio_path: str,
                          dvae: DiscreteVAE,
                          mel_transform: TorchMelSpectrogram,
                          output_dir: str,
                          device: str = 'cuda') -> dict:
    """Run inference on a single audio file."""
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != 22050:
        resampler = torchaudio.transforms.Resample(sample_rate, 22050)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Move to device
    waveform = waveform.to(device)
    
    # Convert to mel spectrogram
    with torch.no_grad():
        mel_spec = mel_transform(waveform)
        
        # Ensure mel spectrogram is divisible by 4 for DVAE
        remainder = mel_spec.shape[-1] % 4
        if remainder:
            mel_spec = mel_spec[:, :, :-remainder]
        
        # Add batch dimension if needed
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        # DVAE forward pass
        recon_loss, commitment_loss, reconstructed_mel = dvae(mel_spec)
        
        # Remove batch dimension
        original_mel = mel_spec.squeeze(0)
        reconstructed_mel = reconstructed_mel.squeeze(0)
        
        # Compute metrics
        metrics = compute_reconstruction_metrics(original_mel, reconstructed_mel)
        metrics['commitment_loss'] = commitment_loss.item()
        metrics['reconstruction_loss'] = recon_loss.mean().item()
        
        # Convert back to audio using librosa
        try:
            reconstructed_audio = mel_to_audio_librosa(reconstructed_mel)
            original_audio_from_mel = mel_to_audio_librosa(original_mel)
        except Exception as e:
            print(f"Warning: Could not convert mel to audio: {e}")
            reconstructed_audio = None
            original_audio_from_mel = None
        
        # Save results
        audio_name = Path(audio_path).stem
        
        # Save mel spectrogram comparison plot
        plot_path = os.path.join(output_dir, f"{audio_name}_mel_comparison.png")
        plot_mel_comparison(original_mel, reconstructed_mel, plot_path)
        
        # Save audio files if conversion was successful
        if reconstructed_audio is not None:
            # Save reconstructed audio
            recon_audio_path = os.path.join(output_dir, f"{audio_name}_reconstructed.wav")
            sf.write(recon_audio_path, reconstructed_audio.cpu().numpy(), 22050)
            
            # Save original audio converted from mel (for fair comparison)
            orig_from_mel_path = os.path.join(output_dir, f"{audio_name}_original_from_mel.wav")
            sf.write(orig_from_mel_path, original_audio_from_mel.cpu().numpy(), 22050)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="DVAE Inference Script")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained DVAE checkpoint")
    parser.add_argument("--mel_norm_file", type=str, required=True,
                       help="Path to mel normalization stats file")
    parser.add_argument("--input_audio", type=str,
                       help="Path to single audio file for testing")
    parser.add_argument("--input_dir", type=str,
                       help="Directory containing audio files for batch testing")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained DVAE
    dvae = load_trained_dvae(args.checkpoint_path, args.device)
    
    # Initialize mel spectrogram transform
    mel_transform = TorchMelSpectrogram(
        mel_norm_file=args.mel_norm_file,
        sampling_rate=22050
    ).to(args.device)
    
    # Collect audio files
    audio_files = []
    if args.input_audio:
        audio_files = [args.input_audio]
    elif args.input_dir:
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = [
            str(p) for p in Path(args.input_dir).rglob('*')
            if p.suffix.lower() in audio_extensions
        ]
    else:
        raise ValueError("Please provide either --input_audio or --input_dir")
    
    # Run inference
    all_metrics = []
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            metrics = inference_single_audio(
                audio_file, dvae, mel_transform, args.output_dir, args.device
            )
            metrics['filename'] = Path(audio_file).name
            all_metrics.append(metrics)
            
            print(f"\n{Path(audio_file).name}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  SNR: {metrics['snr_db']:.2f} dB")
            print(f"  Reconstruction Loss: {metrics['reconstruction_loss']:.6f}")
            print(f"  Commitment Loss: {metrics['commitment_loss']:.6f}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    # Compute and save overall statistics
    if all_metrics:
        avg_mse = np.mean([m['mse'] for m in all_metrics])
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_snr = np.mean([m['snr_db'] for m in all_metrics])
        avg_recon_loss = np.mean([m['reconstruction_loss'] for m in all_metrics])
        avg_commit_loss = np.mean([m['commitment_loss'] for m in all_metrics])
        
        print(f"\n=== Overall Statistics ===")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Average MAE: {avg_mae:.6f}")
        print(f"Average SNR: {avg_snr:.2f} dB")
        print(f"Average Reconstruction Loss: {avg_recon_loss:.6f}")
        print(f"Average Commitment Loss: {avg_commit_loss:.6f}")
        
        # Save detailed results
        import json
        results_file = os.path.join(args.output_dir, "inference_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'overall_stats': {
                    'avg_mse': avg_mse,
                    'avg_mae': avg_mae,
                    'avg_snr_db': avg_snr,
                    'avg_reconstruction_loss': avg_recon_loss,
                    'avg_commitment_loss': avg_commit_loss
                },
                'per_file_results': all_metrics
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Visual comparisons and audio files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
